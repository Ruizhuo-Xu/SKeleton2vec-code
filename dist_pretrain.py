import argparse
import os
import math
import pdb

import yaml
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import wandb
from timm.loss import LabelSmoothingCrossEntropy

from datasets import datasets
from models import models
import utils


def ddp_setup(rank, world_size, port='12355'):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = port
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def ddp_reduce(x):
    res  = torch.tensor(x).cuda()
    dist.reduce(res, 0)
    return res.item()


def make_data_loader(spec, tag=''):
    if spec is None:
        return None
    
    dataset = datasets.make(spec['dataset'])
    dataset = datasets.make(spec['pretrain'], args={'dataset': dataset})

    if dist.get_rank() == 0:
        log('{} dataset: size={}'.format(tag, len(dataset)))
        for k, v in dataset[0].items():
            log('  {}: shape={}'.format(k, tuple(v.shape)))

    loader = DataLoader(
        dataset,
        batch_size=spec['batch_size'],
        shuffle=False,
        num_workers=spec.get('num_workers', 0),
        pin_memory=True,
        sampler=DistributedSampler(dataset)
    )
    return loader


def make_data_loaders():
    train_loader = make_data_loader(config.get('train_dataset'), tag='train')
    val_loader = make_data_loader(config.get('val_dataset'), tag='val')
    return train_loader, val_loader

    
def prepare_training():
    if config.get('resume') is not None:
        if dist.get_rank() == 0:
            log('resume from the ckp: ' + config['resume'])
        sv_file = torch.load(config['resume'])
        model = models.make(sv_file['model'], load_sd=True).cuda()
        optimizer = utils.make_optimizer(
            model.parameters(), sv_file['optimizer'], load_sd=True)
        epoch_start = sv_file['epoch'] + 1
        if config.get('lr_scheduler') is None:
            lr_scheduler = None
        else:
            lr_scheduler = utils.make_lr_scheduler(optimizer, config['lr_scheduler'])
            for _ in range(epoch_start - 1):
                lr_scheduler.step()
    else:
        model = models.make(config['model']).cuda()
        optimizer = utils.make_optimizer(
            model.parameters(), config['optimizer'])
        epoch_start = 1
        if config.get('lr_scheduler') is None:
            lr_scheduler = None
        else:
            lr_scheduler = utils.make_lr_scheduler(optimizer, config['lr_scheduler'])

    if dist.get_rank() == 0:
        log('model: #params={}'.format(utils.compute_num_params(model, text=True)))
        log(model)
    return model, optimizer, epoch_start, lr_scheduler


def train(train_loader, model, optimizer, epoch=None, lr_scheduler=None):
    model.train()
    if config.get('smooth_l1_beta'):
        beta = config['smooth_l1_beta']
        loss_fn = nn.SmoothL1Loss(reduction='none', beta=beta)
        if dist.get_rank() == 0 and epoch == 0:
            log(f'Using SmoothL1 Loss, beta:{beta}')
    else:
        loss_fn = nn.MSELoss(reduction='none')
    train_loss = utils.Averager()

    with tqdm(train_loader,leave=False, desc='train') as t:
        for iter_step, batch in enumerate(t):
            if lr_scheduler is not None and lr_scheduler.mode == 'step':
                lr_scheduler.step(iter_step / len(train_loader) + epoch)
            for k, v in batch.items():
                batch[k] = v.cuda()

            src = batch['keypoint_mask']
            trg = batch['keypoint']
            mask = batch['mask']
            # num_clips == 1
            assert src.shape[1] == 1 and trg.shape[1] == 1 and mask.shape[1] == 1
            src = src[:, 0]
            trg = trg[:, 0]
            mask = mask[:, 0]
            x, y = model(src, trg, mask)
            # pdb.set_trace()
            # loss = loss_fn(x.float(), y.float()).sum(dim=-1).sum().div(x.size(0))
            loss = loss_fn(x.float(), y.float()).sum(dim=-1).mean()
            train_loss.add(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # EMA update
            model.module.ema_step()
            ema_decay = model.module.ema.decay

            current_lr = optimizer.param_groups[0]['lr']
            tqdm.set_postfix(t, {'loss': train_loss.item(),
                                 'lr': current_lr,
                                 'ema_decay': ema_decay})

            x = None; y = None; loss = None

    return train_loss

@torch.no_grad()
def validate(val_loader, model):
    model.eval()
    if config.get('smooth_l1_beta'):
        beta = config['smooth_l1_beta']
        loss_fn = nn.SmoothL1Loss(reduction='none', beta=beta)
    else:
        loss_fn = nn.MSELoss(reduction='none')
    val_loss = utils.Averager()

    with tqdm(val_loader, leave=False, desc='val') as t:
        for batch in t:
            for k, v in batch.items():
                batch[k] = v.cuda()

            src = batch['keypoint_mask']
            trg = batch['keypoint']
            mask = batch['mask']
            # num_clips == 1
            assert src.shape[1] == 1 and trg.shape[1] == 1 and mask.shape[1] == 1
            src = src[:, 0]
            trg = trg[:, 0]
            mask = mask[:, 0]
            x, y = model(src, trg, mask)
            # pdb.set_trace()
            loss = loss_fn(x.float(), y.float()).sum(dim=-1).sum().div(x.size(0))
            val_loss.add(loss.item())

            x = None; y = None; loss = None
            tqdm.set_postfix(t, {
                'loss': val_loss.item()})

    return val_loss
        


def main(rank, world_size, config_, save_path, port='12355'):
    global config, log
    ddp_setup(rank, world_size, port)
    config = config_
    if rank == 0:
        save_name = save_path.split('/')[-1]
        wandb.init(entity='ruizhuo_xu', project='Skeleton2vec', name=save_name)
        log = utils.set_save_path(save_path)
        with open(os.path.join(save_path, 'config.yaml'), 'w') as f:
            yaml.dump(config, f, sort_keys=False)
    dist.barrier()

    torch.cuda.set_device(rank)
    train_loader, val_loader = make_data_loaders()
    model, optimizer, epoch_start, lr_scheduler = prepare_training()
    model = model.cuda()
    model = DDP(model, device_ids=[rank], output_device=rank)

    epoch_max = config['epoch_max']
    epoch_val = config.get('epoch_val')
    epoch_save = config.get('epoch_save')
    min_val_v = 1e8

    timer = utils.Timer()

    for epoch in range(epoch_start, epoch_max + 1):
        t_epoch_start = timer.t()
        log_info = ['epoch {}/{}'.format(epoch, epoch_max)]
        train_loader.sampler.set_epoch(epoch)
        train_loss = train(train_loader, model, optimizer,
                                      epoch - 1, lr_scheduler)
        current_lr = optimizer.param_groups[0]['lr']
        if lr_scheduler is not None and lr_scheduler.mode == 'epoch':
            lr_scheduler.step()
        v = ddp_reduce(train_loss.v)
        n = ddp_reduce(train_loss.n)
        if rank == 0:
            # print(v, n, correct_num, total_num)
            train_loss = (v / n)
            log_info.append(f'train: loss={train_loss:.4f}')
            wandb.log({'train/loss': train_loss}, epoch)
            wandb.log({'train/lr': current_lr}, epoch)

        model_spec = config['model']
        model_spec['sd'] = model.module.state_dict()
        optimizer_spec = config['optimizer']
        optimizer_spec['sd'] = optimizer.state_dict()
        sv_file = {
            'model': model_spec,
            'optimizer': optimizer_spec,
            'epoch': epoch
        }

        if rank == 0:
            torch.save(sv_file, os.path.join(save_path, 'epoch-last.pth'))
            if (epoch_save is not None) and (epoch % epoch_save == 0):
                torch.save(sv_file,
                    os.path.join(save_path, 'epoch-{}.pth'.format(epoch)))

        if (epoch_val is not None) and (epoch % epoch_val == 0):
            val_loss = validate(val_loader, model)
            v = ddp_reduce(val_loss.v)
            n = ddp_reduce(val_loss.n)
            if rank == 0:
                # print(v, n, correct_num, total_num)
                val_loss = (v / n)
                log_info.append(f'val: loss={val_loss:.4f}')
                wandb.log({'val/loss': val_loss}, epoch)
                if val_loss < min_val_v:
                    min_val_v = val_loss
                    torch.save(sv_file, os.path.join(save_path, 'epoch-best.pth'))

        t = timer.t()
        prog = (epoch - epoch_start + 1) / (epoch_max - epoch_start + 1)
        t_epoch = utils.time_text(t - t_epoch_start)
        t_elapsed, t_all = utils.time_text(t), utils.time_text(t / prog)
        if rank == 0:
            log_info.append('{} {}/{}'.format(t_epoch, t_elapsed, t_all))
            log(', '.join(log_info))
    destroy_process_group()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--name', default=None)
    parser.add_argument('--tag', default=None)
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--port', default='12355')
    args = parser.parse_args()

    # os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        print('config loaded.')

    save_name = args.name
    if save_name is None:
        save_name = '_' + args.config.split('/')[-1][:-len('.yaml')]
    if args.tag is not None:
        save_name += '_' + args.tag
    save_path = os.path.join('./save', save_name)

    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size, config, save_path, args.port), nprocs=world_size)

