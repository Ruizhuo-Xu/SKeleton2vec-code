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
import wandb

import datasets
import models
import utils


def make_data_loader(spec, tag=''):
    if spec is None:
        return None
    
    dataset = datasets.make(spec['dataset'])

    log('{} dataset: size={}'.format(tag, len(dataset)))
    for k, v in dataset[0].items():
        log('  {}: shape={}'.format(k, tuple(v.shape)))

    loader = DataLoader(
        dataset,
        batch_size=spec['batch_size'],
        shuffle=(tag == 'train'),num_workers=spec.get('num_workers', 0),
        pin_memory=True)
    return loader


def make_data_loaders():
    train_loader = make_data_loader(config.get('train_dataset'), tag='train')
    val_loader = make_data_loader(config.get('val_dataset'), tag='val')
    return train_loader, val_loader

    
def prepare_training():
    if config.get('resume') is not None:
        sv_file = torch.load(config['resume'])
        model = models.make(sv_file['model'], load_sd=True).cuda()
        optimizer = utils.make_optimizer(
            model.parameters(), sv_file['optimizer'], load_sd=True)
        epoch_start = sv_file['epoch'] + 1
        if config.get('multi_step_lr') is None:
            lr_scheduler = None
        else:
            lr_scheduler = MultiStepLR(optimizer, **config['multi_step_lr'])
        for _ in range(epoch_start - 1):
            lr_scheduler.step()
    else:
        model = models.make(config['model']).cuda()
        optimizer = utils.make_optimizer(
            model.parameters(), config['optimizer'])
        epoch_start = 1
        if config.get('multi_step_lr') is None:
            lr_scheduler = None
        else:
            lr_scheduler = MultiStepLR(optimizer, **config['multi_step_lr'])

    log('model: #params={}'.format(utils.compute_num_params(model, text=True)))
    return model, optimizer, epoch_start, lr_scheduler


def train(train_loader, model, optimizer):
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    train_loss = utils.Averager()
    train_acc = utils.Accuracy()

    for batch in tqdm(train_loader, leave=False, desc='train'):
        for k, v in batch.items():
            batch[k] = v.cuda()

        inp = batch['keypoint']
        # num_clips == 1
        assert inp.shape[1] == 1
        inp = inp[:, 0]
        labels = batch['label'].squeeze(-1)
        logits = model(inp)
        # pdb.set_trace()
        loss = loss_fn(logits, labels)
        preds = logits.argmax(dim=1)
        train_loss.add(loss.item())
        train_acc.add(preds, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        preds = None; loss = None

    return train_loss.item(), train_acc.item()


def validate(val_loader, model):
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    val_loss = utils.Averager()
    val_acc = utils.Accuracy()

    with torch.no_grad():
        for batch in tqdm(val_loader, leave=False, desc='val'):
            for k, v in batch.items():
                batch[k] = v.cuda()
            inp = batch['keypoint']
            # num_clips == 1
            assert inp.shape[1] == 1
            inp = inp[:, 0]
            labels = batch['label'].squeeze(-1)
            logits = model(inp)
            loss = loss_fn(logits, labels)
            preds = logits.argmax(dim=1)
            val_loss.add(loss.item())
            val_acc.add(preds, labels)
            preds = None; loss = None

    return val_loss.item(), val_acc.item()
        


def main(config_, save_path):
    global config, log
    config = config_
    log = utils.set_save_path(save_path)
    with open(os.path.join(save_path, 'config.yaml'), 'w') as f:
        yaml.dump(config, f, sort_keys=False)

    train_loader, val_loader = make_data_loaders()
    model, optimizer, epoch_start, lr_scheduler = prepare_training()

    epoch_max = config['epoch_max']
    epoch_val = config.get('epoch_val')
    epoch_save = config.get('epoch_save')
    max_val_v = -1e18

    timer = utils.Timer()

    for epoch in range(epoch_start, epoch_max + 1):
        t_epoch_start = timer.t()
        log_info = ['epoch {}/{}'.format(epoch, epoch_max)]
        train_loss, train_acc = train(train_loader, model, optimizer)
        if lr_scheduler is not None:
            lr_scheduler.step()

        log_info.append(f'train: loss={train_loss:.4f}')
        log_info.append(f'train: acc={train_acc:.4f}')
        wandb.log({'train/loss': train_loss, 'train/acc': train_acc}, epoch)

        model_spec = config['model']
        model_spec['sd'] = model.state_dict()
        optimizer_spec = config['optimizer']
        optimizer_spec['sd'] = optimizer.state_dict()
        sv_file = {
            'model': model_spec,
            'optimizer': optimizer_spec,
            'epoch': epoch
        }

        torch.save(sv_file, os.path.join(save_path, 'epoch-last.pth'))
        if (epoch_save is not None) and (epoch % epoch_save == 0):
            torch.save(sv_file,
                os.path.join(save_path, 'epoch-{}.pth'.format(epoch)))

        if (epoch_val is not None) and (epoch % epoch_val == 0):
            val_loss, val_acc = validate(val_loader, model)
            log_info.append(f'val: loss={val_loss:.4f}')
            log_info.append(f'val: acc={val_acc:.4f}')
            wandb.log({'val/loss': val_loss, 'val/acc': val_acc}, epoch)
            if val_acc > max_val_v:
                max_val_v = val_acc
                torch.save(sv_file, os.path.join(save_path, 'epoch-best.pth'))

        t = timer.t()
        prog = (epoch - epoch_start + 1) / (epoch_max - epoch_start + 1)
        t_epoch = utils.time_text(t - t_epoch_start)
        t_elapsed, t_all = utils.time_text(t), utils.time_text(t / prog)
        log_info.append('{} {}/{}'.format(t_epoch, t_elapsed, t_all))

        log(', '.join(log_info))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--name', default=None)
    parser.add_argument('--tag', default=None)
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        print('config loaded.')

    save_name = args.name
    if save_name is None:
        save_name = '_' + args.config.split('/')[-1][:-len('.yaml')]
    if args.tag is not None:
        save_name += '_' + args.tag
    save_path = os.path.join('./save', save_name)
    wandb.init(project='Skeleton2vec', name=save_name)

    main(config, save_path)
