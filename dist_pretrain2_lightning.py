import argparse
import os
import math
import pdb
import sys
import random

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
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
import wandb
from timm.loss import LabelSmoothingCrossEntropy
import timm.optim.optim_factory as optim_factory

from datasets import datasets
from models import models
import utils


def make_data_loader(spec, tag=''):
    if spec is None:
        return None
    
    dataset = datasets.make(spec['dataset'])
    # dataset = datasets.make(spec['pretrain'], args={'dataset': dataset})

    # log('{} dataset: size={}'.format(tag, len(dataset)))
    # for k, v in dataset[0].items():
    #     log('  {}: shape={}'.format(k, tuple(v.shape)))

    # loader = utils.MultiEpochsDataLoader(
    #     dataset,
    #     batch_size=spec['batch_size'],
    #     shuffle=False,
    #     num_workers=spec.get('num_workers', 0),
    #     pin_memory=True,
    #     sampler=DistributedSampler(dataset)
    # )
    loader = DataLoader( dataset,
        batch_size=spec['batch_size'],
        shuffle=True,
        num_workers=spec.get('num_workers', 0),
        pin_memory=True,
        # sampler=DistributedSampler(dataset)
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
        model = models.make(sv_file['model'], load_sd=True)
        optimizer = utils.make_optimizer(
            model.parameters(), sv_file['optimizer'], load_sd=True)
        epoch_start = sv_file['epoch'] + 1
        if config.get('lr_scheduler') is None:
            lr_scheduler = None
        else:
            lr_scheduler = utils.make_lr_scheduler(optimizer, config['lr_scheduler'])
            for _ in range(epoch_start - 1):
                lr_scheduler.step()
        loss_scaler = utils.NativeScalerWithGradNormCount()
        if sv_file.get('scaler'):
            loss_scaler.load_state_dict(sv_file['scaler'])
    else:
        model = models.make(config['model'])

        # following timm: set wd as 0 for bias and norm layers
        if config.get('weight_decay_groups', False):
            weight_decay = config['optimizer']['args'].get('weight_decay', 0)
            no_weight_decay_list = model.no_weight_decay()
            param_groups = optim_factory.param_groups_weight_decay(model, weight_decay, no_weight_decay_list)
            config['optimizer']['args'].pop('weight_decay')
        else:
            param_groups = model.parameters()

        optimizer = utils.make_optimizer(
            param_groups, config['optimizer'])

        epoch_start = 1
        if config.get('lr_scheduler') is None:
            lr_scheduler = None
        else:
            lr_scheduler = utils.make_lr_scheduler(optimizer, config['lr_scheduler'])
        loss_scaler = utils.NativeScalerWithGradNormCount()

    # log('model: #params={}'.format(utils.compute_num_params(model, text=True)))
    # log(model)
    # log(optimizer)
    return model, optimizer, epoch_start, lr_scheduler, loss_scaler

# define the LightningModule
class LitSkeleton2vec(pl.LightningModule):
    def __init__(self, model, optimizer, lr_scheduler,
                 loss_scaler, train_loader_length):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.loss_scaler = loss_scaler
        self.train_loader_length = train_loader_length
        # self.automatic_optimization = False

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        if (batch_idx != 0 and self.current_epoch == 0) or self.current_epoch != 0:
            self.model.ema_step()

        grad_accum_steps = config.get('grad_accum_steps', 1)
        mask_ratio = config.get('mask_ratio', 0.8)
        tube_len = config.get('tube_len', 6)
        num_masked_views = config.get('num_masked_views', 1)
        clip_grad = config.get('clip_grad')
        motion_loss_weight = config.get('motion_loss_weight', 1.0)
        s_tau = config.get('s_tau', 0.2)
        t_tau = config.get('t_tau', 0.2)
        self.lr_scheduler.step((batch_idx + 1) / self.train_loader_length + self.current_epoch)
        src = batch['keypoint']
        motion = batch.get('motion')
        losses = self.model(src, mask_ratio=mask_ratio,
                        tube_len=tube_len,
                        num_masked_views=num_masked_views,
                        motion=motion, s_tau=s_tau, t_tau=t_tau)
        loss = None
        loss_feat = losses.get('feat', torch.tensor(0))
        # loss_motion = losses.get('motion', torch.tensor(0))
        # loss = loss_fn(x.float(), y.float())
        loss = loss_feat
        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()), force=True)
            sys.exit(1)

        # update_grad = (batch_idx + 1) % grad_accum_steps == 0
        # grad_norm = self.loss_scaler(loss, self.optimizer,
        #                         parameters=self.model.parameters(),
        #                         clip_grad=clip_grad,
        #                         update_grad=update_grad)
        # if update_grad:
        #     self.optimizer.zero_grad()
        #     # grad_norm_rec.append(grad_norm.item())
        #     # EMA update
        #     self.model.module.ema_step()
        ema_decay = self.model.ema.decay
        current_lr = self.optimizer.param_groups[0]['lr']
        # import pdb; pdb.set_trace()

        # Logging to TensorBoard (if installed) by default
        self.log("train/loss", loss, prog_bar=True,
                 logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log("train/lr", current_lr, prog_bar=True,
                 logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log("train/ema", ema_decay, prog_bar=True,
                 logger=True, on_step=True)
        return loss

    def configure_optimizers(self):
        optimizer = self.optimizer
        return optimizer


def main(config_, save_path, args):
    global config, log
    # rank = args.node_rank * gpus + local_rank
    # world_size = args.nodes * gpus
    # rank, world_size = ddp_setup(local_rank, gpus, args)
    config = config_
    # if rank == 0:
    #     save_name = save_path.split('/')[-1]
    #     wandb.init(entity='ruizhuo_xu', project='Skeleton2vec', name=save_name)
    #     log = utils.set_save_path(save_path)
    #     log(f'nodes: {args.nodes}, world_size: {world_size}')
    #     with open(os.path.join(save_path, 'config.yaml'), 'w') as f:
    #         yaml.dump(config, f, sort_keys=False)
    # dist.barrier()
    save_name = save_path.split('/')[-1]
    # wandb.init(entity='ruizhuo_xu', project='Skeleton2vec', name=save_name)
    # log = utils.set_save_path(save_path)
    # with open(os.path.join(save_path, 'config.yaml'), 'w') as f:
    #     yaml.dump(config, f, sort_keys=False)

    # torch.cuda.set_device(rank)
    train_loader, val_loader = make_data_loaders()
    model, optimizer, epoch_start, lr_scheduler, loss_scaler = prepare_training()
    lit_model = LitSkeleton2vec(model, optimizer,
                                lr_scheduler, loss_scaler,
                                len(train_loader))

    """
    ckp = 'save/nturgbd120_xset_pretrain_skt2vec2_64BSZ_5D_1e-3baseLR_1e-5minLR_tube5_tau0.2_ema9999_800EP_/epoch-epoch=199.ckpt'
    lit_model = LitSkeleton2vec.load_from_checkpoint(ckp, model=model, optimizer=optimizer,
                                                     lr_scheduler=lr_scheduler, loss_scaler=loss_scaler,
                                                     train_loader_length=len(train_loader),)
    model_spec = config['model']
    model_spec['sd'] = lit_model.model.state_dict()
    sv_file = {'model': model_spec}
    torch.save(sv_file, os.path.join(save_path, 'epoch-200.pth'))
    """

    if args.compile:
        lit_model = torch.compile(lit_model)
    max_epochs = config['epoch_max']
    grad_clip = config.get('clip_grad')
    # epoch_save = config.get('epoch_save')
    wandb_logger = WandbLogger(project='Skeleton2vec', name=save_name, entity='ruizhuo_xu',)
    checkpoint_callback = ModelCheckpoint(dirpath=save_path,
                                          filename='epoch-{epoch}',
                                          save_last=True,
                                          every_n_epochs=100)
    trainer = pl.Trainer(accelerator='gpu',
                         precision = 'bf16-mixed',
                         max_epochs=max_epochs,
                         default_root_dir=save_path,
                         callbacks=[checkpoint_callback],
                        #  strategy='fsdp',
                         logger=wandb_logger,
                         benchmark=True,
                         gradient_clip_val=grad_clip)
    trainer.fit(model=lit_model, train_dataloaders=train_loader)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--name', default=None)
    parser.add_argument('--tag', default=None)
    parser.add_argument('--gpu', default='0')
    # DDP args
    parser.add_argument('--addr', default='localhost')
    parser.add_argument('--port', default='12355')
    parser.add_argument('--nodes', type=int, default=1)
    parser.add_argument('--node_rank', type=int, default=0)

    parser.add_argument('--enable_amp', action='store_true', default=False,
                        help='Enabling automatic mixed precision')
    parser.add_argument('--compile', action='store_true', default=False,
                        help='Enabling torch.Compile')
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
    torch.set_float32_matmul_precision('high')
    
    main(config, save_path, args)

