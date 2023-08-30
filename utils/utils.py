import os
import time
import random
import shutil
import sys
import math

import torch
from torch.optim import SGD, Adam, AdamW
import numpy as np
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import MultiStepLR


class Averager():

    def __init__(self):
        self.n = 0.0
        self.v = 0.0

    def add(self, v, n=1.0):
        # self.v = (self.v * self.n + v * n) / (self.n + n)
        self.v += v * n
        self.n += n

    def item(self):
        return self.v / self.n


class Accuracy():

    def __init__(self):
        self.correct_num = 0
        self.total_num = 0
        self.acc = 0.0

    def add(self, preds: torch.Tensor, labels: torch.Tensor):
        assert preds.shape[0] == labels.shape[0]
        correct_num = (preds == labels).sum().item()
        total_num = preds.shape[0]
        self.correct_num += correct_num
        self.total_num += total_num
        self.acc = self.correct_num / self.total_num

    def item(self):
        return self.acc


class Timer():

    def __init__(self):
        self.v = time.time()

    def s(self):
        self.v = time.time()

    def t(self):
        return time.time() - self.v


def time_text(t):
    if t >= 3600:
        return '{:.1f}h'.format(t / 3600)
    elif t >= 60:
        return '{:.1f}m'.format(t / 60)
    else:
        return '{:.1f}s'.format(t)


_log_path = None


def set_log_path(path):
    global _log_path
    _log_path = path


def log(obj, filename='log.txt'):
    print(obj)
    if _log_path is not None:
        with open(os.path.join(_log_path, filename), 'a') as f:
            print(obj, file=f)


def ensure_path(path, remove=True):
    basename = os.path.basename(path.rstrip('/'))
    if os.path.exists(path):
        sys.stdin = os.fdopen(0, "r")  # 打开标准输入流
        sys.stdout = os.fdopen(1, "w")  # 打开标准输出流
        if remove and (basename.startswith('_')
                or input('{} exists, remove? (y/[n]): '.format(path)) == 'y'):
            shutil.rmtree(path)
            os.makedirs(path)
    else:
        os.makedirs(path)


def set_save_path(save_path, remove=True):
    ensure_path(save_path, remove=remove)
    set_log_path(save_path)
    return log


def compute_num_params(model, text=False):
    tot = int(sum([np.prod(p.shape) for p in model.parameters()]))
    if text:
        if tot >= 1e6:
            return '{:.1f}M'.format(tot / 1e6)
        else:
            return '{:.1f}K'.format(tot / 1e3)
    else:
        return tot


def make_optimizer(param_list, optimizer_spec, load_sd=False):
    Optimizer = {
        'sgd': SGD,
        'adam': Adam,
        'adamw': AdamW,
    }[optimizer_spec['name']]
    optimizer = Optimizer(param_list, **optimizer_spec['args'])
    if load_sd:
        optimizer.load_state_dict(optimizer_spec['sd'])
    return optimizer

    
class CosineDecayWithWarmup(lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_epochs, max_epochs, base_lr, min_lr=0):
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.base_lr = base_lr
        self.min_lr = min_lr
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            return [self.base_lr * (1 + self.last_epoch) / self.warmup_epochs]
        else:
            progress = (self.last_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)
            return [self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + math.cos(math.pi * progress))]


def make_lr_scheduler(optimizer, scheduler_spec):
    Scheduler = {
        'MultiStepLr': MultiStepLR,
        'CosineDecayWithWarmup': CosineDecayWithWarmup,
    }[scheduler_spec['name']]
    scheduler = Scheduler(optimizer, **scheduler_spec['args'])
    return scheduler