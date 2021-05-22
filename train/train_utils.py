import math
import os
from typing import Union, Dict

import numpy as np
import torch
import wandb
from torch import Tensor, nn
from torchvision.utils import make_grid
from wandb.sdk.lib import RunDisabled
from wandb.sdk.wandb_run import Run

from models.vq_vae.dalle3.vqvae3 import VqVae3


def get_model_size(model: nn.Module) -> int:
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    return sum([np.prod(p.size()) for p in model_parameters])


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches) -> str:
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def save_checkpoint(folder_name: str, state: dict, filename: str ='checkpoint.pth.tar'):
    filename = os.path.join(folder_name, filename)
    torch.save(state, filename)


def load_checkpoint(checkpoint_path: str, device_id: int =0):
    loc = 'cuda:{}'.format(device_id)
    checkpoint = torch.load(checkpoint_path, map_location=loc)
    return checkpoint


def train_visualize(model: nn.Module, n_images: int, images: Tensor, image_recs: Tensor) -> Dict:
    with torch.no_grad():
        codes = model.get_codebook_indices(images)
        hard_recons = model.decode(codes)
    #TODO: unnormalize images
    images, recs, hard_recs, codes = map(lambda t: t.detach().cpu(), (images, image_recs, hard_recons, codes))
    images, recs, hard_recs = map(lambda t: make_grid(t.float(), nrow=int(math.sqrt(n_images)), normalize=True, range=(-1, 1)),
                                  (images, recs, hard_recs))

    return {
        'sampled images': wandb.Image(images, caption='original images'),
        'reconstructions': wandb.Image(recs, caption='reconstructions'),
        'hard reconstructions': wandb.Image(hard_recs, caption='hard reconstructions'),
        'codebook_indices': wandb.Histogram(codes)
    }