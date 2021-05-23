import math
import os
from typing import Dict

import numpy as np
import torch
import wandb
from torchvision.transforms import Normalize
from torchvision.utils import make_grid


def get_model_size(model):
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

    def _get_batch_fmtstr(self, num_batches):
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


def save_checkpoint(folder_name, state, filename='checkpoint.pth.tar'):
    filename = os.path.join(folder_name, filename)
    torch.save(state, filename)


def load_checkpoint(checkpoint_path, device_id=0):
    loc = 'cuda:{}'.format(device_id)
    checkpoint = torch.load(checkpoint_path, map_location=loc)
    return checkpoint


def train_visualize(unnormalize: torch.nn.Module, n_images: int, images: torch.Tensor,
                    image_recs: torch.Tensor) -> Dict:
    images, recs = map(lambda t: unnormalize(t).detach().cpu(), (images, image_recs))
    images, recs = map(lambda t: make_grid(t.float(), nrow=int(math.sqrt(n_images)), normalize=True, range=(-1, 1)),
                       (images, recs))

    return {
        'sampled images': wandb.Image(images, caption='original'),
        'reconstructions': wandb.Image(recs, caption='recons')
    }


class NormalizeInverse(Normalize):
    """
    Undoes the normalization and returns the reconstructed images in the input domain.
    """

    def __init__(self, mean, std):
        mean = torch.as_tensor(mean)
        std = torch.as_tensor(std)
        std_inv = 1 / (std + 1e-7)
        mean_inv = -mean * std_inv
        super().__init__(mean=mean_inv, std=std_inv)

    def __call__(self, tensor):
        return super().__call__(tensor.clone())
