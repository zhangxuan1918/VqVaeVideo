import os.path
import time
from typing import Union

import attr
import numpy as np
import torch
import torch.nn.functional as F
import wandb
from einops import rearrange
from torch import nn
from torchvision.transforms import transforms
from wandb.sdk.lib import RunDisabled
from wandb.sdk.wandb_run import Run

from models.transformer.gpt import GPT
from models.vq_vae.vq_vae0.vq_vae import VqVae
from train.gpt.data_util import NumpyDataset
from train.images.data_util import ImagesDataset
from train.train_utils import get_model_size, save_checkpoint, AverageMeter, ProgressMeter, NormalizeInverse, \
    train_visualize, save_images


@attr.s(eq=False, repr=False)
class TrainGPT:
    model: nn.Module = attr.ib()
    training_loader: torch.utils.data.DataLoader = attr.ib()
    run_wandb: Union[Run, RunDisabled, None] = attr.ib()
    num_steps: int = attr.ib()
    lr: float = attr.ib()
    lr_decay: float = attr.ib()
    folder_name: str = attr.ib()
    checkpoint_path: str = attr.ib(default=None)

    # one image has 32x32 code from vqvae which is too big, we input smaller patch into transformer
    # the patch has size [patch_size, patch_size] with stride
    patch_size: int = attr.ib(default=4)
    stride: int = attr.ib(default=2)

    def __attrs_post_init__(self):

        self.path_img_orig = os.path.join(self.folder_name, 'images_orig')
        self.path_img_recs = os.path.join(self.folder_name, 'images_recs')
        if not os.path.exists(self.folder_name):
            # shutil.rmtree(folder_name)
            os.mkdir(self.folder_name)
        if not os.path.exists(self.path_img_orig):
            os.mkdir(self.path_img_orig)
        if not os.path.exists(self.path_img_recs):
            os.mkdir(self.path_img_recs)

        if self.checkpoint_path is not None:
            # load model from checkpoint
            loc = 'cuda:0'
            checkpoint = torch.load(self.checkpoint_path, map_location=loc)
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.scheduler.load_state_dict(checkpoint['scheduler'])
            self.start_steps = checkpoint['steps']
            print("=> loaded checkpoint '{}' (steps {})"
                  .format(self.checkpoint_path, self.start_steps))
        else:
            self.start_steps = 0

    @property
    def optimizer(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr, amsgrad=False)

    @property
    def scheduler(self):
        return torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=self.lr_decay)

    def train(self):
        self.model.train()

        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        meter_loss = AverageMeter('Loss', ':6.2f')
        progress = ProgressMeter(
            len(self.training_loader),
            [batch_time, data_time, meter_loss],
            prefix="Steps: [{}]".format(self.num_steps))

        data_iter = iter(self.training_loader)
        end = time.time()

        for i in range(self.start_steps, self.num_steps):
            # measure output loading time
            data_time.update(time.time() - end)

            try:
                # codes shape: [b, f, 32, 32]
                codes = next(data_iter)
            except StopIteration:
                data_iter = iter(self.training_loader)
                codes = next(data_iter)

            # create patch, patch size 4x4
            h, w = codes.size()[2:]
            for i in range(0, h - self.patch_size, self.stride):
                loc_h = np.repeat(list(range(i, i + self.patch_size)), self.patch_size).astype(np.long)
                loc_h = torch.from_numpy(loc_h)
                for j in range(0, w - self.patch_size, self.stride):
                    loc_w = np.asarray(list(range(j, j + self.patch_size)) * self.patch_size).astype(np.long)
                    loc_w = torch.from_numpy(loc_w)

                    patch = codes[:, :, i:i + self.patch_size, j:j + self.patch_size]
                    patch = patch.to('cuda')
                    # c: first frame, we condition on the first frame to predict the following frame
                    c = rearrange(patch[:, 0:1], 'b f h w -> b (f h w)')
                    # x: following frame, also as target to predict
                    x = rearrange(patch[:, 1:], 'b f h w -> b (f h w)')
                    self.optimizer.zero_grad()

                    logits, loss = self.model(x, c, loc_h, loc_w)
                    loss.backward()

                    self.optimizer.step()

                    meter_loss.update(loss.item(), 1)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 5 == 0:
                progress.display(i)

            if i % 100 == 0:
                print('saving ...')
                save_checkpoint(self.folder_name, {
                    'steps': i,
                    'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'scheduler': self.scheduler.state_dict()
                }, 'checkpoint%s.pth.tar' % i)

                self.scheduler.step()
                if self.run_wandb:
                    logs = {
                        'iter': i,
                        'loss': meter_loss.val,
                        'lr': self.scheduler.get_last_lr()[0]
                    }
                    self.run_wandb.log(logs)

        print('saving ...')
        save_checkpoint(self.folder_name, {
            'steps': self.num_steps,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
        }, 'checkpoint%s.pth.tar' % self.num_steps)


def train_images():
    from train.gpt.gpt_utils import params
    data_args = params['data_args']
    train_args = params['train_args']
    model_args = params['model_args']

    if params['use_wandb']:
        wandb.login(key=os.environ['wanda_api_key'])
        run_wandb = wandb.init(
            project='train_gpt',
            job_type='train_model',
            config=params,
            resume=train_args['checkpoint_path'] is not None
        )
    else:
        run_wandb = RunDisabled()

    model = GPT(**model_args).to('cuda')
    print('num of trainable parameters: %d' % get_model_size(model))
    print(model)

    training_data = NumpyDataset(
        root_dir=data_args['root_dir'],
        max_seq_length=data_args['max_seq_length'],
        padding_file=data_args['padding_file']
    )
    training_loader = torch.utils.data.DataLoader(
        training_data, batch_size=data_args['batch_size'], shuffle=True, num_workers=data_args['num_workers'])

    train_object = TrainGPT(model=model, training_loader=training_loader, run_wandb=run_wandb, **train_args)
    try:
        train_object.train()
    finally:
        run_wandb.finish()


if __name__ == '__main__':
    train_images()
