import os.path
import time
from typing import Union, Optional

import attr
import torch
import wandb
from einops import rearrange
from torch import nn
from wandb.sdk.lib import RunDisabled
from wandb.sdk.wandb_run import Run

from models.cond_transformer.vgpt import VGPT
from train.gpt.data_util import NumpyDataset
from train.train_utils import get_model_size, save_checkpoint, AverageMeter, ProgressMeter


@attr.s(eq=False, repr=False)
class TrainGPT:
    model: nn.Module = attr.ib()
    training_loader: Optional[torch.utils.data.DataLoader] = attr.ib()
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
        if not os.path.exists(self.folder_name):
            # shutil.rmtree(folder_name)
            os.mkdir(self.folder_name)

        if self.checkpoint_path is not None:
            # load model from checkpoint
            loc = 'cuda:0'
            checkpoint = torch.load(self.checkpoint_path, map_location=loc)
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.scheduler.load_state_dict(checkpoint['scheduler'])
            self.start_steps = checkpoint['steps']
            print("=> loaded checkpoint '{}' (steps {})".format(self.checkpoint_path, self.start_steps))
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
            try:
                # codes shape: [b, f, 32, 32]
                codes = next(data_iter)
            except StopIteration:
                data_iter = iter(self.training_loader)
                codes = next(data_iter)

            # create patch, patch size 4x4
            h, w = codes.size()[2:]
            for dx in range(0, h - self.patch_size, self.stride):
                for dy in range(0, w - self.patch_size, self.stride):
                    x = codes[:, :, dx:dx + self.patch_size, dy:dy + self.patch_size]
                    x = rearrange(x, 'b f h w -> b (f h w)').to('cuda')
                    self.optimizer.zero_grad()

                    logits, loss = self.model(x)
                    loss.backward()

                    self.optimizer.step()

                    meter_loss.update(loss.item(), 1)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

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

    def sample(self, c, topk=None, sample=False):
        self.model.eval()
        x = self.model.sample(c=c, patch_size=self.patch_size, stride=self.stride, topk=topk, sample=sample)
        return x


def train_gpt():
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

    model = VGPT(**model_args).to('cuda')
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
    train_gpt()
