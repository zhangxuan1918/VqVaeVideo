import math
import os.path
import time
from typing import Union

import attr
import torch
import wandb
from dalle_pytorch import DiscreteVAE
from torch import nn
from torchvision import datasets
from torchvision.transforms import transforms
from wandb.sdk.lib import RunDisabled
from wandb.sdk.wandb_run import Run

from train.train_utils import get_model_size, save_checkpoint, AverageMeter, ProgressMeter, train_visualize


@attr.s(eq=False, repr=False)
class TrainVqVae:
    model: nn.Module = attr.ib()
    training_loader: torch.utils.data.DataLoader = attr.ib()
    run_wandb: Union[Run, RunDisabled, None] = attr.ib()
    num_steps: int = attr.ib()
    lr: float = attr.ib()
    lr_decay: float = attr.ib()
    folder_name: str = attr.ib()
    checkpoint_path: str = attr.ib(default=None)
    temp: float = attr.ib(default=1.0)
    temp_end: float = attr.ib(default=1.0 / 16.)
    temp_anneal_rate: float = attr.ib(default=1e-4)
    kl_weight: float = attr.ib(default=0.0)
    n_images_save: int = attr.ib(default=4)

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
            self.temp = checkpoint['temperature']
            self.kl_weight = checkpoint['kl_weight']
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
        meter_loss = AverageMeter('Loss', ':.4e')

        progress = ProgressMeter(
            len(self.training_loader),
            [batch_time, data_time, meter_loss],
            prefix="Steps: [{}]".format(self.num_steps))

        data_iter = iter(self.training_loader)
        end = time.time()

        temp = self.temp
        for i in range(self.start_steps, self.num_steps):
            # measure output loading time
            data_time.update(time.time() - end)

            try:
                images, _ = next(data_iter)
            except StopIteration:
                data_iter = iter(self.training_loader)
                images, _ = next(data_iter)
            images = images.to('cuda')
            self.optimizer.zero_grad()
            loss, images_recs = self.model(
                images,
                return_loss=True,
                return_recons=True,
                temp=self.temp
            )
            loss.backward()
            self.optimizer.step()

            meter_loss.update(loss.item(), 1)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i > 0 and i % 1000 == 0:
                print('saving ...')
                save_checkpoint(self.folder_name, {
                    'steps': i,
                    'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'scheduler': self.scheduler.state_dict(),
                    'temperature': temp,
                    'kl_weight': self.kl_weight,
                }, 'checkpoint%s.pth.tar' % i)

                self.scheduler.step()

                temp = max(temp * math.exp(-self.temp_anneal_rate * i), self.temp_end)

            if i % 20 == 0:
                progress.display(i)
                if i % 100 == 0:
                    logs = train_visualize(
                        model=self.model, images=images[:self.n_images_save], n_images=self.n_images_save,
                        image_recs=images_recs[:self.n_images_save])

                    logs = {
                        **logs,
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
            'temperature': temp,
            'kl_weight': self.kl_weight,
        }, 'checkpoint%s.pth.tar' % self.num_steps)

        self.run_wandb.finish()


def train_images():
    from image_utils import params
    data_args = params['data_args']
    train_args = params['train_args']
    model_args = params['model_args']

    if params['use_wandb']:
        wandb.login(key=os.environ['wanda_api_key'])
        run_wandb = wandb.init(
            project='dalle_train_vae',
            job_type='train_model',
            config=params
        )
    else:
        run_wandb = RunDisabled()

    model = DiscreteVAE(**model_args).cuda()
    print('num of trainable parameters: %d' % get_model_size(model))
    print(model)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    training_data = datasets.ImageFolder(
        data_args['root_dir'],
        transforms.Compose([
            transforms.RandomResizedCrop(model_args['image_size']),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
    )

    training_loader = torch.utils.data.DataLoader(
        training_data, batch_size=data_args['batch_size'], shuffle=True, num_workers=data_args['num_workers'])

    train_object = TrainVqVae(model=model, training_loader=training_loader, run_wandb=run_wandb,
                              **train_args)
    train_object.train()


if __name__ == '__main__':
    train_images()
