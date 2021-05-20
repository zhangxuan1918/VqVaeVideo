import math
import os.path
import time
from typing import Union

import attr
import torch
import wandb
from torchvision.transforms import transforms
from wandb.sdk.lib import RunDisabled
from wandb.sdk.wandb_run import Run

from models.vq_vae.dalle3.vqvae3 import VqVae3
from train.data_util import ImagesDataset
from train.train_utils import get_model_size, save_checkpoint, AverageMeter, ProgressMeter, train_visualize


@attr.s(eq=False, repr=False)
class TrainVqVae:
    model: VqVae3 = attr.ib()
    training_loader: torch.utils.data.DataLoader = attr.ib()
    run_wandb: Union[Run, RunDisabled, None] = attr.ib()
    num_steps: int = attr.ib()
    lr: float = attr.ib()
    lr_decay: float = attr.ib()
    folder_name: str = attr.ib()
    checkpoint_path: str = attr.ib(default=None)
    temp_start: float = attr.ib(default=1.0)
    temp_end: float = attr.ib(default=1.0 / 16.)
    temp_anneal_rate: float = attr.ib(default=1e-4)
    kl_weight_start: float = attr.ib(default=0.0)
    kl_weight_end: float = attr.ib(default=6.6)
    kl_anneal_rate: float = attr.ib(default=0.0)
    n_images_save: int = attr.ib(default=4)

    def __attrs_post_init__(self):

        if not os.path.exists(self.folder_name):
            # shutil.rmtree(folder_name)
            os.mkdir(self.folder_name)

        if self.checkpoint_path is not None:
            # load model from checkpoint
            loc = 'cuda:0'
            checkpoint = torch.load(self.checkpoint_path, map_location=loc)
            self.model.encoder.load_state_dict(checkpoint['encoder_state_dict'])
            self.model.decoder.load_state_dict(checkpoint['decoder_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.scheduler.load_state_dict(checkpoint['scheduler'])
            self.temp_start = checkpoint['temperature']
            self.kl_weight_start = checkpoint['kl_weight']
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
        meter_recs_loss = AverageMeter('Constr', ':6.2f')
        meter_kl_loss = AverageMeter('Perplexity', ':6.2f')
        progress = ProgressMeter(
            len(self.training_loader),
            [batch_time, data_time, meter_loss, meter_recs_loss, meter_kl_loss],
            prefix="Steps: [{}]".format(self.num_steps))

        temperature = self.temp_start
        kl_weight = self.kl_weight_start

        data_iter = iter(self.training_loader)
        end = time.time()

        for i in range(self.start_steps, self.num_steps):
            # measure output loading time
            data_time.update(time.time() - end)

            try:
                images = next(data_iter)
            except StopIteration:
                images = next(data_iter)
            images = images.to('cuda')
            is_recs = i > 0 and i % 1000 == 0
            self.optimizer.zero_grad()
            images_recs, loss_rec, loss_kl = self.model(images, temperature, kl_weight, is_recs)
            loss = loss_rec + loss_kl
            loss.backward()
            self.optimizer.step()

            meter_recs_loss.update(loss_rec.item(), 1)
            meter_kl_loss.update(loss_kl.item(), 1)
            meter_loss.update(loss.item(), 1)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i > 0 and i % 1000 == 0:
                print('saving ...')
                save_checkpoint(self.folder_name, {
                    'steps': i,
                    'encoder_state_dict': self.model.encoder.state_dict(),
                    'decoder_state_dict': self.model.decoder.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'scheduler': self.scheduler.state_dict(),
                    'temperature': temperature,
                    'kl_weight': kl_weight,
                }, 'checkpoint%s.pth.tar' % i)

                self.scheduler.step()

                temperature = max(temperature * math.exp(-self.temp_anneal_rate * i), self.temp_end)
                kl_weight = min(kl_weight + self.kl_anneal_rate * i, self.kl_weight_end)

                if self.run_wandb:
                    logs = train_visualize(
                        model=self.model, images=images[:self.n_images_save], n_images=self.n_images_save,
                        image_recs=images_recs[:self.n_images_save])

                    logs = {
                        **logs,
                        # 'iter': i,
                        # 'loss': meter_loss.val,
                        'loss_recs': meter_recs_loss.val,
                        'loss_recs_avg': meter_recs_loss.avg,
                        # 'loss_kl': meter_kl_loss.val,
                        # 'loss_kl_avg': meter_kl_loss.avg,
                        'lr': self.scheduler.get_last_lr(),
                        'temp': temperature,
                        # 'kl_weight': kl_weight
                    }
                    self.run_wandb.log(logs)

            if i % 100 == 0:
                progress.display(i)

        print('saving ...')
        save_checkpoint(self.folder_name, {
            'steps': self.num_steps,
            'encoder_state_dict': self.model.encoder.state_dict(),
            'decoder_state_dict': self.model.decoder.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'temperature': temperature,
            'kl_weight': kl_weight,
        }, 'checkpoint%s.pth.tar' % self.num_steps)

        self.run_wandb.finish()


def train_images():
    from image_utils import params
    data_args = params['data_args']
    train_args = params['train_args']
    model_args = params['model_args']

    run_wandb = None
    if params['use_wandb']:
        wandb.login(key=os.environ['wanda_api_key'])
        run_wandb = wandb.init(
            project='dalle_train_vae',
            job_type='train_model',
            config=params
        )

    model = VqVae3(**model_args)
    n_params = sum(map(get_model_size, [model.encoder, model.decoder]))
    print('num of trainable parameters: %d' % n_params)
    print(model)

    training_data = ImagesDataset(root_dir=data_args['root_dir'], transform=transforms.ToTensor())
    training_loader = torch.utils.data.DataLoader(
        training_data, batch_size=data_args['batch_size'], shuffle=True, num_workers=data_args['num_workers'])

    train_object = TrainVqVae(model=model, training_loader=training_loader, run_wandb=run_wandb,
                              **train_args)
    train_object.train()


if __name__ == '__main__':
    train_images()
