import os.path
import shutil
import time

import numpy as np
import torch
import torch.nn.functional as F
from nvidia.dali.plugin.pytorch import DALIGenericIterator
from torchvision import datasets
from torchvision.transforms import transforms

from models.vq_vae.vq_vae import VqVae
from train.train_utils import get_model_size, save_checkpoint, AverageMeter, ProgressMeter


class TrainVqVae:

    def __init__(self, model, training_loader, num_steps, lr, folder_name, data_std):

        self.model = model
        self.training_loader = training_loader
        self.lr = lr
        self.num_steps = num_steps
        self.folder_name = folder_name
        self.data_std = data_std

        if os.path.isdir(folder_name):
            shutil.rmtree(folder_name)
        os.mkdir(folder_name)

    @property
    def optimizer(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr, amsgrad=False)

    def train(self):
        self.model.train()

        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')
        constr = AverageMeter('Constr', ':6.2f')
        perp = AverageMeter('Perplexity', ':6.2f')
        progress = ProgressMeter(
            len(self.training_loader),
            [batch_time, data_time, losses, constr, perp],
            prefix="Steps: [{}]".format(self.num_steps))

        end = time.time()
        for i in range(self.num_steps):
            # measure output loading time
            data_time.update(time.time() - end)

            if self.model.is_video:
                data = next(self.training_loader)[0]['data'].float()
                B, D, H, W, C = data.size()
                data = data.view(B, C, D, H, W)
                data = data / 127.5 - 1
            else:
                data, _ = next(iter(self.training_loader))
                data = data.to('cuda')

            self.optimizer.zero_grad()

            vq_loss, data_recon, perplexity = self.model(data)
            recon_error = F.mse_loss(data_recon, data) / self.data_std
            loss = recon_error + vq_loss
            loss.backward()

            self.optimizer.step()

            constr.update(recon_error.item(), 1)
            perp.update(perplexity.item(), 1)
            losses.update(loss.item(), 1)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % 10 == 0:
                progress.display(i)

            if (i + 1) % 1000 == 0:
                print('saving ...')
                save_checkpoint(self.folder_name, {
                    'steps': i + 1,
                    'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                }, 'checkpoint%s.pth.tar' % (i + 1))


def train_images():
    from image_utils import params
    data_args = params['data_args']
    train_args = params['train_args']
    model_args = params['model_args']

    model = VqVae(is_video=False, **model_args).to('cuda')
    print('num of trainable parameters: %d' % get_model_size(model))
    print(model)

    training_data = datasets.CIFAR10(root="data", train=True, download=True,
                                     transform=transforms.Compose([
                                         transforms.ToTensor(),
                                         transforms.Normalize((0.5, 0.5, 0.5), (1.0, 1.0, 1.0))
                                     ]))

    training_loader = torch.utils.data.DataLoader(training_data, batch_size=data_args['batch_size'], shuffle=True)
    train_object = TrainVqVae(model=model, training_loader=training_loader, **train_args)
    train_object.train()


def train_videos():
    from video_utils import video_pipe, params

    data_args = params['data_args']
    train_args = params['train_args']
    model_args = params['model_args']

    model = VqVae(is_video=True, **model_args).to('cuda')
    print('num of trainable parameters: %d' % get_model_size(model))
    print(model)

    training_pipe = video_pipe(batch_size=data_args['batch_size'],
                               num_threads=data_args['num_threads'],
                               device_id=data_args['device_id'],
                               filenames=data_args['training_data_files'],
                               seed=data_args['seed'])
    training_pipe.build()
    training_loader = DALIGenericIterator(training_pipe, ['data'])
    train_object = TrainVqVae(model=model, training_loader=training_loader, **train_args)
    train_object.train()


if __name__ == '__main__':
    # original resolution: 1920 x 1080
    # we can scale it down to 256 *144
    is_video = False

    if is_video:
        train_videos()
    else:
        train_images()
