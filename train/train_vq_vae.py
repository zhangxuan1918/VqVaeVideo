import os.path
import time
from collections import OrderedDict

import torch
import torch.nn.functional as F
from dall_e import load_model
from torch import nn
from torchvision.transforms import transforms

from models.vq_vae.dalle.vqvae import VqVae
from train.data_util import NumpyDataset
from train.train_utils import get_model_size, save_checkpoint, AverageMeter, ProgressMeter


class TrainVqVae:

    def __init__(self, model: nn.Module, dalle_embed: nn.Module, training_loader: torch.utils.data.DataLoader,
                 num_steps: int, lr: float, lr_decay: float, folder_name: str,
                 checkpoint_path: str = None):

        self.model = model
        self.dalle_embed = dalle_embed
        self.training_loader = training_loader
        self.lr = lr
        self.lr_decay = lr_decay
        self.num_steps = num_steps
        self.folder_name = folder_name

        if not os.path.exists(folder_name):
            # shutil.rmtree(folder_name)
            os.mkdir(folder_name)
        self.checkpoint_path = checkpoint_path

        if checkpoint_path is not None:
            # load model from checkpoint
            loc = 'cuda:0'
            checkpoint = torch.load(checkpoint_path, map_location=loc)
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.scheduler.load_state_dict(checkpoint['scheduler'])
            self.start_steps = checkpoint['steps']
            print("=> loaded checkpoint '{}' (steps {})"
                  .format(checkpoint_path, self.start_steps))
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
        self.dalle_embed.eval()

        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')
        constr = AverageMeter('Constr', ':6.2f')
        perp = AverageMeter('Perplexity', ':6.2f')
        progress = ProgressMeter(
            len(self.training_loader),
            [batch_time, data_time, losses, constr, perp],
            prefix="Steps: [{}]".format(self.num_steps))

        data_iter = iter(self.training_loader)
        end = time.time()

        for i in range(self.start_steps, self.num_steps):
            # measure output loading time
            data_time.update(time.time() - end)

            try:
                z = next(data_iter)
            except StopIteration:
                z = next(data_iter)

            # data is indices
            z = z.to(torch.int64).to('cuda')
            # one hot encoding, dall-e vocab size 8192
            # shape [b, 32, 32] -> [b, 8192, 32, 32]
            z = F.one_hot(z, num_classes=8192).permute(0, 3, 1, 2).float()
            # embed indices to codes
            # shape [b, 8192, 32, 32] -> [b, 128, 32, 32]
            z = self.dalle_embed(z)

            self.optimizer.zero_grad()

            z_recon, loss_kl_div = self.model(z)
            loss_recon = F.mse_loss(z_recon, z)
            loss = loss_recon + loss_kl_div
            loss.backward()

            self.optimizer.step()

            constr.update(loss_recon.item(), 1)
            perp.update(loss_kl_div.item(), 1)
            losses.update(loss.item(), 1)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % 10 == 0:
                progress.display(i + 1)

            if (i + 1) % 1000 == 0:
                print('saving ...')
                save_checkpoint(self.folder_name, {
                    'steps': i + 1,
                    'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'scheduler': self.scheduler.state_dict(),
                }, 'checkpoint%s.pth.tar' % (i + 1))

                self.scheduler.step()

        print('saving ...')
        save_checkpoint(self.folder_name, {
            'steps': self.num_steps + 1,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
        }, 'checkpoint%s.pth.tar' % (self.num_steps + 1))


def train_images():
    from image_utils import params
    data_args = params['data_args']
    train_args = params['train_args']
    model_args = params['model_args']

    model = VqVae(**model_args).to('cuda')
    print('num of trainable parameters: %d' % get_model_size(model))
    print(model)

    # the input layer of dalle decoder is used to embed indices to codes, code dim = 128
    dalle_decoder = load_model("/opt/project/data/dall-e/decoder.pkl").to('cuda')
    dalle_embed = nn.Sequential(OrderedDict([('embed', list(dalle_decoder.children())[0][0])]))

    training_data = NumpyDataset(root_dir=data_args['root_dir'])
    training_loader = torch.utils.data.DataLoader(
        training_data, batch_size=data_args['batch_size'], shuffle=True, num_workers=data_args['num_workers'])

    train_object = TrainVqVae(model=model, dalle_embed=dalle_embed, training_loader=training_loader, **train_args)
    train_object.train()


if __name__ == '__main__':
    train_images()
