import os.path
import time

import torch
import torch.nn.functional as F
from nvidia.dali.plugin.pytorch import DALIGenericIterator
from torchvision.transforms import transforms

from models.vq_vae.vq_vae import VqVae
from train.data_util import ImagesDataset
from train.train_utils import get_model_size, save_checkpoint, AverageMeter, ProgressMeter


class TrainVqVae:

    def __init__(self, model, training_loader, num_steps, num_iters_epoch, lr, lr_decay, folder_name,
                 checkpoint_path=None):

        self.model = model
        self.training_loader = training_loader
        self.lr = lr
        self.lr_decay = lr_decay
        self.num_steps = num_steps
        self.folder_name = folder_name
        self.num_iters_epoch = num_iters_epoch
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

        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')
        constr = AverageMeter('Constr', ':6.2f')
        perp = AverageMeter('Perplexity', ':6.2f')
        progress = ProgressMeter(
            self.num_iters_epoch,
            [batch_time, data_time, losses, constr, perp],
            prefix="Steps: [{}]".format(self.num_steps))

        if self.model.is_video:
            data_iter = DALIGenericIterator(self.training_loader, ['data'], auto_reset=True)
        else:
            data_iter = iter(self.training_loader)
        end = time.time()

        mean = torch.as_tensor([0.485, 0.456, 0.406], device='cuda')
        mean = mean.view(-1, 1, 1, 1)
        std = torch.as_tensor([0.229, 0.224, 0.225], device='cuda')
        std = std.view(-1, 1, 1, 1)
        for i in range(self.start_steps, self.num_steps):
            # measure output loading time
            data_time.update(time.time() - end)

            try:
                data = next(data_iter)
            except StopIteration as e:
                if self.model.is_video:
                    data_iter = DALIGenericIterator(self.training_loader, ['data'], auto_reset=True)
                else:
                    data_iter = iter(self.training_loader)
                data = next(data_iter)

            if self.model.is_video:
                data = data[0]['data'].float()
                B, D, H, W, C = data.size()
                data = data.view(B, C, D, H, W)
                data = data / 127.5 - 1
                data.sub_(mean).div_(std)
            else:
                data = data.to('cuda')

            self.optimizer.zero_grad()

            vq_loss, data_recon, perplexity = self.model(data)
            recon_error = F.mse_loss(data_recon, data)
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

    model = VqVae(is_video=False, **model_args).to('cuda')
    print('num of trainable parameters: %d' % get_model_size(model))
    print(model)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    training_data = ImagesDataset(
        root_dir=data_args['root_dir'],
        transform=transforms.Compose([transforms.ToTensor(), normalize]))
    training_loader = torch.utils.data.DataLoader(
        training_data, batch_size=data_args['batch_size'], shuffle=True, num_workers=data_args['num_workers'])
    num_iters_epoch = len(training_loader)
    train_object = TrainVqVae(model=model, training_loader=training_loader, num_iters_epoch=num_iters_epoch,
                              **train_args)
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
    num_iters_epoch = training_pipe.epoch_size()['__Video_0']
    train_object = TrainVqVae(model=model, training_loader=training_pipe, num_iters_epoch=num_iters_epoch, **train_args)
    train_object.train()


if __name__ == '__main__':
    # original resolution: 1920 x 1080
    # we can scale it down to 256 *144
    is_video = True

    if is_video:
        train_videos()
    else:
        train_images()
