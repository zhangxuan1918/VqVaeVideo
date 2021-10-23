import os.path
import time
from typing import Union

import attr
import torch
import torch.nn.functional as F
import wandb
from einops import rearrange
from torch import nn
from torchvision.transforms import transforms
from wandb.sdk.lib import RunDisabled
from wandb.sdk.wandb_run import Run

from models.vq_vae.vq_vae0 import VqVae
from models.vq_vae.vq_vae2_video import VqVae2
from train.train_utils import get_model_size, save_checkpoint, AverageMeter, ProgressMeter, NormalizeInverse, \
    train_visualize, save_images
from train.videos2.data_util import VideoNumpyDataset


@attr.s(eq=False, repr=False)
class TrainVqVae:
    model: nn.Module = attr.ib()
    model_img: nn.Module = attr.ib()
    normalize: nn.Module = attr.ib()
    unnormalize: nn.Module = attr.ib()
    training_loader: torch.utils.data.DataLoader = attr.ib()
    run_wandb: Union[Run, RunDisabled, None] = attr.ib()
    num_steps: int = attr.ib()
    lr: float = attr.ib()
    lr_decay: float = attr.ib()
    folder_name: str = attr.ib()
    checkpoint_path: str = attr.ib(default=None)
    n_images_save: int = attr.ib(default=16)

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
        meter_loss = AverageMeter('Loss', ':.4e')
        meter_loss_constr = AverageMeter('Constr', ':6.2f')
        meter_loss_perp = AverageMeter('Perplexity', ':6.2f')
        progress = ProgressMeter(
            len(self.training_loader),
            [batch_time, data_time, meter_loss, meter_loss_constr, meter_loss_perp],
            prefix="Steps: [{}]".format(self.num_steps))

        data_iter = iter(self.training_loader)
        end = time.time()

        for i in range(self.start_steps, self.num_steps):
            # measure output loading time
            data_time.update(time.time() - end)

            try:
                codes = next(data_iter)
            except StopIteration:
                data_iter = iter(self.training_loader)
                codes = next(data_iter)
            codes = codes.to('cuda')
            b, d, _, _ = codes.shape
            codes = rearrange(codes, 'b d h w -> (b d) h w')
            quantized = self.model_img.code2quantized(codes)
            quantized = rearrange(quantized, '(b d) c h w -> b d c h w', b=b, d=d)
            self.optimizer.zero_grad()

            vq_loss, quantized_recon, perplexity = self.model(quantized)
            recon_error = F.mse_loss(quantized_recon, quantized)
            loss = recon_error + vq_loss
            loss.backward()

            self.optimizer.step()

            meter_loss_constr.update(recon_error.item(), 1)
            meter_loss_perp.update(perplexity.item(), 1)
            meter_loss.update(loss.item(), 1)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 200 == 0:
                progress.display(i)

            if i % 2000 == 0:
                print('saving ...')
                save_checkpoint(self.folder_name, {
                    'steps': i,
                    'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'scheduler': self.scheduler.state_dict()
                }, 'checkpoint%s.pth.tar' % i)

                self.scheduler.step()
                quantized_sample, quantized_recon_sample = map(lambda t: t[0, :self.n_images_save], [quantized, quantized_recon])
                images, images_recon = map(lambda t: self.model_img.quantized2image(t), [quantized_sample, quantized_recon_sample])
                images_orig, images_recs = train_visualize(
                    unnormalize=self.unnormalize, images=images, n_images=self.n_images_save, image_recs=images_recon)

                save_images(file_name=os.path.join(self.path_img_orig, f'image_{i}.png'), image=images_orig)
                save_images(file_name=os.path.join(self.path_img_recs, f'image_{i}.png'), image=images_recs)

                if self.run_wandb:
                    logs = {
                        'iter': i,
                        'loss_recs': meter_loss_constr.val,
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


def restore_eval_model(model_args, classObject, checkpoint_path):
    model = classObject(**model_args)
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model


def train_videos():
    from video_utils import params
    data_args = params['data_args']
    train_args = params['train_args']
    model_args = params['model_args']

    if params['use_wandb']:
        wandb.login(key=os.environ['wanda_api_key'])
        run_wandb = wandb.init(
            project='video-vae',
            job_type='train_model',
            config=params,
            resume=train_args['checkpoint_path'] is not None
        )
    else:
        run_wandb = RunDisabled()

    from train.images.image_utils import params as params2
    model_img = restore_eval_model(params2['model_args'], VqVae, params['image_checkpoint_path']).to('cuda')
    model = VqVae2(**model_args).to('cuda')
    print(model)
    print('num of trainable parameters: %d' % get_model_size(model))

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    unnormalize = NormalizeInverse(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    training_data = VideoNumpyDataset(
        root_dir=data_args['root_dir'],
        sequence_length=data_args['sequence_length'],
        padding_file=data_args['padding_file']
    )

    training_loader = torch.utils.data.DataLoader(
        training_data, batch_size=data_args['batch_size'], shuffle=True, num_workers=data_args['num_workers'])

    train_object = TrainVqVae(model=model, model_img=model_img, training_loader=training_loader, run_wandb=run_wandb,
                              normalize=normalize, unnormalize=unnormalize,
                              **train_args)
    try:
        train_object.train()
    finally:
        run_wandb.finish()


if __name__ == '__main__':
    train_videos()
