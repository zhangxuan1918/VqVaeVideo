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

from models.transformer.gpt.gpt import GPT
from train.codes.data_util import NumpyDataset
from train.train_utils import get_model_size, save_checkpoint, AverageMeter, ProgressMeter, NormalizeInverse


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

    patch_length: int = attr.ib(default=4, validator=lambda i, a, x: x & 1 == 0) # 4x4 path for each code when feeding to transformer
    max_frame_length: int = attr.ib(default=30)
    embed_dim: int = attr.ib(default=512)

    n_images_save: int = attr.ib(default=16)

    def __attrs_post_init__(self):
        # max seq length for transformer: 1 (bos) + path_length ** 2 * num_frames (tokens)
        self.max_seq_length = 1 + self.max_frame_length * (self.patch_length ** 2)
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

    def _train_helper(self, _input: torch.Tensor) -> torch.Tensor:
        # scan the video code where each frame is encoded by vqvae
        # each time, we have a small patch of size (batch, patch_size, patch_size) from the original full code
        # we flatten it to (batch, patch_size ** 2), then feed it to the transformer
        b, h, w = _input.size()
        patch_half = self.patch_length / 2
        total_loss = 0.0
        # the beginning token
        bos = torch.zeros((b, self.embed_dim))
        for r in range(0, h):
            if r <= patch_half:
                # row left upper
                i = r
            elif h - r < patch_half:
                # row left bottom
                i = self.patch_length - (h - r)
            else:
                # row middle
                i = patch_half
            for c in range(0, w):
                if c <= patch_half:
                    # column left upper
                    j = c
                elif w - c < patch_half:
                    # column right upper
                    j = self.patch_length - (w - c)
                else:
                    # column middle
                    j = patch_half

                i_start = r - i
                i_end = i_start + self.patch_length
                j_start = c - j
                j_end = j_start + self.patch_length
                patch = _input[:, i_start:i_end, j_start:j_end]
                patch = rearrange(patch, 'b h w -> b (h w)')
                logits, loss = self.model(input=patch, embeddings=bos, targets=patch)
                # update the beginning to be the last token predicted
                bos = logits[:, -1, :]
                total_loss += loss
        return total_loss / (w * h)

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
            self.optimizer.zero_grad()

            # we need to feed patch from code to model
            loss = self._train_helper(codes)
            loss.backward()

            self.optimizer.step()

            meter_loss_constr.update(recon_error.item(), 1)
            meter_loss_perp.update(perplexity.item(), 1)
            meter_loss.update(loss.item(), 1)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 20 == 0:
                progress.display(i)

            if i % 1000 == 0:
                print('saving ...')
                save_checkpoint(self.folder_name, {
                    'steps': i,
                    'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'scheduler': self.scheduler.state_dict()
                }, 'checkpoint%s.pth.tar' % i)

                self.scheduler.step()
                # codes_orig, codes_recs = train_visualize(
                #     unnormalize=self.unnormalize, images=codes[:self.n_images_save], n_images=self.n_images_save,
                #     image_recs=codes_recon[:self.n_images_save])
                #
                # save_images(file_name=os.path.join(self.path_img_orig, f'image_{i}.png'), image=codes_orig)
                # save_images(file_name=os.path.join(self.path_img_recs, f'image_{i}.png'), image=codes_recs)

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


def train_codes():
    from train.codes.code_utils import params
    data_args = params['data_args']
    train_args = params['train_args']
    model_args = params['model_args']

    if params['use_wandb']:
        wandb.login(key=os.environ['wanda_api_key'])
        run_wandb = wandb.init(
            project='dalle_train_vae',
            job_type='train_model',
            config=params,
            resume=train_args['checkpoint_path'] is not None
        )
    else:
        run_wandb = RunDisabled()

    model = VqVae(**model_args).to('cuda')
    print('num of trainable parameters: %d' % get_model_size(model))
    print(model)

    # mean and std of codes are computed using imagenet dataset
    normalize = transforms.Normalize(mean=[0.1635], std=[0.1713])
    unnormalize = NormalizeInverse(mean=[0.1635], std=[0.1713])

    training_data = NumpyDataset(
        data_args['root_dir'],
        data_args['max_seq_length'],
        data_args['padding_file'],
        transforms.Compose([Rescale(code_size=8192.), normalize])
    )

    training_loader = torch.utils.data.DataLoader(
        training_data, batch_size=data_args['batch_size'], shuffle=True, num_workers=data_args['num_workers'])

    train_object = TrainVqVae(model=model, training_loader=training_loader, run_wandb=run_wandb,
                              unnormalize=unnormalize,
                              **train_args)
    try:
        train_object.train()
    finally:
        run_wandb.finish()


if __name__ == '__main__':
    """
    we already compress images by vqvae, in this script, we aim to compress videos further in time dim
    assume the images are converted to codes
    
    for each video codes, we compress it by passing it through a new vqvae encoder
    the max frames for a video is defined in `input_channels`
    * input shape: [batch_size, max_seq_length, 32, 32]
    
    if the number of frames is smaller than max_seq_length, we pad it with black image code
    """
    train_codes()
