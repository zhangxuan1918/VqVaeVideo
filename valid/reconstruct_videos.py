import cv2
import numpy as np
import torch
from einops import rearrange
from torchvision.transforms import transforms

from models.vq_vae.vq_vae0.vq_vae import VqVae
from train.train_utils import load_checkpoint, NormalizeInverse
from train.videos.video_utils import params


def save_video(model, images, normalize, unnormalize):
    np_images = np.asarray(images)
    images = torch.from_numpy(np_images).to('cuda')
    b, d, _, _, c = images.size()
    images = rearrange(images, 'b d h w c -> (b d) c h w')
    images = normalize(images.float() / 255.)
    images = rearrange(images, '(b d) c h w -> b (d c) h w', b=b, d=d, c=c)

    _, images_recon, _ = model(images)
    images_recon = rearrange(images_recon, 'b (d c) h w -> b d c h w', b=b, d=d, c=c)
    images_recon = unnormalize(images_recon)
    images_recon = torch.clip(images_recon, 0.0, 1.0)
    images_recon = rearrange(images_recon, 'b d c h w -> b d h w c').detach().cpu().numpy()
    images_recon = (images_recon * 255).astype('uint8')

    return images_recon


def reconstruct(checkpoint_path, batch_size, model_args, video_file, resolution=256):
    checkpoint = load_checkpoint(checkpoint_path, device_id=0)

    model = VqVae(**model_args).to('cuda')
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    unnormalize = NormalizeInverse(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    video_in = cv2.VideoCapture(video_file)
    fps = video_in.get(cv2.CAP_PROP_FPS)

    video_out = cv2.VideoWriter('reconst.mp4', cv2.VideoWriter_fourcc(*'FMP4'), fps, (resolution, resolution))

    raw_images = []
    raw_seqs = []
    i = 1
    while video_in.isOpened():
        ret, frame = video_in.read()
        if ret:
            raw_seqs.append(frame)
        else:
            break

        if len(raw_seqs) == 16:
            raw_images.append(np.stack(raw_seqs, 0))
            raw_seqs = []

        if len(raw_images) == batch_size:
            print('batch %d' % i)
            i += 1
            raw_images = np.stack(raw_images, 0)
            images_recon = save_video(model, raw_images, normalize, unnormalize)
            for seqs in images_recon:
                for frame in seqs:
                    video_out.write(frame)
            raw_images = []

    if len(raw_images) > 0:
        print('batch %d' % i)
        i += 1
        raw_images = np.stack(raw_images, 0)
        images_recon = save_video(model, raw_images, normalize, unnormalize)
        for seqs in images_recon:
            for frame in seqs:
                video_out.write(frame)

    video_out.release()
    video_in.release()
    cv2.destroyAllWindows()


def reconstruct_test(batch_size, video_file, resolution=256):

    video_in = cv2.VideoCapture(video_file)
    fps = video_in.get(cv2.CAP_PROP_FPS)

    video_out = cv2.VideoWriter('reconst.mp4', cv2.VideoWriter_fourcc(*'FMP4'), fps, (resolution, resolution))

    raw_images = []
    raw_seqs = []
    i = 1
    while video_in.isOpened():
        ret, frame = video_in.read()
        if ret:
            raw_seqs.append(frame)
        else:
            break

        if len(raw_seqs) == 16:
            raw_images.append(raw_seqs)
            raw_seqs = []

        if len(raw_images) == batch_size:
            print('batch %d' % i)
            i += 1
            for seqs in raw_images:
                for frame in seqs:
                    video_out.write(frame)
            raw_images = []

    # if len(raw_seqs) > 0:
    #     raw_images.append(raw_seqs)
    if len(raw_images) > 0:
        print('batch %d' % i)
        i += 1
        for seqs in raw_images:
            for frame in seqs:
                video_out.write(frame)
    video_out.release()
    video_in.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    model_args = params['model_args']
    model_id = '2021-05-27'
    checkpoint_file = 'checkpoint24000.pth.tar'
    checkpoint_path = '/opt/project/data/trained_video/%s/%s' % (model_id, checkpoint_file)

    # video_file = '/data/Doraemon/raw/256x256/2014.mp4'
    video_file = '/data/Doraemon/video_clips/256x256/2014-18.mp4'
    batch_size = 48
    reconstruct(checkpoint_path, batch_size, model_args, video_file, resolution=256)
    # reconstruct_test(batch_size, video_file, resolution=256)
