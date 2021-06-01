import random
from einops import rearrange
from nvidia.dali.plugin.pytorch import DALIGenericIterator
from torchvision.transforms import transforms

from models.vq_vae.vq_vae0.vq_vae import VqVae
from train.train_utils import load_checkpoint, NormalizeInverse, train_visualize, save_images
from train.videos.video_utils import video_pipe, list_videos2, params


def reconstruct(checkpoint_path, data_args, model_args):
    training_loader = video_pipe(batch_size=data_args['batch_size'],
                                 num_threads=data_args['num_threads'],
                                 device_id=data_args['device_id'],
                                 filenames=data_args['training_data_files'],
                                 seed=data_args['seed'])
    training_loader.build()
    training_loader = DALIGenericIterator(training_loader, ['data'])
    checkpoint = load_checkpoint(checkpoint_path, device_id=0)

    model = VqVae(**model_args).to('cuda')
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    unnormalize = NormalizeInverse(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    images = next(training_loader)[0]['data']
    b, d, _, _, c = images.size()
    images = rearrange(images, 'b d h w c -> (b d) c h w')
    images = normalize(images.float() / 255.)
    images = rearrange(images, '(b d) c h w -> b (d c) h w', b=b, d=d, c=c)

    vq_loss, images_recon, _ = model(images)
    print('reconstruct error: %6.2f' % vq_loss)
    images, images_recon = map(lambda t: rearrange(t, 'b (d c) h w -> (b d) c h w', b=b, d=d, c=c),
                               [images, images_recon])
    images_orig, images_recs = train_visualize(
        unnormalize=unnormalize, images=images, n_images=b * d, image_recs=images_recon)

    save_images(file_name='images_orig.png', image=images_orig)
    save_images(file_name='images_recon.png', image=images_recs)


if __name__ == '__main__':
    data_args = params['data_args']
    model_args = params['model_args']

    data_args['batch_size'] = 4
    data_args['seed'] = random.randint(0, 100)
    data_args['training_data_files'] = list_videos2('/data/Doraemon/video_clips/256x256/')

    model_id = '2021-05-27'
    checkpoint_file = 'checkpoint24000.pth.tar'
    checkpoint_path = '/opt/project/data/trained_video/%s/%s' % (model_id, checkpoint_file)
    device_id = 0
    batch_size = 16
    num_threads = 2
    reconstruct(checkpoint_path, data_args, model_args)
