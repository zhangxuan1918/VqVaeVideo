import torch
import torch.nn.functional as F
from torchvision.transforms import transforms, Normalize
from torchvision.utils import make_grid
from models.vq_vae.vq_vae0.vq_vae import VqVae
from train.images.data_util import ImagesDataset
from train.images.image_utils import params
from train.train_utils import load_checkpoint, NormalizeInverse
from valid.reconstruct_untils import save_images2


def reconstruct_images(checkpoint_path, data_args, model_args):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    training_data = ImagesDataset(
        root_dir=data_args['root_dir'],
        transform=transforms.Compose([transforms.ToTensor(), normalize]))
    training_loader = torch.utils.data.DataLoader(
        training_data, batch_size=data_args['batch_size'], shuffle=True, num_workers=data_args['num_workers'])

    checkpoint = load_checkpoint(checkpoint_path, device_id=0)

    model = VqVae(**model_args).to('cuda')
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    data = next(iter(training_loader))
    data = data.to('cuda')
    _, data_recon, _ = model(data)
    recon_error = F.mse_loss(data_recon, data)
    print('reconstruct error: %6.2f' % recon_error)

    unnormalize = NormalizeInverse(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])

    data_recon_unnormalized = unnormalize(data_recon)
    data_orig_unnormalized = unnormalize(data)
    save_images2(make_grid(data_recon_unnormalized.cpu().data), 'recon')
    save_images2(make_grid(data_orig_unnormalized.cpu().data), 'orig')


if __name__ == '__main__':
    model_args = params['model_args']
    data_args = params['data_args']
    data_args['batch_size'] = 32
    # data_args['root_dir'] = '/data/breaking_bad/images/256x256'
    data_args['root_dir'] = '/data2'

    model_id = '2021-05-25'
    checkpoint_file = 'checkpoint80000.pth.tar'
    checkpoint_path = '/opt/project/data/trained_image/%s/%s' % (model_id, checkpoint_file)
    batch_size = 16
    reconstruct_images(checkpoint_path, data_args, model_args)
