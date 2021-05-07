import torch
import torch.nn.functional as F
from torchvision.transforms import transforms, Normalize
from torchvision.utils import make_grid

from models.vq_vae.vq_vae import VqVae
from train.data_util import ImagesDataset
from train.image_utils import params
from train.train_utils import load_checkpoint
from valid.reconstruct_untils import save_images2


class NormalizeInverse(Normalize):
    """
    Undoes the normalization and returns the reconstructed images in the input domain.
    """

    def __init__(self, mean, std):
        mean = torch.as_tensor(mean)
        std = torch.as_tensor(std)
        std_inv = 1 / (std + 1e-7)
        mean_inv = -mean * std_inv
        super().__init__(mean=mean_inv, std=std_inv)

    def __call__(self, tensor):
        return super().__call__(tensor.clone())


def reconstruct_images(checkpoint_path, data_args, model_args, is_video=False):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    training_data = ImagesDataset(
        root_dir=data_args['root_dir'],
        transform=transforms.Compose([transforms.ToTensor(), normalize]))
    training_loader = torch.utils.data.DataLoader(
        training_data, batch_size=data_args['batch_size'], shuffle=True, num_workers=data_args['num_workers'])

    checkpoint = load_checkpoint(checkpoint_path, device_id=0)

    model = VqVae(is_video=is_video, **model_args).to('cuda')
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
    data_args['batch_size'] = 16

    model_id = '2021-05-01'
    checkpoint_file = 'checkpoint250001.pth.tar'
    checkpoint_path = '/opt/project/data/trained_image/%s/%s' % (model_id, checkpoint_file)
    batch_size = 16
    reconstruct_images(checkpoint_path, data_args, model_args, is_video=False)
