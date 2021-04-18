import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets
from torchvision.transforms import transforms
from torchvision.utils import make_grid

from models.vq_vae.vq_vae import VqVae
from train.train_utils import load_checkpoint
from valid.reconstruct_untils import save_images2


def reconstruct_images(checkpoint_path, batch_size, model_args, is_video=False):
    training_data = datasets.CIFAR10(root="data", train=True, download=True,
                                     transform=transforms.Compose([
                                         transforms.ToTensor(),
                                         transforms.Normalize((0.5, 0.5, 0.5), (1.0, 1.0, 1.0))
                                     ]))

    training_loader = torch.utils.data.DataLoader(training_data, batch_size=batch_size, shuffle=True)
    checkpoint = load_checkpoint(checkpoint_path, device_id=0)

    model = VqVae(is_video=is_video, **model_args).to('cuda')
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    data, _ = next(iter(training_loader))
    data = data.to('cuda')
    _, data_recon, _ = model(data)
    recon_error = F.mse_loss(data_recon, data)
    print('reconstruct error: %6.2f' % recon_error)

    save_images2(make_grid(data_recon.cpu().data) + 0.5, 'recon')
    save_images2(make_grid(data.cpu().data) + 0.5, 'orig')


if __name__ == '__main__':
    model_args = {
        'num_hiddens': 128,
        'num_residual_hiddens': 32,
        'num_residual_layers': 2,
        'embedding_dim': 64,
        'num_embeddings': 512,
        'commitment_cost': 0.25,
        'decay': 0.99
    }
    model_id = '2021-04-18'
    checkpoint_file = 'checkpoint1000.pth.tar'
    checkpoint_path = '/opt/project/data/trained_image/%s/%s' % (model_id, checkpoint_file)
    batch_size = 16
    reconstruct_images(checkpoint_path, batch_size, model_args, is_video=False)
