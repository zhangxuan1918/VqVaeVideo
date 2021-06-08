import gzip
import os

import numpy as np
import torch
from torchvision.transforms import transforms

from models.vq_vae.vq_vae0.vq_vae import VqVae
from train.codes.data_util import NumpyDataset, Rescale
from train.codes.code_utils import params
from train.train_utils import load_checkpoint, NormalizeInverse


def reconstruct_codes(checkpoint_path, data_args, model_args, np_folder):
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

    checkpoint = load_checkpoint(checkpoint_path, device_id=0)

    model = VqVae(**model_args).to('cuda')
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    data = next(iter(training_loader))
    data = data.to('cuda')
    _, data_recon, _ = model(data)
    data_recon = torch.clip(unnormalize(data_recon) * 8192, 0, 8192).int()
    data_recon = data_recon.to('cpu').numpy()
    data = torch.clip(unnormalize(data) * 8192, 0, 8192).int()
    data = data.to('cpu').numpy()

    for i in range(data_args['batch_size']):
        p_recon = os.path.join(np_folder, 'recon', f'{i}.pny.gz')
        with gzip.GzipFile(p_recon, 'w') as f:
            np.save(file=f, arr=data_recon[i])
        p_orig = os.path.join(np_folder, 'orig', f'{i}.pny.gz')
        with gzip.GzipFile(p_orig, 'w') as f:
            np.save(file=f, arr=data[i])


if __name__ == '__main__':
    model_args = params['model_args']
    data_args = params['data_args']
    data_args['batch_size'] = 32
    data_args['root_dir'] = '/data/breaking_bad/np_arrays/256x256'
    data_args['padding_file'] = '/opt/project/train/codes/black_images_code_2021-05-25.npy.gz'

    model_id = '2021-06-06'
    checkpoint_file = 'checkpoint80000.pth.tar'
    checkpoint_path = '/opt/project/data/trained_code/%s/%s' % (model_id, checkpoint_file)
    np_folder = 'data3'
    reconstruct_codes(checkpoint_path, data_args, model_args, np_folder)
