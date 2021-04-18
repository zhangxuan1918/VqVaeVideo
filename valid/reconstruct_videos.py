from nvidia.dali.plugin.pytorch import DALIGenericIterator
import numpy as np
from torchvision.utils import make_grid

from models.vq_vae.vq_vae import VqVae
from train.train_utils import load_checkpoint
from train.video_utils import video_pipe, list_videos
from valid.reconstruct_untils import save_images, save_images2
import torch.nn.functional as F
import torch


def reconstruct(checkpoint_path, batch_size, num_threads, device_id, training_data_files, model_args, seed=2021, is_video=True):
    training_pipe = video_pipe(batch_size=batch_size,
                               num_threads=num_threads,
                               device_id=device_id,
                               filenames=training_data_files,
                               seed=seed)
    training_pipe.build()
    training_loader = DALIGenericIterator(training_pipe, ['data'])
    checkpoint = load_checkpoint(checkpoint_path, device_id=0)

    model = VqVae(is_video=is_video, **model_args).to('cuda')
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    data = next(training_loader)[0]['data'].float()
    B, D, H, W, C = data.size() # batch size = 1
    data = data.view(B, C, D, H, W)
    data = data / 127.5 - 1

    _, data_recon, _ = model(data)
    recon_error = F.mse_loss(data_recon, data)
    print('reconstruct error: %6.2f' % recon_error)
    recon = ((torch.clip(data_recon, -1.0, 1.0).view(B, D, H, W, C).cpu().detach().numpy() + 1) * 127.5).astype(
        np.uint8)
    original = ((data.view(B, D, H, W, C).cpu().detach().numpy() + 1) * 127.5).astype(np.uint8)

    for i in range(B):
        # interleave original and reconstructed

        interleaved = np.empty((2 * D, H, W, C), dtype=np.uint8)
        interleaved[0::2, :, :, :] = recon[i]
        interleaved[1::2, :, :, :] = original[i]

        save_images(interleaved, i)

    # for i in range(B):
    #     save_images(recon[i], i)


if __name__ == '__main__':
    model_args = {
        'num_hiddens': 128,
        'num_residual_hiddens': 32,
        'num_residual_layers': 2,
        'embedding_dim': 64,
        'embedding_video_depth': 4,
        'num_embeddings': 512,
        'commitment_cost': 0.25,
        'decay': 0.99
    }
    model_id = '2021-04-16'
    checkpoint_file = 'checkpoint2000.pth.tar'
    checkpoint_path = '/opt/project/data/trained_video/%s/%s' % (model_id, checkpoint_file)
    device_id = 0
    training_data_files = list_videos('/data/GOT_256_144/')
    batch_size = 8
    num_threads = 2
    reconstruct(checkpoint_path, batch_size, num_threads, device_id, training_data_files, model_args, seed=1987,
                is_video=True)