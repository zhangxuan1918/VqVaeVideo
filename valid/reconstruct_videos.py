from nvidia.dali.plugin.pytorch import DALIGenericIterator
import numpy as np
from einops import rearrange
from models.vq_vae.dalle0.vq_vae import VqVae
from train.train_utils import load_checkpoint
from train.video_utils import video_pipe, list_videos2
from valid.reconstruct_untils import save_images
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

    std = torch.as_tensor([0.229, 0.224, 0.225], device='cuda')
    std_inv = 1.0 / (std + 1e-7)

    mean = torch.as_tensor([0.485, 0.456, 0.406], device='cuda')
    mean_inv = -mean * std_inv

    std = std.view(-1, 1, 1, 1)
    std_inv = std_inv.view(-1, 1, 1, 1)
    mean = mean.view(-1, 1, 1, 1)
    mean_inv = mean_inv.view(-1, 1, 1, 1)

    data = next(training_loader)[0]['data'].float()
    B, D, H, W, C = data.size() # batch size = 1
    data = rearrange(data, 'b d h w c -> b c d h w')
    data = data / 127.5 - 1
    data.sub_(mean).div_(std)

    _, data_recon, _ = model(data)
    recon_error = F.mse_loss(data_recon, data)
    print('reconstruct error: %6.2f' % recon_error)
    recon = rearrange(torch.clip(data_recon.sub_(mean_inv).div_(std_inv), -1.0, 1.0), 'b c d h w -> b d h w c')
    original = rearrange(data.sub_(mean_inv).div_(std_inv), 'b c d h w -> b d h w c')

    for i in range(B):
        # interleave original and reconstructed

        interleaved = np.empty((2 * D, H, W, C), dtype=np.uint8)
        recon_one = ((recon[i] + 1) * 127.5).cpu().detach().numpy().astype(np.uint8)
        orig_one = ((original[i] + 1) * 127.5).cpu().detach().numpy().astype(np.uint8)

        interleaved[0::2, :, :, :] = recon_one
        interleaved[1::2, :, :, :] = orig_one

        save_images(interleaved, i)

    # for i in range(B):
    #     save_images(recon[i], i)


if __name__ == '__main__':
    model_args = {
        'num_hiddens': 128,
        'num_residual_hiddens': 32,
        'num_residual_layers': 2,
        'embedding_dim': 256,
        'embedding_mul': 4,
        'num_embeddings': 8192,
        'commitment_cost': 0.25,
        'decay': 0.99
    }
    model_id = '2021-05-08'
    checkpoint_file = 'checkpoint46000.pth.tar'
    checkpoint_path = '/opt/project/data/trained_video/%s/%s' % (model_id, checkpoint_file)
    device_id = 0
    training_data_files = list_videos2('/data/Doraemon/video_clips/')
    batch_size = 16
    num_threads = 2
    reconstruct(checkpoint_path, batch_size, num_threads, device_id, training_data_files, model_args, seed=60,
                is_video=True)