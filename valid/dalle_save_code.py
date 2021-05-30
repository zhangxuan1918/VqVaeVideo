# pass image in and save the code as a numpy array
import os
import torchvision.transforms as T
import torch.nn.functional as F
import torch
from dall_e import map_pixels, unmap_pixels, load_model
import numpy as np
from torchvision.utils import make_grid

from train.images.data_util import ImagesDataset
from valid.reconstruct_untils import save_images2


class ImagesDatasetWFilename(ImagesDataset):
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.image_files.iloc[idx, 0])
        image = self.loader(img_name)

        if self.transform:
            image = self.transform(image)

        return self.image_files.iloc[idx, 0], image


def check_dalle(images_dir, dalle_encoder, dalle_decoder, batch_size=4, num_workers=6):
    image_data = ImagesDatasetWFilename(root_dir=images_dir, transform=T.ToTensor())
    image_loader = torch.utils.data.DataLoader(image_data, batch_size=batch_size, shuffle=True,
                                               num_workers=num_workers)
    for _, x_org in image_loader:
        x_org = x_org.to('cuda')
        x = map_pixels(x_org)
        z_logits = dalle_encoder(x)
        z = torch.argmax(z_logits, axis=1)
        z = F.one_hot(z, num_classes=dalle_encoder.vocab_size).permute(0, 3, 1, 2).float()

        x_stats = dalle_decoder(z).float()
        x_rec = unmap_pixels(torch.sigmoid(x_stats[:, :3]))

        save_images2(make_grid(x_rec.cpu().data), 'recon')
        save_images2(make_grid(x_org.cpu().data), 'orig')

        break


def get_dalle_codes(images_dir, numpy_dir, dalle_encoder, batch_size=256, num_workers=6):
    """
    save encoded image as np array, shape [8192, 32, 32] for each image
    the size is too big, uncompressed 33m for each image, compressed 29m for each image
    :param images_dir:
    :param numpy_dir:
    :param dalle_encoder:
    :param batch_size:
    :param num_workers:
    :return:
    """
    image_data = ImagesDatasetWFilename(root_dir=images_dir, transform=T.ToTensor())
    image_loader = torch.utils.data.DataLoader(image_data, batch_size=batch_size, num_workers=num_workers)

    num_batches = len(image_loader)
    for i, (filenames, x) in enumerate(image_loader):
        if (i + 1) % 100 == 0:
            print('progress %d / %d' % (i+1, num_batches))
        x = x.to('cuda')
        x = map_pixels(x)
        z_logits = dalle_encoder(x)
        z = z_logits.cpu().detach().numpy()

        for f, a in zip(filenames, z):
            p = os.path.join(numpy_dir, f)
            np.save(p, a)
            # np.savez_compressed(p, z=a)
        break


def check_np(numpy_dir, dalle_encoder, dalle_decoder, batch_size=16):
    np_files = os.listdir(numpy_dir)
    batch_size = min(batch_size, len(np_files))
    np_array = np.empty((batch_size, dalle_encoder.vocab_size, 32, 32))
    for i in range(batch_size):
        np_array[i] = np.load(os.path.join(numpy_dir, np_files[i]))

    z_logits = torch.from_numpy(np_array).float().to('cuda')
    z = torch.argmax(z_logits, axis=1)
    z = F.one_hot(z, num_classes=dalle_encoder.vocab_size).permute(0, 3, 1, 2).float()

    x_stats = dalle_decoder(z).float()
    x_rec = unmap_pixels(torch.sigmoid(x_stats[:, :3]))

    save_images2(make_grid(x_rec.cpu().data), 'recon')


if __name__ == '__main__':
    images_dir = '/data/Doraemon/images'
    dalle_encoder = load_model("/opt/project/data/dall-e/encoder.pkl", 'cuda')
    dalle_decoder = load_model("/opt/project/data/dall-e/decoder.pkl", 'cuda')

    # check_dalle(images_dir, dalle_encoder, dalle_decoder, batch_size=16)

    numpy_dir = '/data/Doraemon/np_arrays'
    get_dalle_codes(images_dir, numpy_dir, dalle_encoder, batch_size=16, num_workers=6)

    # check_np(numpy_dir, dalle_encoder, dalle_decoder, batch_size=32)