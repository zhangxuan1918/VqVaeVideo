import os

import PIL
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
from dall_e import map_pixels, unmap_pixels, load_model

from train.train_utils import get_model_size


def load_image(filename):
    return PIL.Image.open(filename)


def preprocess(img, target_image_size=256):
    s = min(img.size)

    if s < target_image_size:
        raise ValueError(f'min dim for image {s} < {target_image_size}')

    r = target_image_size / s
    s = (round(r * img.size[1]), round(r * img.size[0]))
    img = TF.resize(img, s, interpolation=PIL.Image.LANCZOS)
    img = TF.center_crop(img, output_size=2 * [target_image_size])
    img = torch.unsqueeze(T.ToTensor()(img), 0)
    return map_pixels(img)


def reconstruct_with_dalle(x, encoder, decoder, do_preprocess=False):
    # takes in tensor (or optionally, a PIL image) and returns a PIL image
    if do_preprocess:
        x = preprocess(x).to('cuda')
    z_logits = encoder(x)
    z = torch.argmax(z_logits, axis=1)

    print(f"DALL-E: latent shape: {z.shape}")
    z = F.one_hot(z, num_classes=encoder.VOCAB_SIZE).permute(0, 3, 1, 2).float()

    x_stats = decoder(z).float()
    x_rec = unmap_pixels(torch.sigmoid(x_stats[:, :3]))
    x_rec = T.ToPILImage(mode='RGB')(x_rec[0])

    return x_rec


if __name__ == '__main__':
    encoder_dalle = load_model("/opt/project/data/dall-e/encoder.pkl", 'cuda')
    decoder_dalle = load_model("/opt/project/data/dall-e/decoder.pkl", 'cuda')

    folder = '/opt/project/valid/data2'
    filename = 'image-35081.png'
    x = load_image(os.path.join(folder, filename))
    recon_x = reconstruct_with_dalle(x, encoder_dalle, decoder_dalle, do_preprocess=True)

    recon_x.save(os.path.join(folder, filename.split('.')[0] + '_recon.jpg'))

    # print('encoder:')
    # print(encoder_dalle)
    # print('encoder size', get_model_size(encoder_dalle))
    #
    # print('decoder:')
    # print(decoder_dalle)
    # print('decoder size', get_model_size(decoder_dalle))