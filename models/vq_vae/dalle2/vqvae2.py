from collections import OrderedDict

import torch
from dall_e import load_model
from torch import nn
import attr
import torch.nn.functional as F
from models.vq_vae.dalle import map_pixels


@attr.s(eq=False, repr=False)
class VqVae2:

    vqvae: nn.Module = attr.ib()
    dalle_encoder_path: str = attr.ib(default='/opt/project/data/dall-e/encoder.pkl')
    dalle_decoder_path: str = attr.ib(default='/opt/project/data/dall-e/decoder.pkl')

    def __attrs_post_init__(self):
        super().__init__()

        self.dalle_encoder = load_model(self.dalle_encoder_path)
        decoder = load_model(self.dalle_decoder_path)
        # split decoder
        layers = list(decoder.children())[0]
        self.dalle_embed = nn.Sequential(OrderedDict([('embed', layers[0])]))
        self.dalle_decoder = nn.Sequential(*layers[1:])

    @torch.no_grad()
    def encode(self, x) -> torch.Tensor:
        """
        x -> dalle_encoder -> dalle_embed -> vqvae.encoder

        [B, 3, 255, 255] -> [B, 128, 32, 32] -> [B, 4, 4]
        :return:
        """
        # x->dalle_encoder
        x = map_pixels(x)
        # z1 shape: [B, 8192, 32, 32]
        z = self.dalle_encoder(x)

        # encode
        # z1 shape: [B, 32, 32]
        z = torch.argmax(z, dim=1)
        # z1 shape: [B, 8192, 32, 32]
        z = F.one_hot(z, num_classes=self.dalle_encoder.vocab_size).permute(0, 3, 1, 2).float()

        # embed code into codebook
        # z1 shape: [B, 128, 32, 32]
        z = self.dalle_embed(z)

        # z1 -> vqvae.encoder
        # z2 shape: [B, 8192, 4, 4]
        z = self.vqvae.encoder(z)

        # encode
        # z2 shape: [B, 4, 4]
        z = torch.argmax(z, dim=1)
        return z

    @torch.no_grad()
    def decode(self, z) -> torch.Tensor:
        """
        z -> vqvae.decoder -> dalle_decoder

        [B, 4, 4] -> [B, 128, 32, 32] -> [B, 3, 256, 256]
        :return:
        """
        # z -> vqvae.decoder
        # z shape: [B, 8192, 4, 4]
        z = F.one_hot(z, num_classes=self.vqvae.encoder.vocab_size).permute(0, 3, 1, 2).float()

        # x1 shape: [B, 128, 32, 32]
        x = self.vqvae.decoder(z)

        # todo, get the exact code from dalle, the vqvae will get an approximation
        #   for each code, we can find the closest code in dalle and replace it, doing the vectorization here

        # decode
        # x2 shape: [B, 6, 255, 255]
        x = self.dalle_decoder(x)
        return x