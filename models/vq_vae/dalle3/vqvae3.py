from collections import OrderedDict
from typing import Iterator
from functools import partial
from itertools import chain
import attr
import torch
import torch.nn.functional as F
from dall_e import load_model
from torch import nn, Tensor
from torch.nn import Parameter

from models.vq_vae.dalle.decoder import DecoderBlock
from models.vq_vae.dalle.encoder import EncoderBlock
from models.vq_vae.dalle.utils import Conv2d


@attr.s(eq=False, repr=False)
class VqVae3:
    dalle_encoder_path: str = attr.ib(default='/opt/project/data/dall-e/encoder.pkl')
    dalle_decoder_path: str = attr.ib(default='/opt/project/data/dall-e/decoder.pkl')

    group_count: int = 4
    n_hid: int = attr.ib(default=256, validator=lambda i, a, x: x >= 64)
    n_init: int = attr.ib(default=128, validator=lambda i, a, x: x >= 8)
    n_blk_per_group: int = attr.ib(default=2, validator=lambda i, a, x: x >= 1)
    vocab_size: int = attr.ib(default=8192, validator=lambda i, a, x: x >= 512)

    device: torch.device = attr.ib(default=torch.device('cpu'))
    requires_grad: bool = attr.ib(default=False)
    use_mixed_precision: bool = attr.ib(default=True)

    def __attrs_post_init__(self):
        super().__init__()

        encoder = load_model(path=self.dalle_encoder_path, device=self.device)
        encoder.device = self.device
        # remove output layer
        encoder_layers1 = list(list(encoder.children())[0].named_children())[:-1]
        # add maxpooling after conv for the last layer extracted
        encoder_layers1[-1][1].add_module('pool', nn.MaxPool2d(kernel_size=2))
        self.dalle_encoder = nn.Sequential(OrderedDict(encoder_layers1)).eval()

        decoder = load_model(path=self.dalle_decoder_path, device=self.device)
        decoder.device = self.device
        # remove input and group_1 layer
        decoder_layer2 = list(list(decoder.children())[0].named_children())[1:]
        self.dalle_decoder = nn.Sequential(OrderedDict(decoder_layer2)).eval()

        # create additional layers for encoder and decoder
        blk_range = range(self.n_blk_per_group)
        n_layers = self.group_count * self.n_blk_per_group
        make_conv = partial(Conv2d, device=self.device, requires_grad=self.requires_grad,
                            use_float16=self.use_mixed_precision)
        make_encode_blk = partial(EncoderBlock, n_layers=n_layers, device=self.device, requires_grad=self.requires_grad,
                                  use_float16=self.use_mixed_precision)

        # encoder continues to reduce the spatial dim
        self.encoder = nn.Sequential(OrderedDict([
                ('group_5', nn.Sequential(OrderedDict([
                    *[(f'block_{i + 1}', make_encode_blk(8 * self.n_hid if i == 0 else 1 * self.n_hid, 1 * self.n_hid))
                      for i in blk_range],
                    ('pool', nn.MaxPool2d(kernel_size=2))
                ]))),
                ('group_6', nn.Sequential(OrderedDict([
                    *[(f'block_{i + 1}', make_encode_blk(1 * self.n_hid if i == 0 else 2 * self.n_hid, 2 * self.n_hid))
                      for i in blk_range],
                    ('pool', nn.MaxPool2d(kernel_size=2))
                ]))),
                ('output', nn.Sequential(OrderedDict([
                    ('relu', nn.ReLU()),
                    ('conv', make_conv(2 * self.n_hid, self.vocab_size, 1)),
                ])))
            ]))

        # decoder continues to reduce the spatial dim
        make_decode_blk = partial(DecoderBlock, n_layers=n_layers, device=self.device, requires_grad=self.requires_grad,
                                  use_float16=self.use_mixed_precision)
        self.decoder = nn.Sequential(OrderedDict([
            ('input', make_conv(self.vocab_size, self.n_init, 1, use_float16=self.use_mixed_precision)),
            ('group_5', nn.Sequential(OrderedDict([
                *[(f'block_{i + 1}', make_decode_blk(1 * self.n_init if i == 0 else 4 * self.n_init, 4 * self.n_init))
                  for i in blk_range],
                ('upsample', nn.Upsample(scale_factor=2, mode='nearest')),
            ]))),
            ('group_6', nn.Sequential(OrderedDict([
                *[(f'block_{i + 1}', make_decode_blk(4 * self.n_init if i == 0 else 2 * self.n_init, 2 * self.n_init))
                  for i in blk_range],
                ('upsample', nn.Upsample(scale_factor=2, mode='nearest')),
            ]))),
            ('group_7', nn.Sequential(OrderedDict([
                *[(f'block_{i + 1}', make_decode_blk(2 * self.n_init if i == 0 else self.n_init, self.n_init))
                  for i in blk_range],
                ('upsample', nn.Upsample(scale_factor=2, mode='nearest')),
            ])))
        ]))

    @torch.no_grad()
    def encode(self, x) -> torch.Tensor:
        """
        [B, 3, 255, 255] -> [B, 4, 4]
        :return:
        """
        # z shape: [B, 2048, 32, 32]
        z = self.dalle_encoder(x)
        # z shape: [B, 8192, 4, 4]
        z = self.encoder(z)
        z = torch.argmax(z, dim=1)
        return z

    @torch.no_grad()
    def decode(self, z) -> torch.Tensor:
        """
        [B, 4, 4] -> [B, 6, 256, 256]
        :return:
        """
        # z shape: [B, 4, 4] -> [B, 8192, 4, 4]
        z = F.one_hot(z, num_classes=self.vocab_size).permute(0, 3, 1, 2).float()
        # x shape: [B, 128, 32, 32]
        x = self.decoder(z)
        # x shape: [B, 6, 256, 256]
        x = self.dalle_decoder(x)
        return x

    def __str__(self) -> str:
        model_str = '\n========== Encoder =========\n'
        model_str += str(self.dalle_encoder)
        model_str += '\n'
        model_str += str(self.encoder)

        model_str += '\n========= Decoder =========\n'
        model_str += str(self.decoder)
        model_str += '\n'
        model_str += str(self.dalle_decoder)

        return model_str

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        return chain(self.encoder.parameters(recurse=recurse), self.decoder.parameters(recurse=recurse))

    def train(self):
        self.encoder.train()
        self.decoder.train()

    def eval(self):
        self.encoder.eval()
        self.decoder.eval()

    def forward(self, x) -> Tensor:
        # x must be first mapped using map_pixels

        # encoding
        # z shape: [B, 3, 256, 256] -> [B, 2048, 32, 32]
        z = self.dalle_encoder(x)
        # z shape: [B, 2048, 32, 32] -> [B, 8192, 4, 4]
        z = self.encoder(z)

        # decoding
        # x shape: [B, 8192, 4, 4] -> [B, 128, 32, 32]
        x_ = self.decoder(z)
        # x shape: [B, 128, 32, 32] -> [B, 6, 256, 256]
        x_ = self.dalle_decoder(x_)
        return x_

    def __call__(self, x):
        return self.forward(x)