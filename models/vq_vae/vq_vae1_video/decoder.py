from collections import OrderedDict
from functools import partial
from typing import Tuple

import attr
import torch
from einops import rearrange
from torch import nn

from models.vq_vae.vq_vae0.decoder import DecoderBlock


@attr.s(eq=False, repr=False)
class Decoder1(nn.Module):
    group_count: int = attr.ib()

    # code book dim
    n_init: int = attr.ib(default=128, validator=lambda i, a, x: x >= 8)

    n_hid: int = attr.ib(default=256, validator=lambda i, a, x: x >= 64)
    n_blk_per_group: int = attr.ib(default=2, validator=lambda i, a, x: x >= 1)
    output_channels: int = attr.ib(default=3, validator=lambda i, a, x: x >= 1)

    # video max frames
    sequence_length: int = attr.ib(default=16, validator=lambda i, a, x: x > 0)

    # whether upsample spatially
    upsample: bool = attr.ib(default=True)

    def __attrs_post_init__(self) -> None:
        super().__init__()

        blk_range = range(self.n_blk_per_group)
        n_layers = self.group_count * self.n_blk_per_group
        make_conv = nn.Conv2d
        make_blk = partial(DecoderBlock, n_layers=n_layers)

        def make_grp(gid: int, n: int, n_prev: int, upsample: bool = True) -> Tuple[str, nn.Sequential]:

            blks = [(f'block_{i + 1}', make_blk(n_in=n_prev if i == 0 else n, n_out=n)) for
                    i in blk_range]
            if upsample:
                blks += [('upsample', nn.Upsample(scale_factor=2, mode='nearest'))]
            return f'spatial_{gid}', nn.Sequential(OrderedDict(blks))

        n = self.group_count * self.n_hid
        # decode spatially
        # shape of input tensor: [b, seq_length * n_channel, h, w] -> [b * seq_length, n_channel, h, w]
        decode_blks = []
        n_prev = self.n_init
        for gid in range(1, self.group_count):
            decode_blks.append(make_grp(gid=gid, n=n, n_prev=n_prev, upsample=self.upsample))
            n_prev = n
            n = (self.group_count - gid) * self.n_hid
        decode_blks.append(make_grp(gid=self.group_count, n=self.n_hid, n_prev=n_prev, upsample=False))
        decode_blks.append(
            ('output', nn.Sequential(OrderedDict([
                ('relu_1', nn.ReLU()),
                ('conv_1', make_conv(in_channels=self.n_hid, out_channels=4 * self.n_hid, kernel_size=7, padding=3)),
                ('relu_2', nn.ReLU()),
                ('conv_2', make_conv(in_channels=4 * self.n_hid, out_channels=self.output_channels * self.sequence_length, kernel_size=7, padding=3)),
            ]))))
        self.blocks = nn.Sequential(OrderedDict(decode_blks))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)
