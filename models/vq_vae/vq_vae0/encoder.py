from collections import OrderedDict
from functools import partial
from typing import Tuple

import attr
import torch
from torch import nn


@attr.s(repr=False, eq=False)
class EncoderBlock(nn.Module):
    n_in: int = attr.ib(validator=lambda i, a, x: x >= 1)
    n_out: int = attr.ib(validator=lambda i, a, x: x >= 1 and x % 4 == 0)
    n_layers: int = attr.ib(validator=lambda i, a, x: x >= 1)

    def __attrs_post_init__(self) -> None:
        super().__init__()
        self.n_hid = self.n_out // 4
        self.post_gain = 1 / (self.n_layers ** 2)

        self.id_path = nn.Conv2d(in_channels=self.n_in, out_channels=self.n_out, kernel_size=1) \
            if self.n_in != self.n_out else nn.Identity()
        self.res_path = nn.Sequential(OrderedDict([
            ('relu_1', nn.ReLU()),
            ('conv_1', nn.Conv2d(in_channels=self.n_in, out_channels=self.n_hid, kernel_size=3, padding=1)),
            ('relu_2', nn.ReLU()),
            ('conv_2', nn.Conv2d(in_channels=self.n_hid, out_channels=self.n_hid, kernel_size=3, padding=1)),
            ('relu_3', nn.ReLU()),
            ('conv_3', nn.Conv2d(in_channels=self.n_hid, out_channels=self.n_hid, kernel_size=3, padding=1)),
            ('relu_4', nn.ReLU()),
            ('conv_4', nn.Conv2d(in_channels=self.n_hid, out_channels=self.n_out, kernel_size=1)),
        ]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.id_path(x) + self.post_gain * self.res_path(x)


@attr.s(repr=False, eq=False)
class Encoder(nn.Module):
    group_count: int = attr.ib()
    n_hid: int = attr.ib(default=256, validator=lambda i, a, x: x >= 64)
    n_blk_per_group: int = attr.ib(default=2, validator=lambda i, a, x: x >= 1)
    input_channels: int = attr.ib(default=3, validator=lambda i, a, x: x >= 1)
    # codebook dim
    n_init: int = attr.ib(default=128, validator=lambda i, a, x: x > 0)

    # whether downsample spatially
    downsample: bool = attr.ib(default=True)

    def __attrs_post_init__(self):
        super().__init__()

        blk_range = range(self.n_blk_per_group)
        n_layers = self.group_count * self.n_blk_per_group

        make_conv = nn.Conv2d
        make_blk = partial(EncoderBlock, n_layers=n_layers)

        def make_grp(gid: int, n: int, n_prev, downsample: bool = True) -> Tuple[str, nn.Sequential]:
            blks = [(f'block_{i + 1}', make_blk(n_in=n_prev if i == 0 else n, n_out=n)) for
                    i in blk_range]
            if downsample:
                blks += [('pool', nn.MaxPool2d(kernel_size=2))]
            return f'group_{gid}', nn.Sequential(OrderedDict(blks))

        encode_blks = [('input', make_conv(in_channels=self.input_channels, out_channels=self.n_hid, kernel_size=7, padding=3))]
        n, n_prev = self.n_hid, self.n_hid
        for gid in range(1, self.group_count):
            encode_blks.append(make_grp(gid=gid, n=n, n_prev=n_prev, downsample=self.downsample))
            n_prev = n
            n = (gid + 1) * self.n_hid
        encode_blks.append(make_grp(gid=self.group_count, n=n, n_prev=n_prev, downsample=False))
        encode_blks.append(
            ('output', nn.Sequential(OrderedDict([
                ('relu', nn.ReLU()),
                ('conv',
                 make_conv(in_channels=self.group_count * self.n_hid, out_channels=self.n_init, kernel_size=1))
            ]))))
        self.blocks = nn.Sequential(OrderedDict(encode_blks))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.blocks(inputs)
