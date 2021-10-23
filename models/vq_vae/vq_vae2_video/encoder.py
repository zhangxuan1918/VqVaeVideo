from collections import OrderedDict

import attr
import numpy as np
import torch
from torch import nn


@attr.s(repr=False, eq=False)
class Encoder2(nn.Module):
    group_count: int = attr.ib()

    # codebook dim from first VQ-VAE encoder
    n_hid: int = attr.ib(default=512, validator=lambda i, a, x: x > 0)
    # codebook dim
    n_init: int = attr.ib(default=128, validator=lambda i, a, x: x > 0)
    # video max frames, must be 2^v
    sequence_length: int = attr.ib(default=16, validator=lambda i, a, x: np.exp2(np.log2(x).astype(int)) == x)
    # conv groups, to reduce memory cost
    n_group: int = attr.ib(default=4, validator=lambda i, a, x: x > 0)

    def __attrs_post_init__(self):
        super().__init__()

        make_conv = nn.Conv2d
        # input shape: (b, d, c, h, w) -> (b, d*c, h, w)
        # only conv layer applied
        n_prev = self.n_hid * self.sequence_length
        n = n_prev // 4
        encode_blks = []
        for gid in range(1, self.group_count):
            encode_blks.append(
                (f'group_{gid}', nn.Sequential(OrderedDict([
                    ('conv', make_conv(in_channels=n_prev, out_channels=n, kernel_size=3, padding=1, groups=n_prev // self.n_group)),
                    ('relu', nn.ReLU())
                ]))))
            n_prev = n
            n //= 4
        encode_blks.append(
            ('output', nn.Sequential(OrderedDict([
                ('conv', make_conv(in_channels=n_prev, out_channels=self.n_init, kernel_size=1))
            ]))))
        self.blocks = nn.Sequential(OrderedDict(encode_blks))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.blocks(inputs)
