from collections import OrderedDict

import attr
import numpy as np
import torch
from torch import nn


@attr.s(eq=False, repr=False)
class Decoder2(nn.Module):
    group_count: int = attr.ib()
    # codebook dim from first VQ-VAE encoder
    n_hid: int = attr.ib(default=512, validator=lambda i, a, x: x > 0)

    # code book dim
    n_init: int = attr.ib(default=128, validator=lambda i, a, x: x >= 8)
    # video max frames
    sequence_length: int = attr.ib(default=16, validator=lambda i, a, x: x > 0)
    # conv groups, to reduce memory cost
    n_group: int = attr.ib(default=4, validator=lambda i, a, x: x > 0)

    def __attrs_post_init__(self) -> None:
        super().__init__()

        make_conv = nn.Conv2d
        # input shape: (b, d'*c, h, w)
        # only transposed conv layer applied
        n_prev, n = self.n_init, self.n_hid * self.sequence_length // (4 ** (self.group_count-1))

        decode_blks = [
            ('input', nn.Sequential(OrderedDict([
                ('relu', nn.ReLU()),
                ('conv', make_conv(in_channels=n_prev, out_channels=n, kernel_size=1))
            ])))
        ]
        for gid in range(1, self.group_count-1):
            n_prev = n
            n *= 4
            decode_blks.append((f'group_{gid}', nn.Sequential(OrderedDict([
                    ('conv', make_conv(in_channels=n_prev, out_channels=n, kernel_size=3, padding=1, groups=n_prev // self.n_group)),
                    ('relu', nn.ReLU())
                ]))))
        n_prev = n
        n *= 4
        decode_blks.append((f'group_{self.group_count-1}', nn.Sequential(OrderedDict([
            ('conv', make_conv(in_channels=n_prev, out_channels=n, kernel_size=3, padding=1, groups=n_prev // self.n_group))
        ]))))
        self.blocks = nn.Sequential(OrderedDict(decode_blks))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)
