from typing import Tuple

import attr
import torch
from torch import nn

from models.vq_vae.dalle0 import Encoder, Decoder
from models.vq_vae.dalle0.layer import VectorQuantizerEMA, VectorQuantizer


@attr.s(repr=False, eq=False)
class VqVae(nn.Module):
    group_count: int = attr.ib()
    # init hidden features for encoder
    n_hid: int = attr.ib(default=256, validator=lambda i, a, x: x >= 64)
    # codebook dim, also init hidden features for decoder
    n_init: int = attr.ib(default=128, validator=lambda i, a, x: x >= 8)
    # number of blocks in each group
    n_blk_per_group: int = attr.ib(default=2, validator=lambda i, a, x: x >= 1)

    input_channels: int = attr.ib(default=3, validator=lambda i, a, x: x >= 1)
    output_channels: int = attr.ib(default=3, validator=lambda i, a, x: x >= 1)
    # number of code books
    vocab_size: int = attr.ib(default=8192, validator=lambda i, a, x: x >= 512)

    # codes commit cost
    commitment_cost: float = attr.ib(default=0.25, validator=lambda i, a, x: x >= 0.0)
    # use EMA for quantization
    decay: float = attr.ib(default=0.99, validator=lambda i, a, x: x >= 0.0)

    def __attrs_post_init__(self):
        super().__init__()

        self.encoder = Encoder(
            group_count=self.group_count,
            n_hid=self.n_hid,
            n_blk_per_group=self.n_blk_per_group,
            input_channels=self.input_channels,
            n_init=self.n_init)

        if self.decay > 0.0:
            self.vq_vae = VectorQuantizerEMA(
                num_embeddings=self.vocab_size,
                embedding_dim=self.n_init,
                commitment_cost=self.commitment_cost,
                decay=self.decay)
        else:
            self.vq_vae = VectorQuantizer(
                num_embeddings=self.vocab_size,
                embedding_dim=self.n_init,
                commitment_cost=self.commitment_cost)

        self.decoder = Decoder(
            group_count=self.group_count,
            n_init=self.n_init,
            n_hid=self.n_hid,
            n_blk_per_group=self.n_blk_per_group,
            output_channels=self.output_channels
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor]:
        z = self.encoder(x)
        loss, quantized, perplexity, _ = self.vq_vae(z)
        x_recon = self.decoder(quantized)
        return loss, x_recon, perplexity

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        _, quantized, _, _ = self.vq_vae(x)
        return quantized

    @torch.no_grad()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)
