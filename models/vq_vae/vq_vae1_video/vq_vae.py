from typing import Tuple

import attr
import torch
from einops import rearrange
from torch import nn

from models.vq_vae.vq_vae0.layer import VectorQuantizerEMA, VectorQuantizer
from models.vq_vae.vq_vae1_video import Encoder1, Decoder1


@attr.s(repr=False, eq=False)
class VqVae1(nn.Module):
    """
    input shape: (b, d, c, h, w) -> (b, d*c, h, w)
    """
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

    # video max frames
    sequence_length: int = attr.ib(default=16, validator=lambda i, a, x: x > 0)

    # whether downsample spatially in encoder
    downsample: bool = attr.ib(default=True)
    # whether upsample spatially in decoder
    upsample: bool = attr.ib(default=True)

    def __attrs_post_init__(self):
        super().__init__()

        self.encoder = Encoder1(
            group_count=self.group_count,
            n_hid=self.n_hid,
            n_blk_per_group=self.n_blk_per_group,
            input_channels=self.input_channels,
            n_init=self.n_init,
            sequence_length=self.sequence_length,
            downsample=self.downsample
        )

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

        self.decoder = Decoder1(
            group_count=self.group_count,
            n_init=self.n_init,
            n_hid=self.n_hid,
            n_blk_per_group=self.n_blk_per_group,
            sequence_length=self.sequence_length,
            output_channels=self.output_channels,
            upsample=self.upsample
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor]:
        x = rearrange(x, 'b d h w c -> b (d c) h w')
        z = self.encoder(x)
        loss, quantized, perplexity, _ = self.vq_vae(z)
        x_recon = self.decoder(quantized)
        x_recon = rearrange(x_recon, 'b (d c) h w -> b d h w c', d=self.sequence_length, c=self.output_channels)
        return loss, x_recon, perplexity

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        _, _, _, codes = self.vq_vae(z)
        return codes

    @torch.no_grad()
    def decode(self, encode_indices: torch.Tensor) -> torch.Tensor:
        b, h, w = encode_indices.size()
        encode_indices = rearrange(encode_indices, 'b h w -> (b h w) 1').to(torch.int64)
        encodings = torch.zeros(encode_indices.shape[0], self.vq_vae.num_embeddings, device=encode_indices.device)
        encodings.scatter_(1, encode_indices, 1)
        quantized = rearrange(torch.matmul(encodings, self.vq_vae.embedding.weight), self.vq_vae.p_flatten,
                              b=b, h=h, w=w)
        quantized = rearrange(quantized, self.vq_vae.p_space_last)
        return self.decoder(quantized)
