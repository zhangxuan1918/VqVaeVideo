from typing import Tuple

import attr
import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

from models.vq_vae.dalle import EncoderDalle, DecoderDalle


@attr.s(eq=False, repr=False)
class VqVae(nn.Module):
    n_hid: int = attr.ib(default=256, validator=lambda i, a, x: x >= 64)
    n_blk_per_group: int = attr.ib(default=2, validator=lambda i, a, x: x >= 1)
    input_channels: int = attr.ib(default=3, validator=lambda i, a, x: x >= 1)
    vocab_size: int = attr.ib(default=8192, validator=lambda i, a, x: x >= 512)

    n_init: int = attr.ib(default=128, validator=lambda i, a, x: x >= 8)
    output_channels: int = attr.ib(default=3, validator=lambda i, a, x: x >= 1)

    device: torch.device = attr.ib(default=torch.device('cpu'))
    requires_grad: bool = attr.ib(default=False)
    use_mixed_precision: bool = attr.ib(default=True)

    temperature: float = attr.ib(default=0.9, validator=lambda i, a, x: 0.0 <= x <= 1.0)
    straight_through: bool = attr.ib(default=False)
    kl_div_loss_weight: float = attr.ib(default=0.5)

    def __attrs_post_init__(self) -> None:
        super().__init__()

        self.encoder = EncoderDalle(
            n_hid=self.n_hid,
            n_blk_per_group=self.n_blk_per_group,
            input_channels=self.input_channels,
            vocab_size=self.vocab_size,
            device=self.device,
            requires_grad=self.requires_grad,
            use_mixed_precision=self.use_mixed_precision
        )
        self.decoder = DecoderDalle(
            n_init=self.n_init,
            n_hid=self.n_hid,
            n_blk_per_group=self.n_blk_per_group,
            output_channels=self.output_channels,
            vocab_size=self.vocab_size,
            device=self.device,
            requires_grad=self.requires_grad,
            use_mixed_precision=self.use_mixed_precision
        )

    @torch.no_grad()
    def encode(self, x):
        """
        encode image
        :return: encoded image, shape [B, H, W]
        """
        z_logits = self.encoder(x)
        z = torch.argmax(z_logits, dim=1)
        return z

    @torch.no_grad()
    def decode(self, z):
        """
        decode
        :return: decoded image, shape [B, C, H, W]
        """
        z = F.one_hot(z, num_classes=self.vocab_size).permute(0, 3, 1, 2).float()
        x = self.decoder(z).float()
        return x

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor]:
        z_logits = self.encoder(x)

        soft_one_hot = F.gumbel_softmax(z_logits, tau=self.temperature, dim=1, hard=self.straight_through)
        x_recon = self.decoder(soft_one_hot)

        # kl divergence
        z_logits = rearrange(z_logits, 'b n h w -> b (h w) n')
        log_qy = F.log_softmax(z_logits, dim=-1)
        log_uniform = torch.log(torch.tensor([1. / self.vocab_size], device=self.device))
        kl_div = F.kl_div(log_uniform, log_qy, None, None, 'batchmean', log_target=True)

        return x_recon, kl_div * self.kl_div_loss_weight
