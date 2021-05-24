from collections import OrderedDict
from typing import Tuple

import attr
import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange


@attr.s(repr=False, eq=False)
class VectorQuantizer(nn.Module):
    num_embeddings: int = attr.ib(default=8192, validator=lambda i, a, x: x >= 512)
    embedding_dim: int = attr.ib(default=256, validator=lambda i, a, x: x > 128)
    commitment_cost: float = attr.ib(default=0.25, validator=lambda i, a, x: x >= 0.0)

    def __attrs_post_init__(self):
        super().__init__()
        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embedding.weight.data.uniform_(-1 / self.num_embeddings, 1 / self.num_embeddings)
        self.p_depth_last = 'b c h w -> b h w c'
        self.p_space_last = 'b h w c -> b c h w'
        self.p_group = 'b h w c -> (b h w) c'
        self.p_flatten = '(b h w) c -> b h w c'

    def _forward(self, inputs, flat_input, **kwargs):
        # inputs shape: [b h w c]
        # flat_input shape: already flattened shape [b*h*w, c]

        # Calculate distances
        distances = (torch.sum(flat_input ** 2, dim=1, keepdim=True)
                     + torch.sum(self.embedding.weight ** 2, dim=1)
                     - 2 * torch.matmul(flat_input, self.embedding.weight.t()))

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize and unflatten
        quantized = rearrange(torch.matmul(encodings, self.embedding.weight), self.p_flatten,
                              **kwargs)

        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return loss, quantized, perplexity, encodings

    def forward(self, inputs):
        # convert inputs from BCHW -> BHWC
        inputs = rearrange(inputs, self.p_depth_last)
        # convert inputs from BHWC -> (BxHxW)C
        flat_input = rearrange(inputs, self.p_group)
        kwargs = {
            'b': inputs.size()[0],
            'h': inputs.size()[1],
            'w': inputs.size()[2],
            'c': inputs.size()[3]
        }
        loss, quantized, perplexity, encodings = self._forward(inputs, flat_input, **kwargs)

        # convert quantized from BHWC -> BCHW
        quantized = rearrange(quantized, self.p_flatten)
        return loss, quantized, perplexity, encodings


@attr.s(repr=False, eq=False)
class VectorQuantizerEMA(nn.Module):
    num_embeddings: int = attr.ib(default=8192, validator=lambda i, a, x: x >= 512)
    embedding_dim: int = attr.ib(default=256, validator=lambda i, a, x: x > 128)
    commitment_cost: float = attr.ib(default=0.25, validator=lambda i, a, x: x >= 0.0)
    decay: float = attr.ib(default=0.99, validator=lambda i, a, x: 0.9 <= x < 1.0)
    epsilon: float = attr.ib(default=1e-5, validator=lambda i, a, x: x > 0)

    def __attrs_post_init__(self):
        super().__init__()
        # input shape: [B, C, H, W]
        # embedding_dim = C
        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embedding.weight.data.normal_()

        self.register_buffer('ema_cluster_size', torch.zeros(self.num_embeddings))
        self.ema_w = nn.Parameter(torch.Tensor(self.num_embeddings, self.embedding_dim))
        self.ema_w.data.normal_()

        self.p_depth_last = 'b c h w -> b h w c'
        self.p_space_last = 'b h w c -> b c h w'
        self.p_group = 'b h w c -> (b h w) c'
        self.p_flatten = '(b h w) c -> b h w c'

    def _forward(self, inputs: torch.Tensor, flat_input: torch.Tensor, **kwargs) -> Tuple[torch.Tensor]:
        # inputs shape: [b h w c]
        # flat_input shape: already flattened shape [b*h*w, c]

        # Calculate distances
        distances = (torch.sum(flat_input ** 2, dim=1, keepdim=True)
                     + torch.sum(self.embedding.weight ** 2, dim=1)
                     - 2 * torch.matmul(flat_input, self.embedding.weight.t()))

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize and unflatten
        quantized = rearrange(torch.matmul(encodings, self.embedding.weight), self.p_flatten,
                              **kwargs)

        # Use EMA to update the embedding vectors
        if self.training:
            self.ema_cluster_size = self.ema_cluster_size * self.decay + \
                                     (1 - self.decay) * torch.sum(encodings, 0)

            # Laplace smoothing of the cluster size
            n = torch.sum(self.ema_cluster_size.data)
            self.ema_cluster_size = ((self.ema_cluster_size + self.epsilon)
                    / (n + self.num_embeddings * self.epsilon) * n)

            dw = torch.matmul(encodings.t(), flat_input)
            self.ema_w = nn.Parameter(self.ema_w * self.decay + (1 - self.decay) * dw)

            self.embedding.weight = nn.Parameter(self.ema_w / self.ema_cluster_size.unsqueeze(1))

        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        loss = self.commitment_cost * e_latent_loss

        # Straight Through Estimator
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return loss, quantized, perplexity, encodings

    def forward(self, inputs) -> Tuple[torch.Tensor]:
        # convert inputs from BCHW -> BHWC
        inputs = rearrange(inputs, self.p_depth_last)
        # convert inputs from BHWC -> (BxHxW)C
        flat_input = rearrange(inputs, self.p_group)

        kwargs = {
            'b': inputs.size()[0],
            'h': inputs.size()[1],
            'w': inputs.size()[2],
            'c': inputs.size()[3]
        }
        loss, quantized, perplexity, encodings = self._forward(inputs, flat_input, **kwargs)

        # convert quantized from BHWC -> BCHW
        quantized = rearrange(quantized, self.p_space_last)
        return loss, quantized, perplexity, encodings

