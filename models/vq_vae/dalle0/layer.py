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
        self.embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
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
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize and unflatten
        quantized = rearrange(torch.matmul(encodings, self.embedding.weight), self.p_flatten,
                              **kwargs)

        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self._commitment_cost * e_latent_loss

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
        # input shape: [B, C, H, W]
        # embedding_dim = C

        self.embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self.embedding.weight.data.normal_()

        self.register_buffer('ema_cluster_size', torch.zeros(self.num_embeddings))
        self.ema_w = nn.Parameter(torch.Tensor(self.num_embeddings, self.embedding_dim))
        self.ema_w.data.normal_()

        self._p_depth_last = 'b c h w -> b h w c'
        self._p_space_last = 'b h w c -> b c h w'
        self._p_group = 'b h w c -> (b h w) c'
        self._p_flatten = '(b h w) c -> b h w c'

    def _forward(self, inputs, flat_input, **kwargs):
        # inputs shape: [b h w c]
        # flat_input shape: already flattened shape [b*h*w, c]

        # Calculate distances
        distances = (torch.sum(flat_input ** 2, dim=1, keepdim=True)
                     + torch.sum(self.embedding.weight ** 2, dim=1)
                     - 2 * torch.matmul(flat_input, self.embedding.weight.t()))

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize and unflatten
        quantized = rearrange(torch.matmul(encodings, self.embedding.weight), self._p_flatten,
                              **kwargs)

        # Use EMA to update the embedding vectors
        if self.training:
            self._ema_cluster_size = self._ema_cluster_size * self._decay + \
                                     (1 - self._decay) * torch.sum(encodings, 0)

            # Laplace smoothing of the cluster size
            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size = (
                    (self._ema_cluster_size + self._epsilon)
                    / (n + self._num_embeddings * self._epsilon) * n)

            dw = torch.matmul(encodings.t(), flat_input)
            self.ema_w = nn.Parameter(self.ema_w * self._decay + (1 - self._decay) * dw)

            self.embedding.weight = nn.Parameter(self.ema_w / self._ema_cluster_size.unsqueeze(1))

        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        loss = self._commitment_cost * e_latent_loss

        # Straight Through Estimator
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return loss, quantized, perplexity, encodings

    def forward(self, inputs):
        # convert inputs from BCHW -> BHWC
        inputs = rearrange(inputs, self._p_depth_last)
        # convert inputs from BHWC -> (BxHxW)C
        flat_input = rearrange(inputs, self._p_group)

        kwargs = {
            'b': inputs.size()[0],
            'h': inputs.size()[1],
            'w': inputs.size()[2],
            'c': inputs.size()[3]
        }
        loss, quantized, perplexity, encodings = self._forward(inputs, flat_input, **kwargs)

        # convert quantized from BHWC -> BCHW
        quantized = rearrange(quantized, self._p_space_last)
        return loss, quantized, perplexity, encodings


@attr.s(repr=False, eq=False)
class Residual(nn.Module):
    which_conv: nn.Module = attr.ib()
    in_channels: int = attr.ib()
    num_hiddens: int = attr.ib()
    num_residual_hiddens: int = attr.ib()

    def __attrs_post_init__(self):
        self.block = nn.Sequential(
            nn.ReLU(True),
            self.which_conv(in_channels=self.in_channels,
                            out_channels=self.num_residual_hiddens,
                            kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(True),
            self.which_conv(in_channels=self.num_residual_hiddens,
                            out_channels=self.num_hiddens,
                            kernel_size=1, stride=1, bias=False)
        )

    def forward(self, x):
        return x + self.block(x)


@attr.s(repr=False, eq=False)
class ResidualStack(nn.Module):
    which_conv: nn.Module = attr.ib()
    in_channels: int = attr.ib()
    num_hiddens: int = attr.ib()
    num_residual_hiddens: int = attr.ib()
    num_residual_layers: int = attr.ib()

    def __attrs_post_init__(self):
        self.layers = nn.ModuleList([
            Residual(which_conv=self.which_conv,
                     in_channels=self.in_channels,
                     num_hiddens=self.num_hiddens,
                     num_residual_hiddens=self.num_residual_hiddens)
            for _ in range(self._num_residual_layers)])

    def forward(self, x):
        for i in range(self.num_residual_layers):
            x = self.layers[i](x)
        return F.relu(x)
