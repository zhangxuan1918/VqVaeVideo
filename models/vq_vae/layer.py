import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange


class VectorQuantizer(nn.Module):

    def __init__(self, num_embeddings, embedding_dim, commitment_cost, is_video=False):
        # if is_video = False => we are dealing with image
        # input shape: [B, C, H, W]
        # embedding_dim = C

        # if is_video = True => we are dealing with video
        # input shape: [B, C, D, H, W]
        # embedding_dim = C * D

        super(VectorQuantizer, self).__init__()

        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings

        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.uniform_(-1 / self._num_embeddings, 1 / self._num_embeddings)
        self._commitment_cost = commitment_cost

        self._is_video = is_video
        if is_video:
            self._p_depth_last = 'b c d h w -> b h w c d'
            self._p_space_last = 'b h w c d -> b c d h w'
            self._p_group = 'b h w c d -> (b h w) (c d)'
            self._p_flatten = '(b h w) (c d) -> b h w c d'
        else:
            self._p_depth_last = 'b c h w -> b h w c'
            self._p_space_last = 'b h w c -> b c h w'
            self._p_group = 'b h w c -> (b h w) c'
            self._p_flatten = '(b h w) c -> b h w c'

    def _forward(self, inputs, flat_input, **kwargs):
        # inputs shape: [b h w c d] or [b h w c]
        # flat_input shape: already flattened shape [b*h*w, c * d] or [b*h*w, c]

        # Calculate distances
        distances = (torch.sum(flat_input ** 2, dim=1, keepdim=True)
                     + torch.sum(self._embedding.weight ** 2, dim=1)
                     - 2 * torch.matmul(flat_input, self._embedding.weight.t()))

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize and unflatten
        quantized = rearrange(torch.matmul(encodings, self._embedding.weight), self._p_flatten,
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
        # convert inputs from BCDHW -> BHWCD or BCHW -> BHWC
        inputs = rearrange(inputs, self._p_depth_last)
        # convert inputs from BHWCD -> (BxHxW)(CxD) or BHWC -> (BxHxW)C
        flat_input = rearrange(inputs, self._p_group)

        if self._is_video:
            kwargs = {
                'b': inputs.size()[0],
                'h': inputs.size()[1],
                'w': inputs.size()[2],
                'c': inputs.size()[3],
                'd': inputs.size()[4],
            }
        else:
            kwargs = {
                'b': inputs.size()[0],
                'h': inputs.size()[1],
                'w': inputs.size()[2],
                'c': inputs.size()[3]
            }
        loss, quantized, perplexity, encodings = self._forward(inputs, flat_input, **kwargs)

        # convert quantized from BHWCD -> BCDHW or BHWC -> BCHW
        quantized = rearrange(quantized, self._p_flatten)
        return loss, quantized, perplexity, encodings


class VectorQuantizerEMA(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost, decay, epsilon=1e-5, is_video=False):
        # if is_video = False => we are dealing with image
        # input shape: [B, C, H, W]
        # embedding_dim = C

        # if is_video = True => we are dealing with video
        # input shape: [B, C, D, H, W]
        # embedding_dim = C * D
        super(VectorQuantizerEMA, self).__init__()

        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings

        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.normal_()
        self._commitment_cost = commitment_cost

        self.register_buffer('_ema_cluster_size', torch.zeros(num_embeddings))
        self._ema_w = nn.Parameter(torch.Tensor(num_embeddings, self._embedding_dim))
        self._ema_w.data.normal_()

        self._decay = decay
        self._epsilon = epsilon

        self._is_video = is_video
        if is_video:
            self._p_depth_last = 'b c d h w -> b h w c d'
            self._p_space_last = 'b h w c d -> b c d h w'
            self._p_group = 'b h w c d -> (b h w) (c d)'
            self._p_flatten = '(b h w) (c d) -> b h w c d'
        else:
            self._p_depth_last = 'b c h w -> b h w c'
            self._p_space_last = 'b h w c -> b c h w'
            self._p_group = 'b h w c -> (b h w) c'
            self._p_flatten = '(b h w) c -> b h w c'

    def _forward(self, inputs, flat_input, **kwargs):
        # inputs shape: [b h w c d] or [b h w c]
        # flat_input shape: already flattened shape [b*h*w, c * d] or [b*h*w, c]


        # Calculate distances
        distances = (torch.sum(flat_input ** 2, dim=1, keepdim=True)
                     + torch.sum(self._embedding.weight ** 2, dim=1)
                     - 2 * torch.matmul(flat_input, self._embedding.weight.t()))

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize and unflatten
        quantized = rearrange(torch.matmul(encodings, self._embedding.weight), self._p_flatten,
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
            self._ema_w = nn.Parameter(self._ema_w * self._decay + (1 - self._decay) * dw)

            self._embedding.weight = nn.Parameter(self._ema_w / self._ema_cluster_size.unsqueeze(1))

        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        loss = self._commitment_cost * e_latent_loss

        # Straight Through Estimator
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return loss, quantized, perplexity, encodings

    def forward(self, inputs):
        # convert inputs from BCDHW -> BHWCD or BCHW -> BHWC
        inputs = rearrange(inputs, self._p_depth_last)
        # convert inputs from BHWCD -> (BxHxW)(CxD) or BHWC -> (BxHxW)C
        flat_input = rearrange(inputs, self._p_group)

        if self._is_video:
            kwargs = {
                'b': inputs.size()[0],
                'h': inputs.size()[1],
                'w': inputs.size()[2],
                'c': inputs.size()[3],
                'd': inputs.size()[4],
            }
        else:
            kwargs = {
                'b': inputs.size()[0],
                'h': inputs.size()[1],
                'w': inputs.size()[2],
                'c': inputs.size()[3]
            }
        loss, quantized, perplexity, encodings = self._forward(inputs, flat_input, **kwargs)

        # convert quantized from BHWCD -> BCDHW or BHWC -> BCHW
        quantized = rearrange(quantized, self._p_space_last)
        return loss, quantized, perplexity, encodings


class Residual(nn.Module):
    def __init__(self, which_conv, in_channels, num_hiddens, num_residual_hiddens):
        super(Residual, self).__init__()
        self._block = nn.Sequential(
            nn.ReLU(True),
            which_conv(in_channels=in_channels,
                       out_channels=num_residual_hiddens,
                       kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(True),
            which_conv(in_channels=num_residual_hiddens,
                       out_channels=num_hiddens,
                       kernel_size=1, stride=1, bias=False)
        )

    def forward(self, x):
        return x + self._block(x)


class ResidualStack(nn.Module):
    def __init__(self, which_conv, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(ResidualStack, self).__init__()
        self._num_residual_layers = num_residual_layers
        self._layers = nn.ModuleList([Residual(which_conv, in_channels, num_hiddens, num_residual_hiddens)
                                      for _ in range(self._num_residual_layers)])

    def forward(self, x):
        for i in range(self._num_residual_layers):
            x = self._layers[i](x)
        return F.relu(x)
