import torch.nn.functional as F
from torch import nn

from models.vq_vae.dalle0.layer import ResidualStack, VectorQuantizerEMA, VectorQuantizer


class Encoder(nn.Module):
    def __init__(self, which_conv, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(Encoder, self).__init__()

        self._conv_1 = which_conv(in_channels=in_channels,
                                  out_channels=num_hiddens // 2,
                                  kernel_size=4,
                                  stride=2, padding=1)
        self._conv_2 = which_conv(in_channels=num_hiddens // 2,
                                  out_channels=num_hiddens,
                                  kernel_size=4,
                                  stride=2, padding=1)
        self._conv_3 = which_conv(in_channels=num_hiddens,
                                  out_channels=num_hiddens,
                                  kernel_size=3,
                                  stride=1, padding=1)
        self._residual_stack = ResidualStack(which_conv=which_conv,
                                             in_channels=num_hiddens,
                                             num_hiddens=num_hiddens,
                                             num_residual_layers=num_residual_layers,
                                             num_residual_hiddens=num_residual_hiddens)

    def forward(self, inputs):
        x = self._conv_1(inputs)
        x = F.relu(x)

        x = self._conv_2(x)
        x = F.relu(x)

        x = self._conv_3(x)
        return self._residual_stack(x)


class Decoder(nn.Module):
    def __init__(self, which_conv, which_transpose_conv,
                 in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(Decoder, self).__init__()

        self._conv_1 = which_conv(in_channels=in_channels,
                                  out_channels=num_hiddens,
                                  kernel_size=3,
                                  stride=1, padding=1)

        self._residual_stack = ResidualStack(which_conv=which_conv,
                                             in_channels=num_hiddens,
                                             num_hiddens=num_hiddens,
                                             num_residual_layers=num_residual_layers,
                                             num_residual_hiddens=num_residual_hiddens)

        self._conv_trans_1 = which_transpose_conv(in_channels=num_hiddens,
                                                  out_channels=num_hiddens // 2,
                                                  kernel_size=4,
                                                  stride=2, padding=1)

        self._conv_trans_2 = which_transpose_conv(in_channels=num_hiddens // 2,
                                                  out_channels=3,
                                                  kernel_size=4,
                                                  stride=2, padding=1)

    def forward(self, inputs):
        x = self._conv_1(inputs)

        x = self._residual_stack(x)

        x = self._conv_trans_1(x)
        x = F.relu(x)

        return self._conv_trans_2(x)


class VqVae(nn.Module):
    def __init__(self, num_hiddens, num_residual_layers, num_residual_hiddens,
                 num_embeddings, embedding_dim, embedding_mul, commitment_cost, decay=0, is_video=False):
        super(VqVae, self).__init__()

        self.is_video = is_video
        if self.is_video:
            # we use conv3d and convtranspose3d
            which_conv = nn.Conv3d
            which_transpose_conv = nn.ConvTranspose3d
        else:
            # we use conv2d and convtranspose2d
            which_conv = nn.Conv2d
            which_transpose_conv = nn.ConvTranspose2d

        self._encoder = Encoder(which_conv, 3, num_hiddens,
                                num_residual_layers,
                                num_residual_hiddens)

        self._pre_vq_conv = which_conv(in_channels=num_hiddens,
                                       out_channels=embedding_dim,
                                       kernel_size=1,
                                       stride=1)
        if decay > 0.0:
            self._vq_vae = VectorQuantizerEMA(num_embeddings, embedding_dim * embedding_mul, commitment_cost, decay, is_video=is_video)
        else:
            self._vq_vae = VectorQuantizer(num_embeddings, embedding_dim * embedding_mul, commitment_cost, is_video=is_video)

        self._decoder = Decoder(which_conv,
                                which_transpose_conv,
                                embedding_dim,
                                num_hiddens,
                                num_residual_layers,
                                num_residual_hiddens)

    def forward(self, x):
        z = self._encoder(x)
        z = self._pre_vq_conv(z)

        # if video shape of z: [batch_size, embedding_dim, depth, height, width]
        # if image shape of z: [batch_size, embedding_dim, height, width]
        loss, quantized, perplexity, _ = self._vq_vae(z)

        x_recon = self._decoder(quantized)

        return loss, x_recon, perplexity
