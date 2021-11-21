"""
taken from: https://github.com/karpathy/minGPT/
GPT model:
- the initial stem consists of a combination of token encoding and a positional encoding
- the meat of it is a uniform sequence of Transformer blocks
    - each Transformer is a sequential combination of a 1-hidden-layer MLP block and a self-attention block
    - all blocks feed into a central residual pathway similar to resnets
- the final decoder is a linear projection into a vanilla Softmax classifier
"""
from collections import OrderedDict
from typing import Optional

import attr
import numpy as np
import torch
from einops import rearrange
from torch import nn
from torch.nn import functional as F

from models.transformer.gpt.layer import CausalSelfAttentionBlock


@attr.s(repr=False, eq=False)
class GPT(nn.Module):
    vocab_size: int = attr.ib()
    max_seq_length: int = attr.ib()
    # size of conditional tokens, these token are fixed
    cond_seq_length: int = attr.ib()
    num_layers: int = attr.ib(default=12)
    num_heads: int = attr.ib(default=8)
    embed_dim: int = attr.ib(default=256)
    embed_dropout_prob: float = attr.ib(default=1.0, validator=lambda i, a, x: 0 <= x <= 1.0)
    img_dim_x: int = attr.ib(default=32)
    img_dim_y: int = attr.ib(default=32)
    mlp_dropout_prob: float = attr.ib(default=0.0, validator=lambda i, a, x: 0.0 <= x < 1.0)
    attn_dropout_prob: float = attr.ib(default=0.0, validator=lambda i, a, x: 0.0 <= x < 1.0)
    pretrained_visual_embed_path: str = attr.ib(default=None)
    label_smoothing: float = attr.ib(default=0.5, validator=lambda i, a, x: 0.0 <= x < 1.0)

    def __attrs_post_init__(self):
        super().__init__()

        # input embedding
        self._set_visual_embed()
        self.embed_drop = nn.Dropout(self.embed_dropout_prob)

        # 2dim embedding for input image
        self.pos_x_embed = nn.Parameter(torch.zeros(1, self.img_dim_x, self.embed_dim))
        self.pos_y_embed = nn.Parameter(torch.zeros(1, self.img_dim_y, self.embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.max_seq_length, self.embed_dim))

        # transformer
        trans_blks = [(f'group_{i}', CausalSelfAttentionBlock(
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                attn_dropout_prob=self.attn_dropout_prob,
                mlp_dropout_prob=self.mlp_dropout_prob))
              for i in range(self.num_layers)]
        self.blocks = nn.Sequential(OrderedDict(trans_blks))

        # attention mask
        # ratio between tokens and conditional tokens
        self.r_x_to_c = self.max_seq_length // self.cond_seq_length - 1
        self.attn_mask = torch.triu(torch.ones(self.max_seq_length, self.max_seq_length)).transpose(0, 1)
        self.attn_mask = self.attn_mask.float().masked_fill(self.attn_mask == 0, float('-inf')).masked_fill(self.attn_mask == 1, float(0.0))
        self.attn_mask[:, :self.cond_seq_length] = 0

        # decoder head
        self.layer_norm = nn.LayerNorm(self.embed_dim)
        self.head = nn.Linear(self.embed_dim, self.vocab_size, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    @staticmethod
    def load_numpy(np_path):
        with np.load(np_path) as data:
            return data['weights']

    def _set_visual_embed(self):
        self.tok_embed = nn.Embedding(self.vocab_size, self.embed_dim)
        if self.pretrained_visual_embed_path is not None:
            # weights provided
            self.tok_embed.weight.data.copy_(torch.from_numpy(self.load_numpy(self.pretrained_visual_embed_path)))
            self.tok_embed = self.tok_embed.requires_grad_(requires_grad=False)

    def forward(self, x, c, loc_h, loc_w):
        """
        :param x: tokens from sub frame 1 to N-1, shape [b, l2, embed_dim]
        :param c: tokens from sub frame 0, used as condition for x, shape [b, l1, embed_dim]
        :param loc_h: loc info for sub frames along x axis
        :param loc_w: loc info for sub frames along y axis
        :return:
        """

        pos_c_embed = self.pos_x_embed[:, loc_h, :] + self.pos_y_embed[:, loc_w, :]
        pos_x_embed = pos_c_embed.repeat((1, self.r_x_to_c, 1))
        tok_x_embed = self.embed_drop(self.tok_embed(x) + pos_x_embed + self.pos_embed[:, self.cond_seq_length:])
        tok_c_embed = self.tok_embed(c) + pos_c_embed + self.pos_embed[:, :self.cond_seq_length]
        tok_cx_embed = torch.cat([tok_c_embed, tok_x_embed], dim=1)

        _y = self.blocks(tok_cx_embed, attn_mask=self.attn_mask)
        _y = self.layer_norm(_y)
        logits = self.head(_y)

        y = logits[:, self.cond_seq_length:, :]
        # cross entropy with smoothed label
        loss = self.cross_entropy_label_smoothing(rearrange(y, 'b l v -> (b l) v'), rearrange(x, 'b l -> (b l)'))

        return y, loss

    def cross_entropy_label_smoothing(self, x, target):
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = (1.0 - self.label_smoothing) * nll_loss + self.label_smoothing * smooth_loss
        return loss.mean()
