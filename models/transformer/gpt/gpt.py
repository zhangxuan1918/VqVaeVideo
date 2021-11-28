"""
taken from: https://github.com/karpathy/minGPT/
GPT model:
- the initial stem consists of a combination of token encoding and a positional encoding
- the meat of it is a uniform sequence of Transformer blocks
    - each Transformer is a sequential combination of a 1-hidden-layer MLP block and a self-attention block
    - all blocks feed into a central residual pathway similar to resnets
- the final decoder is a linear projection into a vanilla Softmax classifier
"""

import attr
import numpy as np
import torch
from torch import nn
from models.transformer.gpt.layer import CausalSelfAttentionBlock


@attr.s(repr=False, eq=False)
class GPT(nn.Module):
    vocab_size: int = attr.ib()
    max_seq_length: int = attr.ib()
    num_layers: int = attr.ib(default=12)
    num_heads: int = attr.ib(default=8)
    embed_dim: int = attr.ib(default=256)
    embed_dropout_prob: float = attr.ib(default=0.1, validator=lambda i, a, x: 0 <= x <= 1.0)
    mlp_dropout_prob: float = attr.ib(default=0.0, validator=lambda i, a, x: 0.0 <= x < 1.0)
    attn_dropout_prob: float = attr.ib(default=0.0, validator=lambda i, a, x: 0.0 <= x < 1.0)
    pretrained_visual_embed_path: str = attr.ib(default=None)

    def __attrs_post_init__(self):
        super().__init__()

        # set up embedding
        self._set_visual_embed()
        self.pos_embed = nn.Parameter(torch.zeros(1, self.max_seq_length, self.embed_dim))
        self.start_embed = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.embed_drop = nn.Dropout(self.embed_dropout_prob)
        # transformer
        trans_blks = [CausalSelfAttentionBlock(
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                attn_dropout_prob=self.attn_dropout_prob,
                mlp_dropout_prob=self.mlp_dropout_prob)
              for i in range(self.num_layers)]
        self.blocks = nn.ModuleList(trans_blks)

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

    def forward(self, x, attn_mask):
        """
        :param x: tokens, shape [b, l]
        :return:
        """
        b = x.size(0)
        tok_x_embed = self.tok_embed(x)
        # append start of sequence token in front
        tok_x_embed = torch.cat([self.start_embed.repeat((b, 1, 1)), tok_x_embed], dim=1)
        tok_x_embed = self.embed_drop(tok_x_embed + self.pos_embed)

        for layer in self.blocks:
            tok_x_embed = layer(tok_x_embed, attn_mask=attn_mask)
        logits = self.head(self.layer_norm(tok_x_embed))
        return logits
