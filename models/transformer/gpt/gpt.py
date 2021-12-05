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
from torch import nn

from models.transformer.gpt.layer import CausalSelfAttentionBlock


@attr.s(repr=False, eq=False)
class GPT(nn.Module):
    vocab_size: int = attr.ib()
    num_layers: int = attr.ib(default=12)
    num_heads: int = attr.ib(default=8)
    embed_dim: int = attr.ib(default=256)
    mlp_dropout_prob: float = attr.ib(default=0.0, validator=lambda i, a, x: 0.0 <= x < 1.0)
    attn_dropout_prob: float = attr.ib(default=0.0, validator=lambda i, a, x: 0.0 <= x < 1.0)
    pretrained_visual_embed_path: str = attr.ib(default=None)

    def __attrs_post_init__(self):
        super().__init__()
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

    def forward(self, x, attn_mask):
        """
        :param x: tokens, shape [b, l, embed_size]
        :return:
        """
        for layer in self.blocks:
            x = layer(x, attn_mask=attn_mask)
        logits = self.head(self.layer_norm(x))
        return logits
