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

import attr
import torch
from einops import rearrange
from torch import nn
from torch.nn import functional as F
from models.transformer.gpt.layer import CausalSelfAttentionBlock


@attr.s(repr=False, eq=False)
class GPT(nn.Module):
    vocab_size: int = attr.ib()
    max_seq_length: int = attr.ib()
    num_layers: int = attr.ib(default=12)
    num_heads: int = attr.ib(default=8)
    embed_dim: int = attr.ib(default=256)
    embed_dropout_prob: float = attr.ib(default=0.0, validator=lambda i, a, x: 0.0 <= x < 1.0)
    mlp_dropout_prob: float = attr.ib(default=0.0, validator=lambda i, a, x: 0.0 <= x < 1.0)
    attn_dropout_prob: float = attr.ib(default=0.0, validator=lambda i, a, x: 0.0 <= x < 1.0)

    def __attrs_post_init__(self):
        super().__init__()

        # input embedding
        self.tok_embed = nn.Embedding(self.vocab_size, self.embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.max_seq_length, self.embed_dim))
        self.embed_drop = nn.Dropout(self.embed_dropout_prob)

        # attention mask
        attn_mask = torch.tril(torch.ones((self.max_seq_length, self.max_seq_length), dtype=torch.uint8)).to('cuda')
        self.register_buffer('attn_mask', attn_mask)

        # transformer
        trans_blks = [(f'group_{i}', CausalSelfAttentionBlock(
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                attn_dropout_prob=self.attn_dropout_prob,
                mlp_dropout_prob=self.mlp_dropout_prob))
              for i in range(self.num_layers)]
        self.blocks = nn.Sequential(OrderedDict(trans_blks))

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

    def forward(self, input, embeddings=None, targets=None, **kwargs):
        # TODO: in training, when passing input, we need to make sure the input have the same length
        #       if not, we can append black images
        # targets should be equal to input if embeddings is not null
        # input: bos, t0, t1, t2, ..., t510
        # output:     t0, t1, t2, ..., t510, t511
        # targets:    t0, t1, t2, ..., t510
        # we use causal transformer, thus, we need to provide a causal attn_mask
        # mask: torch.tril(torch.ones((max_seq_length, max_seq_length), dtype=torch.uint8)

        tok_embeddings = self.tok_embed(input)

        # prepend explicit embeddings
        if embeddings is not None:
            tok_embeddings = torch.cat([embeddings, tok_embeddings], dim=1)
        seq_length = input.size()[1]
        assert seq_length <= self.max_seq_length, f'input sequence too long, max seq length {self.max_seq_length}'
        pos_embedding = self.pos_embed[:, :seq_length, :]

        x = self.embed_drop(tok_embeddings + pos_embedding)
        x, _, _, _ = self.blocks(x, self.attn_mask, **kwargs)
        x = self.layer_norm(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            # the last token will be discarded
            loss = F.cross_entropy(rearrange(logits[:, :-1, :], 'b l v -> (b l) v'), rearrange(targets, 'b l -> (b l)'))

        return logits, loss