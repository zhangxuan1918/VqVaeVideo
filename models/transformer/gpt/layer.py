import attr
from torch import nn
from einops import rearrange


@attr.s(repr=False, eq=False)
class CausalSelfAttention(nn.Module):
    embed_dim: int = attr.ib()
    num_heads: int = attr.ib()
    dropout: float = attr.ib(default=0.0, validator=lambda i, a, x: 0.0 <= x < 1.0)
    bias: bool = attr.ib(default=True)
    add_bias_kv: bool = attr.ib(default=False)
    add_zero_attn: bool = attr.ib(default=False)
    kdim: int = attr.ib(default=None)
    vdim: int = attr.ib(default=None)

    def __attrs_post_init__(self):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            dropout=self.dropout,
            bias=self.bias,
            add_bias_kv=self.add_bias_kv,
            add_zero_attn=self.add_zero_attn,
            kdim=self.kdim,
            vdim=self.vdim
        )
        self.key = nn.Linear(self.embed_dim, self.embed_dim)
        self.query = nn.Linear(self.embed_dim, self.embed_dim)
        self.value = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(self, x, key_padding_mask=None, need_weights=False, attn_mask=None):
        # compute query, key and values
        k = rearrange(self.key(x), 'b l e -> l b e')
        q = rearrange(self.query(x), 'b l e -> l b e')
        v = rearrange(self.value(x), 'b l e -> l b e')

        return self.attention(
            query=q,
            key=k,
            value=v,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            attn_mask=attn_mask)


@attr.s(repr=False, eq=False)
class CausalSelfAttentionBlock(nn.Module):
    embed_dim: int = attr.ib()
    num_heads: int = attr.ib()
    attn_dropout_prob: float = attr.ib(default=0.0, validator=lambda i, a, x: 0.0 <= x < 1.0)
    bias: bool = attr.ib(default=True)
    add_bias_kv: bool = attr.ib(default=False)
    add_zero_attn: bool = attr.ib(default=False)
    kdim: int = attr.ib(default=None)
    vdim: int = attr.ib(default=None)
    mlp_dropout_prob: float = attr.ib(default=0.0, validator=lambda i, a, x: 0.0 <= x < 1.0)

    def __attrs_post_init__(self):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(self.embed_dim)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim)
        self.attention = CausalSelfAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            dropout=self.attn_dropout_prob,
            bias=self.bias,
            add_bias_kv=self.add_bias_kv,
            add_zero_attn=self.add_zero_attn,
            kdim=self.kdim,
            vdim=self.vdim
        )
        self.mlp = nn.Sequential(
            nn.Linear(self.embed_dim, 4 * self.embed_dim),
            nn.GELU(),
            nn.Linear(4 * self.embed_dim, self.embed_dim),
            nn.Dropout(self.mlp_dropout_prob)
        )

    def forward(self, x, attn_mask=None, key_padding_mask=None, need_weights=False):
        y = self.layer_norm1(x)
        y, _ = self.attention(
            x=y,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            attn_mask=attn_mask
        )
        y = rearrange(y, 'l b e -> b l e')
        y = x + y
        y = y + self.mlp(self.layer_norm2(y))
        return y, attn_mask, key_padding_mask, need_weights