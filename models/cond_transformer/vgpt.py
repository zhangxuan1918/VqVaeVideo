import attr
import torch
from einops import rearrange
from torch import nn
from torch.nn import functional as F

from models.transformer.gpt import GPT


@attr.s(repr=False, eq=False)
class VGPT(nn.Module):
    vocab_size: int = attr.ib()
    max_seq_length: int = attr.ib()
    num_layers: int = attr.ib(default=12)
    num_heads: int = attr.ib(default=8)
    embed_dim: int = attr.ib(default=256)
    embed_dropout_prob: float = attr.ib(default=0.1, validator=lambda i, a, x: 0 <= x <= 1.0)
    mlp_dropout_prob: float = attr.ib(default=0.0, validator=lambda i, a, x: 0.0 <= x < 1.0)
    attn_dropout_prob: float = attr.ib(default=0.0, validator=lambda i, a, x: 0.0 <= x < 1.0)
    pretrained_visual_embed_path: str = attr.ib(default=None)
    label_smoothing: float = attr.ib(default=0.5, validator=lambda i, a, x: 0.0 <= x < 1.0)

    def __attrs_post_init__(self):
        super().__init__()
        self.gpt = GPT(
            vocab_size=self.vocab_size,
            max_seq_length=self.max_seq_length,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            embed_dim=self.embed_dim,
            embed_dropout_prob=self.embed_dropout_prob,
            mlp_dropout_prob=self.mlp_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            pretrained_visual_embed_path=self.pretrained_visual_embed_path
        )

        attn_mask = torch.triu(torch.ones(self.max_seq_length, self.max_seq_length)).transpose(0, 1)
        attn_mask = attn_mask.float().masked_fill(attn_mask == 0, float('-inf')).masked_fill(attn_mask == 1, float(0.0))
        self.register_buffer('attn_mask', attn_mask)

    def forward(self, x, comp_loss=True):
        """
        :param x: tokens, shape [b, l]
        :return:
        """
        # x = x[..., :-1]
        # we do not use the last token
        logits = self.gpt(x=x, attn_mask=self.attn_mask)
        y = logits[:, 1:]
        loss = None
        if comp_loss:
            # cross entropy with smoothed label
            # loss = self.cross_entropy_label_smoothing(rearrange(y, 'b l v -> (b l) v'), rearrange(x, 'b l -> (b l)'))
            loss = F.cross_entropy(rearrange(y, 'b l v -> (b l) v'), rearrange(x, 'b l -> (b l)'))
        return y, loss

    def cross_entropy_label_smoothing(self, x, target):
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = (1.0 - self.label_smoothing) * nll_loss + self.label_smoothing * smooth_loss
        return loss.mean()

    def top_k_logits(self, logits, k):
        v, ix = torch.topk(logits, k)
        out = logits.clone()
        out[out < v[..., [-1]]] = -float('Inf')
        return out

    @torch.no_grad()
    def sample(self, x, patch_size, n_cond_frames=1, topk=None, sample=False):
        """
        sample codes from transformer
        sample frame by frame
        :param x: sampled code, shape [b, f, 32, 32]
        :param n_cond_frames: number of frames fixed, the following frames will condition on them
        :param patch_size: patch size used, we split the full codes to smaller patch when passing
        through transformer
        :return: sampled tokens, [b, f, 32, 32]
        """
        b, f, h, w = x.size()
        patch_size_half = patch_size // 2
        for k in range(n_cond_frames, f):
            # sample (dx, dy) on frame df
            for dx in range(0, h):
                if dx <= patch_size_half:
                    i = dx
                elif h - dx < patch_size_half:
                    i = patch_size - (h - dx)
                else:
                    i = patch_size_half
                for dy in range(0, w):
                    if dy <= patch_size_half:
                        j = dy
                    elif w - dy < patch_size_half:
                        j = patch_size - (w - dy)
                    else:
                        j = patch_size_half

                    i_start = dx - i
                    i_end = i_start + patch_size
                    j_start = dy - j
                    j_end = j_start + patch_size

                    xx = rearrange(x[:, i_start:i_end, j_start:j_end], 'b f p q -> b (f p q)').to('cuda')
                    logits, _ = self.forward(x=xx, comp_loss=False)
                    logits = rearrange(logits, 'b (f p q) d -> b f p q d', f=f, p=patch_size, q=patch_size)
                    logits = logits[:, k, i, j]

                    if topk is not None:
                        logits = self.top_k_logits(logits, topk)
                    # apply softmax to convert to prob
                    probs = F.softmax(logits, dim=-1)
                    if sample:
                        ix = torch.multinomial(probs, num_samples=1)
                    else:
                        _, ix = torch.topk(probs, k=1, dim=-1)
                    x[:, k, dx, dy] = ix
        return x
