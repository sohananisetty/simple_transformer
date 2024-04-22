import torch
import torch.nn.functional as F
from attend import CustomMHA
from .params import (
    TransformerParams,
    PositionalEmbeddingParams,
    PositionalEmbeddingType,
)
from utils import default, exists, get_obj_from_str, prob_mask_like
from einops import rearrange, repeat
from torch import einsum, nn
from tqdm.auto import tqdm
from .positional_embeddings import AlibiPositionalBias, ScaledSinusoidalEmbedding
from typing import Optional


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer("beta", torch.zeros(dim))

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)


def FeedForward(
    dim,
    mult=4,
    dropout=0.1,
):
    """https://arxiv.org/abs/2110.09456"""

    inner_dim = int(dim * mult)
    return nn.Sequential(
        nn.Linear(dim, inner_dim, bias=False),
        nn.GELU(),
        LayerNorm(inner_dim),
        nn.Dropout(dropout),
        nn.Linear(inner_dim, dim, bias=False),
    )


class TransformerBlockCustom(nn.Module):
    def __init__(
        self,
        dim: int = 768,
        heads: int = 8,
        attn_dropout: float = 0.0,
        depth: int = 1,
        ff_mult: int = 4,
    ):
        super().__init__()
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        CustomMHA(
                            dim=dim,
                            heads=heads,
                            dropout=attn_dropout,
                            causal=False,
                            flash=True,
                        ),
                        CustomMHA(
                            dim=dim,
                            heads=heads,
                            dropout=attn_dropout,
                            flash=True,
                            causal=False,
                        ),
                        FeedForward(dim=dim, mult=ff_mult, dropout=attn_dropout),
                    ]
                )
            )

        self.norm_sa = LayerNorm(dim)
        self.norm_cross = LayerNorm(dim)
        self.norm_out = LayerNorm(dim)

    def forward(self, x, mask=None, context=None, context_mask=None, rel_pos=None):
        for attn, cross_attn, ff1 in self.layers:

            x = self.norm_sa(x)
            x = (
                attn(
                    q=x,
                    k=x,
                    v=x,
                    key_padding_mask=mask,
                    rel_pos=rel_pos,
                )
                + x
            )

            x = (
                cross_attn(
                    q=self.norm_cross(x),
                    k=context,
                    v=context,
                    key_padding_mask=context_mask,
                )
                + x
            )

            x = self.norm_out(ff1(x) + x)

        return x


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int = 768,
        heads: int = 8,
        attn_dropout: float = 0.0,
        depth: int = 1,
        ff_mult: int = 4,
    ):
        super().__init__()
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        nn.MultiheadAttention(
                            embed_dim=dim,
                            num_heads=heads,
                            dropout=attn_dropout,
                            bias=False,
                            batch_first=True,
                        ),
                        nn.MultiheadAttention(
                            embed_dim=dim,
                            num_heads=heads,
                            dropout=attn_dropout,
                            bias=False,
                            batch_first=True,
                        ),
                        FeedForward(dim=dim, mult=ff_mult, dropout=attn_dropout),
                    ]
                )
            )

        self.norm_sa = nn.LayerNorm(dim, eps=1e-5, bias=False)
        self.norm_cross = nn.LayerNorm(dim, eps=1e-5, bias=False)
        self.norm_out = nn.LayerNorm(dim, eps=1e-5, bias=False)

    def forward(self, x, mask=None, context=None, context_mask=None, rel_pos=None):
        for attn, cross_attn, ff1 in self.layers:

            x = self.norm_sa(x)

            x = (
                attn(
                    query=x,
                    key=x,
                    value=x,
                    key_padding_mask=mask,
                    need_weights=False,
                )[0]
                + x
            )

            x = (
                cross_attn(
                    query=self.norm_cross(x),
                    key=context,
                    value=context,
                    key_padding_mask=context_mask,
                    need_weights=False,
                )[0]
                + x
            )

            x = self.norm_out(ff1(x) + x)

        return x


class Transformer(nn.Module):
    def __init__(
        self,
        dim=256,
        heads=8,
        num_tokens=512,
        max_seq_len=128,
        add_mask_id=True,
        emb_dropout=0.0,
        attn_dropout=0.0,
        dim_out: Optional[int] = None,
        depth: int = 12,
        ff_mult: int = 4,
        post_emb_norm=False,
        context_dim: int = 768,
        rel_pos=False,
    ):
        super().__init__()
        self.dim = dim
        self.num_tokens = num_tokens
        self.seq_len = max_seq_len

        self.mask_id = self.num_tokens if add_mask_id else None

        self.token_emb = nn.Embedding(self.num_tokens + int(add_mask_id), self.dim)

        if rel_pos:
            self.pos_emb = AlibiPositionalBias(heads=heads // 2, total_heads=heads)
            self.is_abs_pos_emb = False

        else:
            self.pos_emb = ScaledSinusoidalEmbedding(dim=self.dim)

            self.is_abs_pos_emb = True

        self.project_condition = (
            nn.Linear(context_dim, self.dim, bias=False)
            if context_dim != self.dim
            else nn.Identity()
        )

        self.emb_dropout = nn.Dropout(emb_dropout)

        self.transformer_blocks = TransformerBlock(
            dim=self.dim,
            heads=heads,
            attn_dropout=attn_dropout,
            depth=depth,
            ff_mult=ff_mult,
        )

        self.dim_out = default(dim_out, self.num_tokens)
        self.to_logits = nn.Linear(self.dim, self.dim_out, bias=False)

        self.post_emb_norm = nn.LayerNorm(self.dim) if post_emb_norm else nn.Identity()

    def prepare_inputs(
        self, x, mask=None, context_embed=None, context_mask=None, cond_drop_prob=0.0
    ):

        device, b, n = x.device, *x.shape

        if mask is None:
            mask = x != self.mask_id

        # context = self.context_embed_projs[context_type](context_embeds)
        if context_embed is not None and context_mask is None:
            context_embed = self.project_condition(context_embed)
            context_mask = context_embed != self.mask_id

        # classifier free guidance

        if cond_drop_prob > 0.0:
            mask_ = prob_mask_like((b, 1), 1.0 - cond_drop_prob, device)
            context_mask = context_mask & mask_

        return (
            mask,
            context_embed,
            context_mask,
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None,
        return_embed: bool = False,
        return_logits: bool = False,
        labels=None,
        ignore_index: int = -100,
        cond_drop_prob: float = 0.0,
        context_embed=None,
        context_mask=None,
    ):
        device, b, n = x.device, *x.shape
        assert n <= self.seq_len

        mask, context, context_mask = self.prepare_inputs(
            x,
            mask=mask,
            context_embed=context_embed,
            context_mask=context_mask,
            cond_drop_prob=cond_drop_prob,
        )

        # embed tokens
        if self.is_abs_pos_emb:
            x = self.token_emb(x) + self.pos_emb(x)
            rel_pos = None
        else:
            x = self.token_emb(x)
            rel_pos = self.pos_emb

        # post embedding norm, purportedly leads to greater stabilization
        x = self.post_emb_norm(x)

        embed = self.transformer_blocks(
            x,
            mask=mask,
            context=context,
            context_mask=context_mask,
            rel_pos=rel_pos,
        )

        if return_embed:
            return embed

        logits = self.to_logits(embed)

        if not exists(labels):
            return logits

        if self.dim_out == 1:
            loss = F.binary_cross_entropy_with_logits(
                rearrange(logits, "... 1 -> ..."), labels
            )
        else:
            loss = F.cross_entropy(
                rearrange(logits, "b n c -> b c n"), labels, ignore_index=ignore_index
            )

        if not return_logits:
            return loss

        return loss, logits
