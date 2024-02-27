import torch
import torch.nn.functional as F
from attend import Attention
from .params import (
    AttentionParams,
    TransformerParams,
    PositionalEmbeddingParams,
    PositionalEmbeddingType,
)
from utils import default, exists, get_obj_from_str, LayerNorm, prob_mask_like
from einops import rearrange, repeat
from torch import einsum, nn
from tqdm.auto import tqdm


class GEGLU(nn.Module):
    """https://arxiv.org/abs/2002.05202"""

    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return gate * F.gelu(x)


def FeedForward(dim, mult=4):
    """https://arxiv.org/abs/2110.09456"""

    inner_dim = int(dim * mult * 2 / 3)
    return nn.Sequential(
        LayerNorm(dim),
        nn.Linear(dim, inner_dim * 2, bias=False),
        GEGLU(),
        LayerNorm(inner_dim),
        nn.Linear(inner_dim, dim, bias=False),
    )


class TransformerBlock(nn.Module):
    def __init__(
        self,
        self_attention_params: AttentionParams,
        cross_attention_params: AttentionParams = None,
        depth: int = 1,
        ff_mult: int = 4,
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.do_cross = cross_attention_params is not None

        assert cross_attention_params.dim == self_attention_params.dim, "need same dim"

        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        Attention(self_attention_params),
                        (
                            Attention(cross_attention_params)
                            if self.do_cross
                            else nn.Identity()
                        ),
                        FeedForward(dim=self_attention_params.dim, mult=ff_mult),
                    ]
                )
            )

        self.norm = LayerNorm(self_attention_params.dim)

    def forward(self, x, mask=None, context=None, context_mask=None, rel_pos=None):
        for attn, cross_attn, ff1 in self.layers:
            x = attn(x, mask=mask, rel_pos=rel_pos) + x

            if self.do_cross:

                x = (
                    cross_attn(x, mask=mask, context=context, context_mask=context_mask)
                    + x
                )

            x = ff1(x) + x

        return self.norm(x)


class Transformer(nn.Module):
    def __init__(self, transformer_params: TransformerParams):
        super().__init__()
        self.dim = transformer_params.self_attention_params.dim
        self.num_tokens = transformer_params.num_tokens
        self.seq_len = transformer_params.positional_embedding_params.max_seq_len

        self.mask_id = self.num_tokens if transformer_params.add_mask_id else None

        self.token_emb = nn.Embedding(
            self.num_tokens + int(transformer_params.add_mask_id), self.dim
        )

        if transformer_params.positional_embedding_params is None:
            transformer_params.positional_embedding_params = PositionalEmbeddingParams(
                dim=self.dim
            )
            transformer_params.positional_embedding = PositionalEmbeddingType.SINE

        self.is_abs_pos_emb = transformer_params.positional_embedding.name in [
            "ABS",
            "SINE",
        ]

        self.pos_emb = get_obj_from_str(transformer_params.positional_embedding.value)(
            transformer_params.positional_embedding_params
        )

        self.emb_dropout = nn.Dropout(transformer_params.emb_dropout)

        self.transformer_blocks = TransformerBlock(
            self_attention_params=transformer_params.self_attention_params,
            cross_attention_params=transformer_params.cross_attention_params,
            depth=transformer_params.depth,
            ff_mult=transformer_params.ff_mult,
        )
        self.norm = LayerNorm(self.dim)

        self.dim_out = default(transformer_params.dim_out, self.num_tokens)
        self.to_logits = nn.Linear(self.dim, self.dim_out, bias=False)

        self.post_emb_norm = (
            nn.LayerNorm(self.dim)
            if transformer_params.post_emb_norm
            else nn.Identity()
        )

    def prepare_inputs(
        self, x, mask=None, context_embed=None, context_mask=None, cond_drop_prob=0.0
    ):
        device, b, n = x.device, *x.shape

        if mask is None:
            mask = x != self.mask_id

        # context = self.context_embed_projs[context_type](context_embeds)
        if context_embed is not None and context_mask is None:
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

    def forward_with_cond_scale(
        self, *args, cond_scale=3.0, return_embed=False, **kwargs
    ):
        if cond_scale == 1:
            return self.forward(
                *args, return_embed=return_embed, cond_drop_prob=0.0, **kwargs
            )

        logits, embed = self.forward(
            *args, return_embed=True, cond_drop_prob=0.0, **kwargs
        )

        null_logits = self.forward(*args, cond_drop_prob=1.0, **kwargs)

        scaled_logits = null_logits + (logits - null_logits) * cond_scale

        if return_embed:
            return scaled_logits, embed

        return scaled_logits

    def forward_with_neg_prompt(
        self,
        *args,
        text_embed: torch.Tensor,
        neg_text_embed: torch.Tensor,
        cond_scale=3.0,
        return_embed=False,
        **kwargs,
    ):
        neg_logits = self.forward(
            *args, neg_text_embed=neg_text_embed, cond_drop_prob=0.0, **kwargs
        )
        pos_logits, embed = self.forward(
            *args,
            return_embed=True,
            text_embed=text_embed,
            cond_drop_prob=0.0,
            **kwargs,
        )

        logits = neg_logits + (pos_logits - neg_logits) * cond_scale

        if return_embed:
            return logits, embed

        return logits

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
