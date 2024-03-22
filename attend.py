import math
from enum import Enum
from functools import partial, wraps

import torch
import torch.nn.functional as F
from utils import (
    LayerNorm,
    create_causal_mask,
    default,
    dropout_seq,
    exists,
    l2norm,
    print_once,
)
from einops import rearrange, repeat
from packaging import version
from torch import einsum, nn


class Attend(nn.Module):
    def __init__(
        self,
        dropout=0.0,
        causal=False,
        scale=None,
        qk_norm=False,
        flash=False,
        causal_map_function=None,
        sdp_kwargs: dict = dict(
            enable_flash=True, enable_math=True, enable_mem_efficient=True
        ),
    ):
        super().__init__()
        self.scale = scale
        self.qk_norm = qk_norm

        self.causal = causal
        self.create_causal_mask = (
            causal_map_function
            if causal_map_function is not None
            else create_causal_mask
        )

        self.attn_fn = (
            partial(F.softmax, dtype=torch.float32) if not qk_norm else F.softmax
        )

        self.dropout = dropout
        self.attn_dropout = nn.Dropout(dropout)

        self.flash = flash
        assert not (
            flash and version.parse(torch.__version__) < version.parse("2.0.0")
        ), "in order to use flash attention, you must be using pytorch 2.0 or above"

        device_properties = torch.cuda.get_device_properties(torch.device("cuda"))

        if device_properties.major == 8 and device_properties.minor == 0:
            print_once(
                "A100 GPU detected, using flash attention if input tensor is on cuda"
            )
            sdp_kwargs = dict(
                enable_flash=True, enable_math=False, enable_mem_efficient=False
            )
        else:
            print_once(
                "Non-A100 GPU detected, using math or mem efficient attention if input tensor is on cuda"
            )
            sdp_kwargs = dict(
                enable_flash=False, enable_math=True, enable_mem_efficient=True
            )

        self.sdp_kwargs = sdp_kwargs

    def flash_attn(
        self,
        q,
        k,
        v,
        mask=None,
        attn_bias=None,
        is_causal=False,  ## mask already has causal mask or not
    ):
        batch, heads, q_len, _, k_len, is_cuda, device = (
            *q.shape,
            k.shape[-2],
            q.is_cuda,
            q.device,
        )

        # Recommended for multi-query single-key-value attention by Tri Dao
        # kv shape torch.Size([1, 512, 64]) -> torch.Size([1, 8, 512, 64])

        if k.ndim == 3:
            k = repeat(k, "b ... -> b h ...", h=q.shape[1])

        if v.ndim == 3:
            v = repeat(v, "b ... -> b h ...", h=q.shape[1])

        # handle scale - by default they scale by dim_head ** -0.5, but need to take care if using cosine sim attention

        if self.qk_norm:
            default_scale = q.shape[-1] ** -0.5
            q = q * (self.scale / default_scale)

        # Check if mask exists and expand to compatible shape
        # The mask is B L, so it would have to be expanded to B H N L

        causal = self.causal

        # in the case of kv caching with one token (q_len == 1), just turn off causal masking
        # in speculative decoding, this may go up to 5-6, so right aligned causal mask will be needed there

        if q_len == 1 and causal:
            causal = False

        # expand key padding mask

        if exists(mask):
            assert mask.ndim == 4
            mask = mask.expand(batch, heads, q_len, k_len).clone()

        # handle kv cache - this should be bypassable in updated flash attention 2

        if k_len > q_len and causal and not is_causal:
            causal_mask = self.create_causal_mask(q_len, k_len, device=device)
            if not exists(mask):
                mask = ~causal_mask
            else:
                mask = mask & ~causal_mask
            causal = False

        # manually handle causal mask, if another mask was given

        row_is_entirely_masked = None

        if exists(mask) and causal:
            if not is_causal:

                causal_mask = self.create_causal_mask(q_len, k_len, device=device)
                # print(causal_mask.shape)
                mask = mask & ~causal_mask

            # protect against an entire row being masked out

            row_is_entirely_masked = ~mask.any(dim=-1)
            mask[..., 0] = mask[..., 0] | row_is_entirely_masked

            causal = False

        # handle alibi positional bias
        # convert from bool to float

        if exists(attn_bias):
            attn_bias = rearrange(attn_bias, "h i j -> 1 h i j").expand(
                batch, heads, -1, -1
            )

            # if mask given, the mask would already contain the causal mask from above logic
            # otherwise, if no mask given but still causal, mask out alibi positional bias to a large negative number

            mask_value = -torch.finfo(q.dtype).max

            if exists(mask):
                attn_bias = attn_bias.masked_fill(~mask, mask_value // 2)
            elif causal:
                causal_mask = self.create_causal_mask(q_len, k_len, device=device)
                attn_bias = attn_bias.masked_fill(causal_mask, mask_value // 2)
                causal = False

            # scaled_dot_product_attention handles attn_mask either as bool or additive bias
            # make it an additive bias here

            mask = attn_bias

        # pytorch 2.0 flash attn: q, k, v, mask, dropout, causal, softmax_scale

        with torch.backends.cuda.sdp_kernel(**self.sdp_kwargs):
            out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=mask,  ##value of True indicates that the element should take part in attention.
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=causal,
            )

        # for a row that is entirely masked out, should zero out the output of that row token

        if exists(row_is_entirely_masked):
            out = out.masked_fill(row_is_entirely_masked[..., None], 0.0)

        return out

    def forward(
        self,
        q,
        k,
        v,
        mask=None,
        attn_bias=None,
        is_causal=False,  ## mask already has causal mask or not
    ):
        """
        einstein notation
        b - batch
        h - heads
        n, i, j - sequence length (base sequence length, source, target)
        d - feature dimension

        mask: True for places to attend to, False for padding
        """

        n, heads, kv_heads, device = q.shape[-2], q.shape[1], k.shape[1], q.device

        scale = default(self.scale, q.shape[-1] ** -0.5)

        # handle grouped multi-query attention

        if kv_heads == 1:
            k, v = map(lambda t: rearrange(t, "b 1 n d -> b n d"), (k, v))
        elif kv_heads < heads:
            k, v = map(
                lambda t: repeat(t, "b kvh n d -> b (r kvh) n d", r=heads // kv_heads),
                (k, v),
            )

        if self.flash:

            return self.flash_attn(
                q, k, v, mask=mask, attn_bias=attn_bias, is_causal=is_causal
            )

        kv_einsum_eq = "b j d" if k.ndim == 3 else "b h j d"

        dots = einsum(f"b h i d, {kv_einsum_eq} -> b h i j", q, k) * scale

        # qk_similarities = dots.clone()

        if exists(attn_bias):
            dots = dots + attn_bias

        i, j, dtype = *dots.shape[-2:], dots.dtype

        mask_value = -torch.finfo(dots.dtype).max

        if exists(mask):

            dots = dots.masked_fill(~mask, mask_value)

        if self.causal and not is_causal:

            causal_mask = self.create_causal_mask(i, j, device=device)  ##padding True
            dots = dots.masked_fill(causal_mask, mask_value)

        # pre_softmax_attn = dots.clone()

        attn = self.attn_fn(dots, dim=-1)
        attn = attn.type(dtype)

        # post_softmax_attn = attn.clone()

        attn = self.attn_dropout(attn)

        out = einsum(f"b h i j, {kv_einsum_eq} -> b h i d", attn, v)

        return out


class CustomMHA(nn.Module):
    def __init__(
        self,
        dim: int = 768,
        heads: int = 8,
        causal: bool = False,
        qk_norm: bool = False,
        qk_norm_scale: int = 8,
        dropout: float = 0.0,
        add_null_kv: bool = False,
        flash: bool = False,
        bias_att: bool = False,
        causal_map_function=None,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.causal = causal
        self.qk_norm = qk_norm
        self.flash = flash
        self.qk_norm_scale = qk_norm_scale
        self.add_null_kv = add_null_kv
        self.dropout = dropout

        self.dim_head = self.dim // self.heads
        self.scale = self.dim_head**-0.5

        inner_dim = self.dim_head * self.heads

        self.norm = LayerNorm(self.dim)

        self.attend = Attend(
            dropout=self.dropout,
            scale=self.qk_norm_scale if self.qk_norm else self.scale,
            causal=self.causal,
            qk_norm=self.qk_norm,
            flash=self.flash,
            causal_map_function=causal_map_function,
        )

        self.null_kv = nn.Parameter(torch.randn(2, self.heads, 1, self.dim_head))

        self.to_q = nn.Linear(self.dim, inner_dim, bias=bias_att)
        self.to_kv = nn.Linear(self.dim, inner_dim * 2, bias=bias_att)

        self.q_scale = nn.Parameter(torch.ones(self.dim_head))
        self.k_scale = nn.Parameter(torch.ones(self.dim_head))

        self.to_out = nn.Linear(inner_dim, self.dim, bias=bias_att)

    def forward(self, q, k, v, key_padding_mask=None, attn_mask=None, rel_pos=None):
        """
        q,k,v: b n d

        make sure to do the norm first/last stuff before entering this function

        key_padding_mask: False for padding
        attn_mask: False for padding

        """

        n = q.shape[-2]
        h = self.heads

        assert k is v, "only same key value suppoted"

        is_causal = True if exists(attn_mask) else False

        mask = None

        if exists(key_padding_mask):
            mask = rearrange(key_padding_mask, "b j -> b 1 1 j")

        if exists(attn_mask):
            attn_mask = repeat(attn_mask, "i j -> b 1 i j", b=q.shape[0])
            attn_mask = attn_mask.to(torch.bool)

            if exists(mask):
                mask = mask & attn_mask
            else:
                mask = attn_mask

        q, k, v = (self.to_q(q), *self.to_kv(k).chunk(2, dim=-1))

        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), (q, k, v))

        if self.add_null_kv:
            nk, nv = self.null_kv
            nk, nv = map(
                lambda t: repeat(t, "h 1 d -> b h 1 d", b=q.shape[0]), (nk, nv)
            )

            k = torch.cat((nk, k), dim=-2)
            v = torch.cat((nv, v), dim=-2)

            if exists(mask):

                mask = repeat(mask, "b 1 1 j -> b h i j", h=h, i=n)
                mask = F.pad(mask, (1, 0), value=True)

        if self.qk_norm:
            q, k = map(l2norm, (q, k))
            q = q * self.q_scale
            k = k * self.k_scale

        i, j = map(lambda t: t.shape[-2], (q, k))
        attn_bias = None
        if exists(rel_pos):
            attn_bias = rel_pos(i, j)

        out = self.attend(q, k, v, mask=mask, attn_bias=attn_bias, is_causal=is_causal)

        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)
