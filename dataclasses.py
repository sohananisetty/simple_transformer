from dataclasses import dataclass
import torch
from typing import List, Dict
from functools import partial
from enum import Enum


class PositionalEmbeddingType(Enum):
    REL = "core.models.positional_embeddings.RelativePositionBias"
    SINE = "core.models.positional_embeddings.ScaledSinusoidalEmbedding"
    ALIBI = "core.models.positional_embeddings.AlibiPositionalBias"
    ABS = "core.models.positional_embeddings.AbsolutePositionalEmbedding"
    SHAW = "core.models.positional_embeddings.ShawRelativePositionalEmbedding"


@dataclass
class AttentionParams:
    dim: int = 768
    heads: int = 8
    causal: bool = False
    qk_norm: bool = False
    qk_norm_scale: int = 8
    dropout: float = 0.0
    cross_attn_tokens_dropout: float = 0.0
    add_null_kv: bool = False
    flash: bool = False


@dataclass
class PositionalEmbeddingParams:
    dim: int = 768
    max_seq_len: int = 512
    causal: bool = False
    num_buckets: int = 32
    scale: float = 10.0
    heads: int = 8
    total_heads: int = 8
    theta = 10000
    max_distance = 128


@dataclass
class TransformerParams:
    attention_params: AttentionParams
    positional_embedding_params: PositionalEmbeddingParams = None
    positional_embedding: PositionalEmbeddingType = None

    num_tokens: int = 1024
    dim_out: int = None
    depth: int = 12
    ff_mult: int = 4
    self_cond: bool = False
    add_mask_id: bool = True
    emb_dropout = 0.0
    post_emb_norm = False
    context_dim: int = 768
    style_dim: int = 768
