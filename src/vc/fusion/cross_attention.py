# hubertvc/models/cross_attention.py
"""
Clean, position-agnostic multi-head cross-attention mechanism for zero-shot voice conversion.
Fuses HuBERT semantic content features (queries) with ECAPA speaker tokens (keys/values).

Key Design Principles:
- No positional encodings on speaker tokens (order- and length-invariant timbre dictionary).
- Standard scaled dot-product multi-head attention.
- Clean residual connection with LayerNorm.
- No artificial detached bias injections, tied-weight hooks, or ad-hoc temperature hacks.
"""

from __future__ import annotations
import math
from typing import Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from vc.config import ModelConfig


class ReferenceFrameEncoder(nn.Module):
    """
    Lightweight 1D conv encoder over reference mel frames.

    Produces position-agnostic timbre tokens directly from the reference signal
    (no learned per-speaker embedding), so conditioning generalizes to unseen
    speakers. Temporal stride keeps the token count small for cheap attention.
    """
    def __init__(self, n_mels: int = 80, dim: int = 64, stride: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_mels, dim, kernel_size=5, stride=stride, padding=2),
            nn.GELU(),
            nn.Conv1d(dim, dim, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(dim, dim, kernel_size=5, padding=2),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        # mel: [B, n_mels, T] -> [B, T', dim]
        h = self.net(mel)
        return self.norm(h.transpose(1, 2))


class PositionAgnosticCrossAttention(nn.Module):
    """
    Position-Agnostic Multi-Head Cross-Attention Layer.
    
    Content features serve as Queries (preserving temporal phonetics).
    Speaker tokens serve as Keys and Values (providing acoustic timbre).
    """
    def __init__(
        self,
        content_dim: int = 64,
        speaker_dim: int = 64,
        d_model: int = 64,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.content_dim = content_dim
        self.speaker_dim = speaker_dim
        self.d_model = d_model
        self.num_heads = num_heads

        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")

        # Projections
        self.q_proj = nn.Linear(content_dim, d_model)
        self.k_proj = nn.Linear(speaker_dim, d_model)
        self.v_proj = nn.Linear(speaker_dim, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.mha = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # Subtle FiLM modulation based on pooled speaker embedding
        self.film = nn.Linear(speaker_dim, d_model * 2)
        nn.init.xavier_uniform_(self.film.weight, gain=0.2)
        nn.init.zeros_(self.film.bias)

        # Residual projection if content_dim != d_model
        if content_dim != d_model:
            self.res_proj = nn.Linear(content_dim, d_model)
        else:
            self.res_proj = nn.Identity()

    def forward(
        self,
        content_features: torch.Tensor,
        speaker_features: Optional[torch.Tensor] = None,
        ref_frames: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            content_features: [B, T, content_dim] (HuBERT phonetic frames)
            speaker_features: [B, K, speaker_dim] (ECAPA speaker tokens)
            ref_frames: optional [B, K2, speaker_dim] reference-frame tokens
                (position-agnostic). Concatenated with speaker tokens as K/V.
            return_attention: if True, returns (fused, attn_weights)
        Returns:
            fused_features: [B, T, d_model]
        """
        # Ensure 3D shapes
        if content_features.dim() == 2:
            content_features = content_features.unsqueeze(0)
        if speaker_features is not None and speaker_features.dim() == 2:
            speaker_features = speaker_features.unsqueeze(1)
        if ref_frames is not None and ref_frames.dim() == 2:
            ref_frames = ref_frames.unsqueeze(0)

        if speaker_features is None and ref_frames is None:
            raise ValueError("cross-attention needs speaker_features and/or ref_frames")

        # Keys/values: global speaker tokens and/or reference-frame tokens
        if speaker_features is None:
            kv = ref_frames
        elif ref_frames is None:
            kv = speaker_features
        else:
            kv = torch.cat([speaker_features, ref_frames], dim=1)

        residual = self.res_proj(content_features)

        # Q from content, K and V from speaker timbre tokens
        q = self.q_proj(content_features)
        k = self.k_proj(kv)
        v = self.v_proj(kv)

        # Multi-head cross-attention
        attended, attn_weights = self.mha(
            query=q,
            key=k,
            value=v,
            need_weights=return_attention,
            average_attn_weights=True
        )

        # Residual + LayerNorm
        out = self.norm(residual + self.dropout(self.out_proj(attended)))

        # Multi-scale FiLM injection (pooled over all timbre tokens)
        spk_pooled = kv.mean(dim=1)  # [B, speaker_dim]
        film_stats = self.film(spk_pooled).unsqueeze(1)  # [B, 1, 2*d_model]
        gamma, beta = film_stats.chunk(2, dim=-1)

        # Soft modulation
        out = out * (1.0 + torch.tanh(gamma)) + beta

        if return_attention:
            return out, attn_weights
        return out
