# hubertvc/models/convnext_decoder.py
"""
Sub-500k Parameter ConvNeXt-1D Mel Decoder with Multi-Scale FiLM Conditioning.

Replaces the under-parameterized MobileNet decoder with modern 1D ConvNeXt blocks.
Directly synthesizes target log-mel spectrogram from fused content and speaker embeddings.
No additive deltas, no split projections, no G2 gating.
"""

from __future__ import annotations
from typing import Optional, Tuple, List, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from vc.config import ModelConfig, AudioConfig


class ConvNeXtBlock1D(nn.Module):
    """
    ConvNeXt 1D Block with FiLM (Feature-wise Linear Modulation) conditioning.
    
    Structure:
      1. Depthwise Conv1d (k=7, padding=3)
      2. LayerNorm (channel-last)
      3. Pointwise Conv1d (C -> 2C)
      4. GELU
      5. Pointwise Conv1d (2C -> C)
      6. FiLM modulation from speaker embedding: y = x * (1 + tanh(gamma)) + beta
      7. Residual connection
    """
    def __init__(self, dim: int, speaker_dim: int, expansion: int = 2, kernel_size: int = 7):
        super().__init__()
        self.dim = dim
        hidden_dim = dim * expansion

        # 1. Large-kernel depthwise conv
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=dim)
        
        # 2. LayerNorm
        self.norm = nn.LayerNorm(dim, eps=1e-6)

        # 3. Pointwise convs
        self.pwconv1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(hidden_dim, dim)

        # 4. FiLM generator: predicts scale (gamma) and shift (beta)
        self.film_proj = nn.Linear(speaker_dim, dim * 2)
        nn.init.xavier_uniform_(self.film_proj.weight, gain=0.2)
        nn.init.zeros_(self.film_proj.bias)

        # Residual scale (starts at 0.5 for balanced gradient flow)
        self.gamma_scale = nn.Parameter(0.5 * torch.ones(dim), requires_grad=True)

    def forward(self, x: torch.Tensor, speaker_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, T]
            speaker_emb: [B, D_spk]
        Returns:
            [B, C, T]
        """
        residual = x

        # Depthwise conv: [B, C, T]
        out = self.dwconv(x)

        # Permute to [B, T, C] for LayerNorm and Linear
        out = out.permute(0, 2, 1)
        out = self.norm(out)
        out = self.pwconv1(out)
        out = self.act(out)
        out = self.pwconv2(out)

        # FiLM conditioning from speaker embedding
        # film_proj: [B, D_spk] -> [B, 2*C]
        film_stats = self.film_proj(speaker_emb).unsqueeze(1)  # [B, 1, 2*C]
        gamma, beta = film_stats.chunk(2, dim=-1)              # [B, 1, C] each

        # Modulate: bounded scale with tanh, shift with beta
        out = out * (1.0 + torch.tanh(gamma)) + beta

        # Apply residual scale and add residual
        out = self.gamma_scale * out
        out = out.permute(0, 2, 1)  # [B, C, T]
        return residual + out


class TemporalResampler(nn.Module):
    _warned_no_source_lengths = False

    """
    Smoothly resamples temporal resolution from HuBERT semantic rate (e.g. 50 Hz / 20ms)
    to target Mel frame rate (e.g. 62.5 Hz / 16ms or 100 Hz / 10ms).
    """
    def __init__(self, channels: int = 96, default_ratio: float = 1.25):
        super().__init__()
        self.channels = channels
        self.default_ratio = default_ratio
        self.refine_conv = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        nn.init.xavier_uniform_(self.refine_conv.weight, gain=0.01)
        nn.init.zeros_(self.refine_conv.bias)

    def forward(self, x: torch.Tensor, target_length: Optional[int] = None,
                target_lengths: Optional[torch.Tensor] = None,
                source_lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [B, T, C]
            target_length: Optional exact frame length (shared across batch)
            target_lengths: Optional per-sample frame lengths [B]. When provided,
                each sample is resampled to its own true length and the batch is
                zero-padded to the max. This prevents the batch-max length bug
                where short utterances are stretched to the longest item.
            source_lengths: Optional per-sample TRUE content lengths [B]. When
                provided, each sample is cropped to its real content frames
                BEFORE interpolation, so the zero-padded batch tail is never
                time-warped into the output (source-side batch-max warp fix).
        Returns:
            [B, T_out, C]
        """
        if target_lengths is not None:
            B, T, C = x.shape
            outs = []
            for i in range(B):
                li = max(1, int(target_lengths[i].item()))
                if source_lengths is not None:
                    si = int(source_lengths[i].item())
                    si = max(1, min(si, T))
                else:
                    if not TemporalResampler._warned_no_source_lengths:
                        print("[TemporalResampler][WARN] target_lengths given but "
                              "source_lengths is None -> cropping to the padded "
                              "batch length (the Bug-B warp is back). Pass "
                              "content_lengths to forward().")
                        TemporalResampler._warned_no_source_lengths = True
                    si = T
                xi = x[i, :si].transpose(0, 1).unsqueeze(0)  # [1, C, si]
                xi = F.interpolate(xi, size=li, mode='linear', align_corners=False)
                xi = xi + 0.1 * self.refine_conv(xi)
                outs.append(xi.transpose(1, 2))  # [1, li, C]
            maxlen = max(o.shape[1] for o in outs)
            out = x.new_zeros(B, maxlen, C)
            for i, o in enumerate(outs):
                out[i, :o.shape[1]] = o[0]
            return out

        B, T, C = x.shape
        out_length = target_length if target_length is not None else round(T * self.default_ratio)

        # Permute to [B, C, T] for interpolation
        x = x.transpose(1, 2)
        x = F.interpolate(x, size=out_length, mode='linear', align_corners=False)
        x = x + 0.1 * self.refine_conv(x)
        return x.transpose(1, 2)  # [B, T_out, C]


class ConvNeXtDecoder(nn.Module):
    """
    Ultra-compact (~200k params) ConvNeXt-1D Mel Decoder.
    Directly synthesizes 80-bin log-mel spectrogram conditioned on speaker embedding.
    """
    def __init__(
        self,
        in_channels: int = 64,
        decoder_channels: int = 96,
        speaker_dim: int = 64,
        out_channels: int = 80,
        num_blocks: int = 4,
        expansion: int = 2,
        kernel_size: int = 7,
        speaker_token_routing: bool = False,
        num_speaker_tokens: int = 4,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.decoder_channels = decoder_channels
        self.speaker_dim = speaker_dim
        self.out_channels = out_channels
        self.speaker_token_routing = speaker_token_routing
        self.speaker_token_logits = None
        if speaker_token_routing:
            # Zero logits give uniform weights, exactly matching the existing
            # mean-pooling path at initialization. Fine-tuning can specialize
            # which of the four learned speaker tokens conditions each block.
            self.speaker_token_logits = nn.Parameter(
                torch.zeros(num_blocks, num_speaker_tokens))

        # Input adapter: projects fused semantic features to decoder channel dimension
        self.adapter = nn.Conv1d(in_channels, decoder_channels, kernel_size=1)
        nn.init.xavier_uniform_(self.adapter.weight)
        nn.init.zeros_(self.adapter.bias)

        # ConvNeXt blocks with FiLM modulation
        self.blocks = nn.ModuleList([
            ConvNeXtBlock1D(
                dim=decoder_channels,
                speaker_dim=speaker_dim,
                expansion=expansion,
                kernel_size=kernel_size
            )
            for _ in range(num_blocks)
        ])

        # Final norm and mel projection
        self.final_norm = nn.LayerNorm(decoder_channels)
        self.mel_proj = nn.Conv1d(decoder_channels, out_channels, kernel_size=1)
        nn.init.xavier_normal_(self.mel_proj.weight, gain=1.0)
        nn.init.constant_(self.mel_proj.bias, -4.5)  # HiFi-GAN log-mel baseline bias

    def forward(
        self,
        fused_features: torch.Tensor,
        speaker_emb: torch.Tensor,
        return_intermediate: bool = False
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        """
        Args:
            fused_features: [B, T, in_channels] (e.g., [B, T, 64])
            speaker_emb:    [B, speaker_dim] or [B, K, speaker_dim]
            return_intermediate: whether to return intermediate block features
        Returns:
            pred_mel: [B, 80, T]
            intermediates: list of intermediate tensors (if return_intermediate=True)
        """
        block_speakers = None
        # Pool speaker embedding if multi-token: [B, K, D] -> [B, D]. The
        # optional router starts at uniform weights and learns block-specific
        # combinations without adding another projection network.
        if speaker_emb.dim() == 3:
            if self.speaker_token_logits is None:
                speaker_pooled = speaker_emb.mean(dim=1)
            else:
                if speaker_emb.shape[1] != self.speaker_token_logits.shape[1]:
                    raise ValueError(
                        f"speaker token count {speaker_emb.shape[1]} does not match "
                        f"router size {self.speaker_token_logits.shape[1]}"
                    )
                routing = self.speaker_token_logits.softmax(dim=-1)
                block_speakers = torch.einsum("nk,bkd->nbd", routing, speaker_emb)
                speaker_pooled = None
        else:
            speaker_pooled = speaker_emb

        # Transpose to [B, C, T]
        x = fused_features.transpose(1, 2)
        x = self.adapter(x)

        intermediates = [] if return_intermediate else None

        for block_idx, block in enumerate(self.blocks):
            speaker_for_block = (block_speakers[block_idx]
                                 if block_speakers is not None else speaker_pooled)
            x = block(x, speaker_for_block)
            if return_intermediate:
                intermediates.append(x)

        # Final norm and mel projection
        x_norm = self.final_norm(x.transpose(1, 2)).transpose(1, 2)
        pred_mel = self.mel_proj(x_norm)  # [B, 80, T]

        if return_intermediate:
            return pred_mel, intermediates
        return pred_mel
