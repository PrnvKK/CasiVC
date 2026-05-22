# hubertvc/models/mobilenet_decoder.py
"""
Light-weight MobileNet-V3-small decoder that turns fused content-and-speaker
features into a log-mel spectrogram.

All configuration comes from config.ModelConfig - nothing is hardcoded.
"""

from __future__ import annotations
from typing import List, Tuple, Dict
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import AudioConfig, ModelConfig


# --------------------------------------------------------------------------- #
#  Helpers                                                                    #
# --------------------------------------------------------------------------- #
class HardSwish(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * F.relu6(x + 3.) / 6.


class SqueezeExcite(nn.Module):
    """
    MobileNet-V3 style Squeeze-and-Excite (adapted for 1-D conv).
    """
    def __init__(self, channels: int, se_ratio: float = 0.25):
        super().__init__()
        hidden = max(8, int(channels * se_ratio))
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Conv1d(channels, hidden, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden, channels, kernel_size=1, bias=True),
            nn.Hardsigmoid(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.fc(self.avg_pool(x))
        return x * scale


def _make_norm(channels: int, norm: str = "gn", num_groups: int = 8) -> nn.Module:
    """Create normalization layer from config."""
    if norm is None or norm.lower() == "none":
        return nn.Identity()
    if norm.lower() == "bn":
        return nn.BatchNorm1d(channels)
    # default GN
    num_groups = min(num_groups, channels)
    return nn.GroupNorm(num_groups, channels)


# --------------------------------------------------------------------------- #
#  MobileNet block                                                            #
# --------------------------------------------------------------------------- #
class MobileNetBlock(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int,
        expand_ratio: int,
        use_se: bool,
        norm: str,
        upsample_first: bool = False,
        residual_scale: float = 1.0,
        residual_identity_scale: float = 1.0
    ):
        super().__init__()
        
        # Upsample BEFORE block processing if configured
        self.upsample_first = None
        if upsample_first:
            self.upsample_first = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv1d(in_ch, in_ch, kernel_size=3, padding=1, bias=True),
            )
            
        norm_cfg = {"norm": norm}
        act = HardSwish()

        hidden_ch = in_ch * expand_ratio
        layers: List[nn.Module] = []

        # 1. Point-wise expansion
        if expand_ratio != 1:
            layers += [
                nn.Conv1d(in_ch, hidden_ch, 1, bias=True),
                _make_norm(hidden_ch, **norm_cfg),
                act,
            ]
        else:
            hidden_ch = in_ch

        # 2. Depth-wise conv
        padding = kernel_size // 2
        layers += [
            nn.Conv1d(
                hidden_ch,
                hidden_ch,
                kernel_size,
                padding=padding,
                groups=hidden_ch,
                bias=False,
            ),
            _make_norm(hidden_ch, **norm_cfg),
            act,
        ]

        # 3. S/E
        if use_se:
            layers.append(SqueezeExcite(hidden_ch))

        # 4. Point-wise projection
        layers += [
            nn.Conv1d(hidden_ch, out_ch, 1, bias=True),
        ]

        self.block = nn.Sequential(*layers)
        
        # ✅ ADD THIS SECTION - Residual connection handling
        self._verbose = True  # set False to suppress per-call debug prints
        self.use_residual = True
        self.residual_scale = residual_scale
        self.residual_identity_scale = residual_identity_scale  # scale on shortcut path
        
        # If input/output channels differ, need 1x1 conv to match dimensions
        if in_ch != out_ch:
            self.residual_proj = nn.Conv1d(in_ch, out_ch, kernel_size=1, bias=False)
        else:
            self.residual_proj = None
    
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.block(x)
        out = x + out if self.use_residual else out
        
        # Apply upsampling if configured
        if self.upsample is not None:
            out = self.upsample(out)
        
        return out
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        v = self._verbose
        if v: print(f"[Block] Input: shape={x.shape}, mean={x.mean():.4f}, std={x.std():.4f}")
        
        # Capture identity BEFORE any modifications
        identity = x
        
        # Apply upsampling to BOTH paths if configured
        if self.upsample_first is not None:
            x = self.upsample_first(x)
            identity = self.upsample_first(identity)  # Now identity exists!
        
        # Process through block
        out = self.block(x)
        if v: print(f"[Block] After block: shape={out.shape}, mean={out.mean():.4f}, std={out.std():.4f}")
        
        # Apply residual connection
        if self.residual_proj is not None:
            identity = self.residual_proj(identity)
        
        out = self.residual_identity_scale * identity + self.residual_scale * out
        if v: print(f"[Block] After residual (id_scale={self.residual_identity_scale:.2f}, body_scale={self.residual_scale}): shape={out.shape}, mean={out.mean():.4f}, std={out.std():.4f}")
        
        return out




# --------------------------------------------------------------------------- #
#  Speaker-conditioned FiLM for block3                                       #
# --------------------------------------------------------------------------- #
class SpeakerFiLM(nn.Module):
    """Lightweight speaker-conditioned FiLM applied at block3 output.

    Combats speaker erasure at the block3 residual projection (cent_cos
    jumps from ~0.49 to ~0.75) by re-injecting speaker identity right
    before mel_proj.  Speaker tokens are pooled → 2-layer MLP → (gamma,
    beta) for 96 channels.  ELU on gamma ensures (1+gamma) ≥ 0
    (inversion guard, consistent with cross_attn).  Small gain=0.1 init
    ensures near-identity start; model learns scale through training.
    """
    def __init__(self, speaker_dim: int = 96, target_channels: int = 96):
        super().__init__()
        self.speaker_dim = speaker_dim
        self.target_channels = target_channels

        self.mlp = nn.Sequential(
            nn.Linear(speaker_dim, speaker_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(speaker_dim, target_channels * 2),
        )

        # Conservative init: near-identity start (gamma ≈ 0, beta ≈ 0)
        with torch.no_grad():
            for m in self.mlp.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight, gain=0.1)
                    nn.init.zeros_(m.bias)

        # Learnable film_scale: softplus reparam so scale can never collapse.
        # Matches cross_attn's raw_film_scale pattern.  Init -0.5 →
        # softplus(-0.5)+0.5 ≈ 1.13, giving the FiLM meaningful
        # authority from the start while allowing training to grow it.
        self.raw_film_scale = nn.Parameter(torch.tensor(-0.5))

        self._verbose = True

    def forward(self, x: torch.Tensor, speaker_feats: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:             [B, C, T] block3 output features
            speaker_feats: [B, num_tokens, speaker_dim] speaker tokens from mel_encoder
        Returns:
            x conditioned on speaker: [B, C, T]
        """
        # Pool speaker tokens: mean over token dimension → [B, speaker_dim]
        spk_pooled = speaker_feats.mean(dim=1)

        # Generate FiLM parameters
        film_params = self.mlp(spk_pooled)                       # [B, target_channels * 2]
        gamma, beta = film_params.chunk(2, dim=-1)               # [B, target_channels] each

        # Learnable scale (softplus reparam, always > 0.5)
        film_scale = F.softplus(self.raw_film_scale) + 0.5

        # Scale + ELU on gamma: ensures (1+gamma) ≥ 0 (inversion guard)
        gamma = F.elu(gamma * film_scale)
        beta  = beta * film_scale

        # Reshape for broadcasting over temporal dim: [B, C, 1]
        gamma = gamma.unsqueeze(-1)
        beta  = beta.unsqueeze(-1)

        if self._verbose:
            g = gamma.squeeze(-1)  # [B, C]
            b = beta.squeeze(-1)
            print(f"[speaker_film] gamma: mean={g.mean():.4f}, std={g.std():.4f}, "
                  f"min={g.min():.4f}, max={g.max():.4f}")
            print(f"[speaker_film] beta:  mean={b.mean():.4f}, std={b.std():.4f}, "
                  f"min={b.min():.4f}, max={b.max():.4f}")
            print(f"[speaker_film] film_scale: {film_scale.item():.4f}")
            print(f"[speaker_film] (1+gamma) range: [{(1+g).min():.4f}, {(1+g).max():.4f}]")

        # Apply FiLM: x * (1 + gamma) + beta
        x = x * (1.0 + gamma) + beta
        return x


# --------------------------------------------------------------------------- #
#  Block3 Identity-Path FiLM — speaker conditioning at residual_proj output  #
# --------------------------------------------------------------------------- #
class Block3IdentityFiLM(nn.Module):
    """Speaker-conditioned FiLM applied to block3's identity projection output.

    The block3 residual_proj (192→96) is the localized speaker eraser:
    cent_cos jumps from ~0.67 (block2) to ~0.87 (b3_identity).  Injecting
    speaker-dependent gamma/beta directly after residual_proj means no
    GroupNorm can undo it — the FiLM sits between residual_proj and the
    residual sum, outside any normalization layer.

    Same pattern as SpeakerFiLM: pooled speaker tokens → 2-layer MLP →
    gamma/beta for 96 channels, ELU guard on gamma, gain=0.1 init,
    learnable film_scale with softplus reparam.
    """
    def __init__(self, speaker_dim: int = 96, target_channels: int = 96):
        super().__init__()
        self.speaker_dim = speaker_dim
        self.target_channels = target_channels

        self.mlp = nn.Sequential(
            nn.Linear(speaker_dim, speaker_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(speaker_dim, target_channels * 2),
        )

        # Conservative init: near-identity start (gamma ≈ 0, beta ≈ 0)
        with torch.no_grad():
            for m in self.mlp.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight, gain=0.2)
                    nn.init.zeros_(m.bias)

        # Learnable film_scale: softplus reparam so scale can never collapse.
        # Init 0.5 → softplus(0.5)+0.5 ≈ 2.19, giving strong authority
        # from the start while allowing training to grow it.
        self.raw_film_scale = nn.Parameter(torch.tensor(0.5))

        self._verbose = True

    def forward(self, id_proj: torch.Tensor, speaker_feats: torch.Tensor) -> torch.Tensor:
        """
        Args:
            id_proj:        [B, 96, T] block3 residual_proj output (the erasure point)
            speaker_feats:  [B, num_tokens, 96] speaker tokens from mel_encoder
        Returns:
            id_proj conditioned on speaker: [B, 96, T]
        """
        # Pool speaker tokens: mean over token dimension → [B, speaker_dim]
        spk_pooled = speaker_feats.mean(dim=1)

        # Generate FiLM parameters
        film_params = self.mlp(spk_pooled)                       # [B, target_channels * 2]
        gamma, beta = film_params.chunk(2, dim=-1)               # [B, target_channels] each

        # Learnable scale (softplus reparam, always > 0.5)
        film_scale = F.softplus(self.raw_film_scale) + 0.5

        # Scale + ELU on gamma: ensures (1+gamma) ≥ 0 (inversion guard)
        gamma = F.elu(gamma * film_scale)
        beta  = beta * film_scale

        # Reshape for broadcasting over temporal dim: [B, C, 1]
        gamma = gamma.unsqueeze(-1)
        beta  = beta.unsqueeze(-1)

        if self._verbose:
            g = gamma.squeeze(-1)  # [B, C]
            b = beta.squeeze(-1)
            print(f"[b3_id_film] gamma: mean={g.mean():.4f}, std={g.std():.4f}, "
                  f"min={g.min():.4f}, max={g.max():.4f}")
            print(f"[b3_id_film] beta:  mean={b.mean():.4f}, std={b.std():.4f}, "
                  f"min={b.min():.4f}, max={b.max():.4f}")
            print(f"[b3_id_film] film_scale: {film_scale.item():.4f}")
            print(f"[b3_id_film] (1+gamma) range: [{(1+g).min():.4f}, {(1+g).max():.4f}]")

        # Apply FiLM: id_proj * (1 + gamma) + beta
        id_proj = id_proj * (1.0 + gamma) + beta
        return id_proj


# --------------------------------------------------------------------------- #
#  Decoder                                                                    #
# --------------------------------------------------------------------------- #
class MobileNetDecoder(nn.Module):
    """
    Turns fused features from cross-attention into log-mel spectrogram.
    All configuration comes from ModelConfig and AudioConfig.
    """

    def __init__(self, config: ModelConfig, return_feats: bool = False):
        super().__init__()
        # Remove old upsample_first logic
        self.config = config
        self.return_feats = return_feats
        
        # Extract parameters
        in_channels = config.mobilenet_input_dim
        channel_progression = config.mobilenet_channel_progression
        expand_ratios = config.mobilenet_expand_ratios
        use_se = config.mobilenet_use_se
        upsample_stages = config.mobilenet_upsample_stages  # NEW
        
        # Adapter (no upsampling here)
        self.adapter = nn.Sequential(
            nn.Conv1d(in_channels, channel_progression[0], kernel_size=1, bias=True),
        )
        
        # Build blocks with integrated upsampling
        blocks: List[nn.Module] = []
        n_blocks = len(expand_ratios)
        
        for idx in range(n_blocks):

          should_upsample = upsample_stages[idx]
          print(f"[INIT] Block {idx}: upsample_first={should_upsample}, "
                f"in_ch={channel_progression[idx]}, out_ch={channel_progression[idx+1]}")
          print(".........................................................................................................")
          blocks.append(
              MobileNetBlock(
                  in_ch=channel_progression[idx],
                  out_ch=channel_progression[idx + 1],
                  kernel_size=config.mobilenet_kernel_size_first if idx == 0 else config.mobilenet_kernel_size,
                  expand_ratio=expand_ratios[idx],
                  use_se=use_se[idx],
                  norm=config.mobilenet_norm,
                  upsample_first=should_upsample,
                  residual_scale=getattr(config, 'block3_residual_scale', 1.0) if idx == 3 else 1.0,
              )
            )
        self.blocks = nn.ModuleList(blocks)
        
        # Output projection (unchanged)
        self.mel_proj = nn.Conv1d(
            channel_progression[-1],  # 80 channels
            80,  # target mel bands
            kernel_size=1, 
            bias=True
        )
        
        # Compromise init for 96→80 mel_proj.
        # Identity anchor on first 80 channels preserves L1 stability and mel
        # variance scale (σ≈1.68), preventing the upstream regression seen with
        # full Xavier (gain=1.0, no identity). Channels 80-95 use Xavier(gain=0.5)
        # — 5× stronger than old gain=0.1 so the gradient can recruit them for
        # speaker-dependent structure without destabilizing training dynamics.
        # Hypothesis: this stops channels 80-95 starvation while keeping the
        # L1 anchor that blocks 0-2 need to converge.
        with torch.no_grad():
            weight_mat = self.mel_proj.weight.squeeze(-1)  # [80, 96]
            nn.init.xavier_uniform_(weight_mat, gain=0.5)
            weight_mat[:, :80] = torch.eye(80)
        if self.mel_proj.bias is not None:
            #nn.init.zeros_(self.mel_proj.bias)
            nn.init.constant_(self.mel_proj.bias, -4.5)  # Initialize in the unnormalized log-mel domain

        # Speaker-conditioned FiLM at block3 output: re-injects speaker
        # identity right before mel_proj, directly targeting the cent_cos
        # jump from ~0.49 (block2) to ~0.75 (block3_sum).  Pooled speaker
        # tokens → 2-layer MLP → gamma/beta for 96 channels.  ELU guard
        # on gamma, conservative gain=0.1 init for near-identity start.
        self.speaker_film = SpeakerFiLM(
            speaker_dim=config.speaker_projection_dim,     # 96
            target_channels=channel_progression[-1],       # 96
        )

        # Block3 identity-path FiLM: conditioned at the residual_proj output,
        # the localized speaker erasure point (cent_cos jumps from ~0.67 to
        # ~0.87).  Applied directly after 192→96 residual_proj, before the
        # residual sum — no GroupNorm between injection and usage.
        # Identity-path-only per consensus; body path only if audit shows need.
        self.block3_id_film = Block3IdentityFiLM(
            speaker_dim=config.speaker_projection_dim,     # 96
            target_channels=channel_progression[-1],       # 96
        )

        # Per-band output scale — SOFTPLUS reparametrised so scale can NEVER collapse.
        # Previously out_scale=1.0 (linear) was pulled back toward 1.0 by L1 mel loss
        # faster than lambda_var=30 could push it up, leaving mel σ at 1.4 vs target 2.5.
        # Now: out_scale = softplus(raw_out_scale) + 1.5  →  always > 1.5, init ≈ 2.19
        # This gives variance loss a structural head-start it cannot lose to L1 regression.
        self.raw_out_scale = nn.Parameter(torch.zeros(1, 80, 1))  # softplus(0) ≈ 0.693 → scale init ≈ 2.19

        # Per-band output bias: handles spectral tilt / per-band mean offset
        # independently of variance scaling.
        self.out_bias = nn.Parameter(torch.zeros(1, 80, 1))
        self._verbose = True  # set False to suppress per-call debug prints



    # ---------------------------------------------------------------------- #
    #  Forward                                                               #
    # ---------------------------------------------------------------------- #
    def compute_structure_preservation(self, current_tensor, reference_tensor):
        """
        Computes comprehensive metrics for information preservation
        between two tensors, with careful handling of edge cases.

        Returns:
            Dict with 'magnitude_preservation', 'per_channel_structure', 'global_structure'
        """
        
        # --- 1. Initial Validation ---
        if current_tensor.ndim != 3 or reference_tensor.ndim != 3:
            raise ValueError(f"Expected 3D tensors (B,C,T), got {current_tensor.shape}, {reference_tensor.shape}")

        if current_tensor.shape[0] != reference_tensor.shape[0]:
            raise ValueError(f"Batch sizes must match: {current_tensor.shape[0]} vs {reference_tensor.shape[0]}")

        # --- 2. Align Tensors ---
        min_channels = min(current_tensor.shape[1], reference_tensor.shape[1])
        min_t = min(current_tensor.shape[-1], reference_tensor.shape[-1])

        if min_channels < 1:
            raise ValueError(f"Cannot compute metrics with 0 channels. Got shapes: {current_tensor.shape}, {reference_tensor.shape}")
        if min_t < 2:
            raise ValueError(f"Temporal dimension must be at least 2, got aligned T={min_t}")

        current_aligned = current_tensor[:, :min_channels, :min_t]
        reference_aligned = reference_tensor[:, :min_channels, :min_t]
        
        epsilon = 1e-8

        # --- 3. Compute Metrics ---
        
        # Metric 1: Magnitude Preservation
        ref_std = reference_aligned.std()
        if ref_std < epsilon:
            magnitude_preservation = np.nan if current_aligned.std() > epsilon else 1.0
        else:
            magnitude_preservation = (current_aligned.std() / ref_std).item()
            
        # Metric 2: Per-Channel Structure
        per_channel_sim = F.cosine_similarity(current_aligned, reference_aligned, dim=-1)
        ref_norms = torch.linalg.norm(reference_aligned.to(torch.float32), dim=-1)
        per_channel_sim[ref_norms < epsilon] = np.nan
        per_channel_structure = torch.nanmean(per_channel_sim).item()

        # Metric 3: Global Structure
        current_temporal = current_aligned.mean(dim=1)
        reference_temporal = reference_aligned.mean(dim=1)
        current_global_vec = current_temporal.flatten()
        ref_global_vec = reference_temporal.flatten()
        
        ref_global_norm = torch.linalg.norm(ref_global_vec.to(torch.float32))
        if ref_global_norm < epsilon:
            global_structure = np.nan
        else:
            global_structure = F.cosine_similarity(current_global_vec, ref_global_vec, dim=0).item()

        return {
            "magnitude_preservation": magnitude_preservation,
            "per_channel_structure": per_channel_structure,
            "global_structure": global_structure
        }
                
        
    def _check(self, x, tag):
        """Debug check for finite values."""
        if not torch.isfinite(x).all():
            if self._verbose:
                print(f"❌  Non-finite values after {tag}")
                print("    mean:", x.mean().item(), "std:", x.std().item())
            raise RuntimeError("Stop trace here")
        else:
            if self._verbose:
                print(f"✅ {tag:16s}  mean={x.mean():7.4f}  std={x.std():7.4f}")

    
    def forward(self, fused: torch.Tensor, return_intermediate: bool = False, speaker_feats: torch.Tensor | None = None) -> Tuple[torch.Tensor, List[torch.Tensor]] | torch.Tensor:

        # Validate input
        if fused.ndim != 3:
            raise ValueError(f"Expected 3-D tensor (B,T,C); got {fused.shape}")
        B, T, C = fused.shape

        # Transpose to (B, C, T)
        x = fused.transpose(1, 2)
        
        # Adapter
        x = self.adapter(x)
        self._check(x, "adapter")
        
        # REMOVE old upsample_first logic entirely
        
        should_return_feats = return_intermediate or self.return_feats
        intermediate: List[torch.Tensor] = []

        # Process blocks — block3 is decomposed to inject identity-path FiLM
        # after residual_proj (192→96), the localized speaker erasure point.
        for i, blk in enumerate(self.blocks):
          x_before = x

          if i == 3 and speaker_feats is not None and self.block3_id_film is not None:
              # ── Decompose block3 to inject FiLM at residual_proj output ──
              identity = x
              # No upsample for block3 (upsample_stages[3] = False)
              if blk.upsample_first is not None:
                  x = blk.upsample_first(x)
                  identity = blk.upsample_first(identity)
              # Body path
              body = blk.block(x)
              if blk._verbose:
                  print(f"[Block] After block: shape={body.shape}, mean={body.mean():.4f}, std={body.std():.4f}")
              # Identity projection: 192→96 (the erasure point)
              if blk.residual_proj is not None:
                  id_proj = blk.residual_proj(identity)
              else:
                  id_proj = identity
              # ── INJECT: speaker-conditioned FiLM at identity projection output ──
              id_proj = self.block3_id_film(id_proj, speaker_feats)
              if self._verbose:
                  self._check(id_proj, "b3_id_film")
              # Residual sum: id_scale * id_proj + body_scale * body
              x = blk.residual_identity_scale * id_proj + blk.residual_scale * body
              if blk._verbose:
                  print(f"[Block] After residual (id_scale={blk.residual_identity_scale:.2f}, body_scale={blk.residual_scale}): shape={x.shape}, mean={x.mean():.4f}, std={x.std():.4f}")
          else:
              x = blk(x)

          self._check(x, f"block {i}")
          
          if should_return_feats:
              intermediate.append(x)
          
        
        # Speaker FiLM at block3 output: re-inject speaker identity
        # before mel_proj compresses it away.  Without this, block3's
        # 192→96 residual_proj crushes speaker info (cent_cos 0.49→0.75).
        if speaker_feats is not None:
            x = self.speaker_film(x, speaker_feats)
            self._check(x, "spk_film")

        mel = self.mel_proj(x)

        # ── Decoder output diagnostics ──────────────────────────────────
        # We log three stages separately: (1) raw conv output, (2) after
        # per-band out_scale, (3) after clamp.  This isolates whether the
        # mel variance deficit comes from the conv backbone, insufficient
        # out_scale growth, or clamp compression.
        # Compute effective out_scale via softplus reparametrisation (always > 1.5)
        out_scale = F.softplus(self.raw_out_scale) + 1.5   # [1, 80, 1]
        out_bias_vals  = self.out_bias.squeeze()            # [80]
        out_scale_vals = out_scale.squeeze()                # [80]
        v = self._verbose
        if v:
            print(f"[decoder] out_scale (softplus+1.5): mean={out_scale_vals.mean().item():.4f}, "
                  f"min={out_scale_vals.min().item():.4f}, max={out_scale_vals.max().item():.4f}")
            print(f"[decoder] raw_out_scale: mean={self.raw_out_scale.mean().item():.4f}")
            print(f"[decoder] out_bias:  mean={out_bias_vals.mean().item():.4f}, "
                  f"min={out_bias_vals.min().item():.4f}, max={out_bias_vals.max().item():.4f}")

        mel_pre_scale_std = mel.std().item()
        mel_pre_scale_mean = mel.mean().item()
        if v: print(f"[decoder] pre-scale  mel: mean={mel_pre_scale_mean:.4f}, std={mel_pre_scale_std:.4f}")

        # DECOUPLED SCALING: center per-band temporal mean so out_scale is a PURE variance knob.
        # Previously: mel * out_scale + bias  →  increasing out_scale also shifts mean,
        # causing L1 gradient to fight the variance gradient (net ≈ 0, scale frozen).
        # Now: out_scale only amplifies the deviation from each band's own mean.
        # L1 gradient on out_scale ≈ 0 (centered input is symmetric around 0).
        # Variance gradient is uncontested → raw_out_scale can grow freely.
        mel_band_mean = mel.mean(dim=-1, keepdim=True)       # [B, 80, 1] per-band temporal mean
        mel_centered  = mel - mel_band_mean                   # zero-mean per band per utterance
        mel_scaled    = mel_centered * out_scale + mel_band_mean + self.out_bias
        mel_post_scale_std = mel_scaled.std().item()
        mel_post_scale_mean = mel_scaled.mean().item()
        if v: print(f"[decoder] post-scale pre-clamp mel: mean={mel_post_scale_mean:.4f}, std={mel_post_scale_std:.4f}")

        mel = torch.clamp(mel_scaled, min=-11.5, max=2.0)

        clamp_mask = (mel_scaled < -11.5) | (mel_scaled > 2.0)
        clamp_pct = clamp_mask.float().mean().item() * 100
        if v: print(f"[decoder] clamp saturation: {clamp_pct:.2f}% of values at boundary")

        self._check(mel, "mel_proj")
        
        if should_return_feats:
            return mel, intermediate
        return mel
    
    # ---------------------------------------------------------------------- #
    #  Utilities                                                             #
    # ---------------------------------------------------------------------- #
    def get_parameter_count(self) -> Dict[str, int]:
        """Get trainable parameter count."""
        return {
            "total": sum(p.numel() for p in self.parameters() if p.requires_grad)
        }
    
    def register_block0_grad_hooks(self, verbose: bool = True):
        """
        Attach backward hooks to first decoder block parameters.
        Prints grad stats when loss.backward() runs.
        """
        if not hasattr(self, "blocks") or len(self.blocks) == 0:
            raise RuntimeError("No decoder blocks found to hook.")

        block0 = self.blocks[0]

        def make_hook(name: str):
            def hook(grad: torch.Tensor):
                finite = torch.isfinite(grad).all().item()
                if verbose:
                    print(f"[HOOK] {name:30s} | mean={grad.mean().item():.3e} "
                          f"| std={grad.std().item():.3e} | finite={finite}")
            return hook

        for name, p in block0.named_parameters():
            if p.requires_grad:
                p.register_hook(make_hook(name))

    @property
    def device(self) -> torch.device:
        """Get device of model parameters."""
        return next(self.parameters()).device


# --------------------------------------------------------------------------- #
#  Test                                                                       #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    from config import ModelConfig, AudioConfig
    
    torch.manual_seed(0)
    
    # Create configs
    model_config = ModelConfig()
    audio_config = AudioConfig()
    
    # Create decoder
    decoder = MobileNetDecoder(
        config=model_config,
        return_feats=True
    )
    
    # Test with dummy input
    dummy = torch.randn(3, 100, model_config.mobilenet_input_dim)
    mel, feats = decoder(dummy)
    
    print(f"\nMel shape: {mel.shape}")
    for i, f in enumerate(feats):
        print(f"Block {i} feat shape: {f.shape}")
