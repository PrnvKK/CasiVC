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
        residual_scale: float = 1.0
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
        self.use_residual = True
        self.residual_scale = residual_scale
        
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
        print(f"[Block] Input: shape={x.shape}, mean={x.mean():.4f}, std={x.std():.4f}")
        
        # Capture identity BEFORE any modifications
        identity = x
        
        # Apply upsampling to BOTH paths if configured
        if self.upsample_first is not None:
            x = self.upsample_first(x)
            identity = self.upsample_first(identity)  # Now identity exists!
        
        # Process through block
        out = self.block(x)
        print(f"[Block] After block: shape={out.shape}, mean={out.mean():.4f}, std={out.std():.4f}")
        
        # Apply residual connection
        if self.residual_proj is not None:
            identity = self.residual_proj(identity)
        
        out = identity + self.residual_scale * out  # Using your residual_scale
        print(f"[Block] After residual (scale={self.residual_scale}): shape={out.shape}, mean={out.mean():.4f}, std={out.std():.4f}")
        
        return out




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
        
        # Identity-initialized 1×1 projection: preserves block3 variance (~2.0)
        # instead of compressing it by ~50% as Xavier init does.  Each output
        # mel band starts as its corresponding input channel + bias, then
        # training learns cross-channel corrections if needed.
        nn.init.eye_(self.mel_proj.weight.squeeze(-1))
        if self.mel_proj.bias is not None:
            #nn.init.zeros_(self.mel_proj.bias)
            nn.init.constant_(self.mel_proj.bias, -4.5)  # Initialize in the unnormalized log-mel domain

        # Per-band output scale: identity mel_proj preserves block3 σ≈2.0,
        # so out_scale starts at 1.0 (no amplification needed initially).
        # Variance loss (λ=15) can push it from 2.0→2.5 without fighting
        # through mel_proj compression first.
        self.out_scale = nn.Parameter(torch.ones(1, 80, 1) * 1.0)

        # Per-band output bias: handles spectral tilt / per-band mean offset
        # independently of variance scaling.  Without this, the single mel_proj
        # bias (-4.5) is shared across all 80 bands, coupling mean correction
        # to variance correction through out_scale alone.  Adding a per-band
        # bias lets out_scale focus on variance while bias handles spectral shape.
        self.out_bias = nn.Parameter(torch.zeros(1, 80, 1))



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
            print(f"❌  Non-finite values after {tag}")
            print("    mean:", x.mean().item(), "std:", x.std().item())
            raise RuntimeError("Stop trace here")
        else:
            print(f"✅ {tag:16s}  mean={x.mean():7.4f}  std={x.std():7.4f}")

    
    def forward(self, fused: torch.Tensor, return_intermediate: bool = False) -> Tuple[torch.Tensor, List[torch.Tensor]] | torch.Tensor:

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

        # Process blocks (upsampling now happens inside blocks)
        for i, blk in enumerate(self.blocks):

          x_before = x
          x = blk(x)  # Block now handles its own upsampling
          self._check(x, f"block {i}")
          
          if should_return_feats:
              intermediate.append(x)
          
        
        mel = self.mel_proj(x)

        # ── Decoder output diagnostics ──────────────────────────────────
        # We log three stages separately: (1) raw conv output, (2) after
        # per-band out_scale, (3) after clamp.  This isolates whether the
        # mel variance deficit comes from the conv backbone, insufficient
        # out_scale growth, or clamp compression.
        out_scale_vals = self.out_scale.squeeze()  # [80]
        out_bias_vals  = self.out_bias.squeeze()   # [80]
        print(f"[decoder] out_scale: mean={out_scale_vals.mean().item():.4f}, "
              f"min={out_scale_vals.min().item():.4f}, max={out_scale_vals.max().item():.4f}")
        print(f"[decoder] out_bias:  mean={out_bias_vals.mean().item():.4f}, "
              f"min={out_bias_vals.min().item():.4f}, max={out_bias_vals.max().item():.4f}")

        mel_pre_scale_std = mel.std().item()
        mel_pre_scale_mean = mel.mean().item()
        print(f"[decoder] pre-scale  mel: mean={mel_pre_scale_mean:.4f}, std={mel_pre_scale_std:.4f}")

        mel_scaled = mel * self.out_scale + self.out_bias
        mel_post_scale_std = mel_scaled.std().item()
        mel_post_scale_mean = mel_scaled.mean().item()
        print(f"[decoder] post-scale pre-clamp mel: mean={mel_post_scale_mean:.4f}, std={mel_post_scale_std:.4f}")

        mel = torch.clamp(mel_scaled, min=-11.5, max=1.7)

        clamp_mask = (mel_scaled < -11.5) | (mel_scaled > 1.7)
        clamp_pct = clamp_mask.float().mean().item() * 100
        print(f"[decoder] clamp saturation: {clamp_pct:.2f}% of values at boundary")

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
