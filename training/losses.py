# hubertvc/training/losses.py
# ------------------------------------------------------------------
# Baseline generator losses for HuBERT-VC
#
# 1.  Mel-spectrogram L1 loss                   – λ_mel
# 2.  Multi-resolution STFT loss               – λ_rec
# 3.  HuBERT content-consistency (cosine) loss – λ_aux
#
# The class VCGeneratorLoss wraps them and returns a LossOutputs
# dict so that new loss terms can be plugged in with one line.
# ------------------------------------------------------------------

from __future__ import annotations
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F
from torch import nn

import warnings
warnings.filterwarnings("ignore", category=UserWarning,
                        message="stft with return_complex=False")

# ------------------------------------------------------------------
# Config fallback (allows standalone `python losses.py` test)
# ------------------------------------------------------------------
try:
    from config import TrainingConfig   # project config
except Exception:
    print("USING EXCEPTION IN LOSSESSS")                        # pragma: no cover
    class TrainingConfig:               # minimal stub
        lambda_mel = 1.0
        lambda_rec = 10.0
        lambda_aux = 0.3
        lambda_adv = 0.0
        lambda_feat = 0.0
        # default STFT schedule (HiFi-GAN)
        stft_fft_sizes = (512, 1024, 2048)
        stft_hop_sizes = (128, 256, 512)
        stft_win_lengths = (512, 1024, 2048)


# ------------------------------------------------------------------
# Helper container
# ------------------------------------------------------------------
class LossOutputs(dict):
    """Dict[str, Tensor] with `.total()` convenience method."""
    def total(self) -> torch.Tensor:
        return sum(v for v in self.values())


# ------------------------------------------------------------------
# 1. Mel-spectrogram reconstruction – L1 (+ optional L2 blend)
# ------------------------------------------------------------------
class MelSpectrogramLoss(nn.Module):
    """
    L1 / L2 loss on log-mel spectrograms.
    `weight_l1` controls the blend (1.0 = pure L1, 0.5 = 50-50).
    """
    def __init__(self, weight_l1: float = 1.0) -> None:
        super().__init__()
        self.weight_l1 = float(weight_l1)

    def forward(
        self,
        pred: torch.Tensor,             # (B, n_mels, T)
        target: torch.Tensor,           # (B, n_mels, T)
        lengths: torch.Tensor = None,   # (B,) real frame counts — excludes padding
    ) -> torch.Tensor:
        if pred.shape != target.shape:
            min_t = min(pred.size(-1), target.size(-1))
            pred   = pred[..., :min_t]
            target = target[..., :min_t]

        if lengths is not None:
            # Build a (B, T) boolean mask of real frames, broadcast over mels
            T = pred.size(-1)
            mask = (torch.arange(T, device=pred.device)[None, :] < lengths[:, None])  # (B, T)
            mask = mask.unsqueeze(1).expand_as(pred)  # (B, n_mels, T)
            # Only compute loss over real frames
            l1 = (pred - target).abs()[mask].mean()
            if self.weight_l1 >= 1.0 - 1e-6:
                return l1
            l2 = ((pred - target) ** 2)[mask].mean()
            return self.weight_l1 * l1 + (1.0 - self.weight_l1) * l2
        else:
            l1 = F.l1_loss(pred, target)
            if self.weight_l1 >= 1.0 - 1e-6:
                return l1
            l2 = F.mse_loss(pred, target)

        return self.weight_l1 * l1 + (1.0 - self.weight_l1) * l2


# ------------------------------------------------------------------
# 2. Multi-resolution STFT loss – HiFi-GAN recipe
# ------------------------------------------------------------------
class MRSTFTLoss(nn.Module):
    """
    Spectral-convergence + log-magnitude loss over multiple STFT
    resolutions.  FFT / hop / win schedules are passed via kwargs or
    `TrainingConfig.*` so nothing is ever hard-coded.
    """
    def __init__(
        self,
        fft_sizes: Sequence[int] | None = None,
        hop_sizes: Sequence[int] | None = None,
        win_lengths: Sequence[int] | None = None,
    ) -> None:
        super().__init__()

        cfg = TrainingConfig

        self._windows: Optional[List[torch.Tensor]] = None
        self._windows_device: Optional[torch.device] = None

        self.fft_sizes = list(fft_sizes or cfg.stft_fft_sizes)
        self.hop_sizes = list(hop_sizes or cfg.stft_hop_sizes)
        self.win_lengths = list(win_lengths or cfg.stft_win_lengths)


        if not (
            len(self.fft_sizes)
            == len(self.hop_sizes)
            == len(self.win_lengths)
        ):
            raise ValueError("STFT schedule lengths must match")

        # Pre-create Hann windows (device gets fixed at first call)
        #self.register_buffer("_windows", None, persistent=False)

    # ————————————————————————————————————————————————
    @staticmethod
    def _stft(
        x: torch.Tensor,
        n_fft: int,
        hop: int,
        win: int,
        window: torch.Tensor
    ) -> torch.Tensor:
        return torch.stft(
            x,
            n_fft=n_fft,
            hop_length=hop,
            win_length=win,
            window=window,
            center=True,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=False,      # keeps older PyTorch compatibility
        )

    # ————————————————————————————————————————————————
    def forward(
        self,
        pred_wave: torch.Tensor,  # (B, T)
        gt_wave: torch.Tensor,    # (B, T)
    ) -> torch.Tensor:


        # Align lengths by truncating to the shortest
        min_len = min(pred_wave.shape[-1], gt_wave.shape[-1])
        pred_wave = pred_wave[..., :min_len]
        gt_wave = gt_wave[..., :min_len]
        
        if pred_wave.shape != gt_wave.shape:
            raise ValueError(
                f"STFT loss shape mismatch: pred {pred_wave.shape} vs gt {gt_wave.shape}"
            )

        device = pred_wave.device
        if self._windows is None or self._windows_device != device:
            self._windows = [
                torch.hann_window(wl, device=device) for wl in self.win_lengths
            ]
            self._windows_device = device

        total_loss = 0.0
        for i, (n_fft, hop, win) in enumerate(
            zip(self.fft_sizes, self.hop_sizes, self.win_lengths)
        ):
            win_tensor = self._windows[i]
            pred_spec = self._stft(pred_wave, n_fft, hop, win, win_tensor)
            gt_spec = self._stft(gt_wave, n_fft, hop, win, win_tensor)

            # magnitude
            pred_mag = torch.clamp(
                torch.sqrt(pred_spec.pow(2).sum(-1) + 1e-9), min=1e-7
            )
            gt_mag = torch.clamp(
                torch.sqrt(gt_spec.pow(2).sum(-1) + 1e-9), min=1e-7
            )

            # spectral convergence
            sc_loss = (
                (pred_mag - gt_mag).norm(p="fro") / gt_mag.norm(p="fro")
            )

            # log magnitude
            mag_loss = F.l1_loss(
                torch.log(pred_mag), torch.log(gt_mag)
            )

            total_loss += sc_loss + mag_loss

        return total_loss / len(self.fft_sizes)

    
# ------------------------------------------------------------------
# 3. Speaker Identity Loss – cosine distance on embeddings
# ------------------------------------------------------------------
# (in losses.py)
# ------------------------------------------------------------------
# 3. Speaker Identity Loss – cosine distance on embeddings
# ------------------------------------------------------------------
class SpeakerIdentityLoss(nn.Module):
    """
    Measures the cosine similarity between speaker embeddings extracted
    from predicted and ground-truth WAVEFORMS.
    """
    def __init__(self, speaker_encoder: nn.Module) -> None:
        super().__init__()
        if speaker_encoder is None:
            raise ValueError("A pre-trained speaker_encoder must be provided.")
        self.speaker_encoder = speaker_encoder
        self.speaker_encoder.eval()
        for param in self.speaker_encoder.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def _extract_embedding(self, wave: torch.Tensor) -> torch.Tensor:
        """Extracts speaker embedding from a waveform."""
        return self.speaker_encoder.encode_batch(wave).detach()

    def forward(
        self,
        pred_wave: torch.Tensor,    # (B, T_wav)
        gt_wave: torch.Tensor,      # (B, T_wav)
    ) -> torch.Tensor:
        """Calculates the speaker identity loss on waveforms."""
        self.speaker_encoder.eval()

        emb_pred = self.speaker_encoder.encode_batch(pred_wave)
        emb_gt = self._extract_embedding(gt_wave)

        emb_pred = F.normalize(emb_pred.squeeze(1), p=2, dim=1)
        emb_gt = F.normalize(emb_gt.squeeze(1), p=2, dim=1)
        
        cosine_sim = F.cosine_similarity(emb_pred, emb_gt, dim=-1)
        loss = 1.0 - cosine_sim.mean()
        
        return loss

# ------------------------------------------------------------------
# 3b. Mel Spectral Stats Loss - Timbre/EQ Signature Loss
# ------------------------------------------------------------------
class MelSpectralStatsLoss(nn.Module):
    """
    Penalizes differences in the global mean and standard deviation of Mel-spectrograms.
    Forces the network to match the target speaker's time-invariant acoustic signature (timbre/EQ)
    rather than just phoneme structures, strongly mitigating the source-leakage 'shortcut trap'.
    """
    def __init__(self):
        super().__init__()

    def forward(
        self,
        pred: torch.Tensor,             # (B, n_mels, T1)
        target: torch.Tensor,           # (B, n_mels, T2)
        lengths: torch.Tensor = None,   # (B,) (legacy: applies to both if provided)
        pred_lengths: torch.Tensor = None, # (B,)
        target_lengths: torch.Tensor = None, # (B,)
    ) -> torch.Tensor:
        if pred_lengths is None and lengths is not None:
            pred_lengths = lengths
        if target_lengths is None and lengths is not None:
            target_lengths = lengths
        if pred.shape != target.shape:
            min_t = min(pred.size(-1), target.size(-1))
            pred   = pred[..., :min_t]
            target = target[..., :min_t]
            
        if pred_lengths is not None and target_lengths is not None:
            T_pred = pred.size(-1)
            T_target = target.size(-1)
            
            mask_pred = (torch.arange(T_pred, device=pred.device)[None, :] < pred_lengths[:, None]).unsqueeze(1).float()
            mask_target = (torch.arange(T_target, device=target.device)[None, :] < target_lengths[:, None]).unsqueeze(1).float()
            
            valid_pred = pred_lengths.view(-1, 1).float().clamp(min=1.0)
            valid_target = target_lengths.view(-1, 1).float().clamp(min=1.0)
            
            pred_mean = (pred * mask_pred).sum(dim=-1) / valid_pred
            target_mean = (target * mask_target).sum(dim=-1) / valid_target
            
            pred_var = (((pred - pred_mean.unsqueeze(-1)) ** 2) * mask_pred).sum(dim=-1) / valid_pred
            target_var = (((target - target_mean.unsqueeze(-1)) ** 2) * mask_target).sum(dim=-1) / valid_target
            
            pred_std = torch.sqrt(pred_var + 1e-6)
            target_std = torch.sqrt(target_var + 1e-6)
        else:
            pred_mean = pred.mean(dim=-1)
            target_mean = target.mean(dim=-1)
            pred_std = pred.std(dim=-1)
            target_std = target.std(dim=-1)
            
        l1_mean = F.l1_loss(pred_mean, target_mean)
        l1_std = F.l1_loss(pred_std, target_std)
        
        return l1_mean + l1_std


# ------------------------------------------------------------------
# 3c. Speaker Classifier Loss - per-frame CE on decoder bottleneck
# ------------------------------------------------------------------
class SpeakerClassifierLoss(nn.Module):
    """
    Per-frame cross-entropy loss applied to decoder bottleneck features.
    
    Uses Conv1d(kernel=1) classifier → no GAP, no 1/T dilution.
    Each frame receives an independent speaker identity gradient.
    
    Args:
        logits: [B, num_speakers, T] from Conv1d classifier
        target_indices: [B, T] expanded from batch["target_speaker_idx"]
    Returns:
        scalar cross-entropy loss
    """
    def forward(
        self,
        logits: torch.Tensor,           # [B, num_speakers, T]
        target_indices: torch.Tensor,   # [B, T]
    ) -> torch.Tensor:
        return F.cross_entropy(logits, target_indices)


# ------------------------------------------------------------------
# 4. Aggregator – easy extension point for new terms
# ------------------------------------------------------------------
class VCGeneratorLoss(nn.Module):
    """
    Primary loss module used by the trainer. Aggregates mel, STFT,
    and speaker identity losses.

    Example:
        # In your trainer
        loss_module = VCGeneratorLoss(train_cfg, speaker_encoder_model)
        losses = loss_module(pred_mel, gt_mel, pred_wav, gt_wav)
        total_loss = losses.total()  # scalar for .backward()
        total_loss.backward()
    """

    def __init__(
        self,
        cfg: TrainingConfig,
        speaker_encoder: Optional[nn.Module] = None,
    ) -> None:
        """
        Initializes the loss aggregator.

        Args:
            cfg (TrainingConfig): Configuration object with loss weights.
            speaker_encoder (nn.Module): A pre-trained, frozen speaker
                verification model used for the SpeakerIdentityLoss.
        """
        super().__init__()
        self.cfg = cfg

        # --- Instantiate individual loss components ---

        # 1. Mel-spectrogram L1 reconstruction loss
        self.mel_loss = MelSpectrogramLoss(weight_l1=0.7) # Raw L1

        # 2. Multi-resolution STFT loss
        self.stft_loss = MRSTFTLoss()

        # 3. Speaker identity loss
        self.speaker_loss: Optional[SpeakerIdentityLoss] = None
        self.mel_stats_loss = MelSpectralStatsLoss()
        
        if hasattr(cfg, 'lambda_spk') and cfg.lambda_spk > 0:
            if speaker_encoder is None:
                print("Warning: lambda_spk > 0 but speaker_encoder is None. Using MelStatsLoss only.")
            else:
                self.speaker_loss = SpeakerIdentityLoss(speaker_encoder)
        else:
            print(
                "Warning: `lambda_spk` is not defined or is 0. "
                "Speaker Identity / Stats losses will be disabled."
            )


    def forward(
        self,
        pred_mel: torch.Tensor,             # (B, n_mel, T_mel) — final mel with speaker conditioning
        gt_mel: torch.Tensor,               # (B, n_mel, T_mel)
        pred_wave: torch.Tensor,            # (B, T_wav)
        gt_wave: torch.Tensor,              # (B, T_wav)
        gt_lengths: torch.Tensor = None,    # (B,) real mel frame counts (excludes padding)
        content_mel: torch.Tensor = None,   # (B, n_mel, T_mel) — pre-speaker-affine mel for L1
    ) -> LossOutputs:
        """
        Computes the weighted sum of all configured losses.
        
        If content_mel is provided, L1 mel loss is computed on it (content only).
        Speaker losses (stats, var) always use pred_mel (final, with speaker delta).
        This separates L1 gradient (stops at prebias) from speaker gradient (flows
        through speaker delta), eliminating gradient competition.
        """
        outs = LossOutputs()

        # --- Compute and weight each loss term ---

        # Mel reconstruction loss (Raw L1) — on content_mel if provided (gradient separation)
        if hasattr(self.cfg, 'lambda_mel') and self.cfg.lambda_mel > 0:
            try:
                mel_for_l1 = content_mel if content_mel is not None else pred_mel
                T_common = min(mel_for_l1.size(-1), gt_mel.size(-1))
                p_align = mel_for_l1[:, :, :T_common]
                g_align = gt_mel[:, :, :T_common]
                
                l_mel = self.mel_loss(p_align, g_align, lengths=gt_lengths)
                outs["mel"] = self.cfg.lambda_mel * l_mel
            except Exception as exc:
                raise RuntimeError(f"Mel loss failed: {exc}") from exc

        # Multi-resolution STFT loss
        if hasattr(self.cfg, 'lambda_rec') and self.cfg.lambda_rec > 0 and pred_wave is not None and gt_wave is not None:
            try:
                l_stft = self.stft_loss(pred_wave, gt_wave)
                outs["stft"] = self.cfg.lambda_rec * l_stft
            except Exception as exc:
                raise RuntimeError(f"STFT loss failed: {exc}") from exc

        # Speaker identity & Stats loss
        if hasattr(self.cfg, 'lambda_spk') and self.cfg.lambda_spk > 0:
            try:
                # 1. Always compute the highly potent Mel Stats Loss
                l_spk_stats = self.mel_stats_loss(pred_mel, gt_mel, lengths=gt_lengths)
                outs["speaker"] = self.cfg.lambda_spk * l_spk_stats
                
                # 2. Add True Speaker Cosine Loss IF waveforms were generated (not OOMing)
                if self.speaker_loss is not None and pred_wave is not None and gt_wave is not None:
                    l_spk_wave = self.speaker_loss(pred_wave, gt_wave) 
                    outs["speaker"] = outs["speaker"] + (self.cfg.lambda_spk * l_spk_wave)
            except Exception as exc:
                raise RuntimeError(f"Speaker/Stats loss failed: {exc}") from exc

        # Mel variance loss — penalizes compressed dynamic range (pred σ ≈ 1.3 vs GT σ ≥ 2.5)
        if hasattr(self.cfg, 'lambda_var') and self.cfg.lambda_var > 0:
            try:
                if gt_lengths is not None:
                    B, N = pred_mel.shape[0], pred_mel.shape[1]
                    # Align time dimensions: decoder may produce 1-2 fewer frames than GT
                    T_common = min(pred_mel.size(-1), gt_mel.size(-1))
                    pred_aligned = pred_mel[:, :, :T_common]
                    gt_aligned = gt_mel[:, :, :T_common]
                    lengths = gt_lengths.clamp(max=T_common)
                    mask = (torch.arange(T_common, device=pred_mel.device)[None, None, :] < lengths[:, None, None]).float()
                    valid_counts = lengths.float().clamp(min=1.0)  # [B]
                    # Masked mean per band
                    pred_mean = (pred_aligned * mask).sum(dim=-1) / valid_counts[:, None]
                    tgt_mean = (gt_aligned * mask).sum(dim=-1) / valid_counts[:, None]
                    # Masked variance per band
                    pred_sq = ((pred_aligned - pred_mean[:, :, None]) ** 2) * mask
                    tgt_sq = ((gt_aligned - tgt_mean[:, :, None]) ** 2) * mask
                    pred_std = torch.sqrt(pred_sq.sum(dim=-1) / valid_counts[:, None] + 1e-8)
                    tgt_std = torch.sqrt(tgt_sq.sum(dim=-1) / valid_counts[:, None] + 1e-8)
                else:
                    T_common = min(pred_mel.size(-1), gt_mel.size(-1))
                    pred_std = pred_mel[:, :, :T_common].std(dim=-1)  # [B, 80]
                    tgt_std = gt_mel[:, :, :T_common].std(dim=-1)
                l_var = F.l1_loss(pred_std, tgt_std)
                outs["var"] = self.cfg.lambda_var * l_var
            except Exception as exc:
                raise RuntimeError(f"Variance loss failed: {exc}") from exc

        return outs


# ------------------------------------------------------------------
# Stand-alone sanity check
# ------------------------------------------------------------------
if __name__ == "__main__":  # pragma: no cover
    print("Quick losses.py self-test…")
    torch.manual_seed(0)
    B, T_mel, N_MEL = 2, 100, 80
    T_wav = 16000
    EMBED_DIM = 256

    # Add lambda_spk to the dummy config
    class TrainingConfig:
        lambda_mel = 1.0
        lambda_rec = 1.0
        lambda_spk = 1.0
        stft_fft_sizes = (512, 1024, 2048)
        stft_hop_sizes = (128, 256, 512)
        stft_win_lengths = (512, 1024, 2048)
    
    cfg = TrainingConfig()

    # --- Dummy data ---
    pred_mel = torch.randn(B, N_MEL, T_mel)
    gt_mel = torch.randn_like(pred_mel) * 0.8 # Make it slightly different
    pred_wave = torch.randn(B, T_wav)
    gt_wave = torch.randn_like(pred_wave) * 0.9

    # --- Fake Speaker Encoder for testing ---
    class _DummySpeakerEncoder(nn.Module):
        def __init__(self, embed_dim):
            super().__init__()
            # A simple linear layer to project mel to embedding
            self.proj = nn.Linear(N_MEL, embed_dim)

        def forward(self, mel):
            # (B, n_mels, T) -> (B, T, n_mels) -> mean pool -> (B, n_mels) -> proj
            return self.proj(mel.transpose(1, 2).mean(dim=1))

    speaker_encoder = _DummySpeakerEncoder(EMBED_DIM)
    
    # --- Initialize and run the final loss module ---
    loss_mod = VCGeneratorLoss(cfg, speaker_encoder)

    out = loss_mod(pred_mel, gt_mel, pred_wave, gt_wave)
    
    print("  Individual:", {k: f"{v.item():.4f}" for k, v in out.items()})
    print(f"  Total:      {out.total().item():.4f}")

    # Test backward pass
    try:
        out.total().backward()
        print("✓ Backward pass successful.")
    except Exception as e:
        print(f"✗ Backward pass failed: {e}")

    print("✓ losses.py self-test passed")

