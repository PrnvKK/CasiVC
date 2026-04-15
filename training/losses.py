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
    DEPRECATED: Previously used as a proxy for speaker identity. Comparing mel stats
    against the SOURCE speaker mel actively fights conversion. Kept for reference only.
    """
    def __init__(self):
        super().__init__()

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        lengths: torch.Tensor = None,
    ) -> torch.Tensor:
        if pred.shape != target.shape:
            min_t = min(pred.size(-1), target.size(-1))
            pred   = pred[..., :min_t]
            target = target[..., :min_t]
            
        pred_mean = pred.mean(dim=-1)
        target_mean = target.mean(dim=-1)
        pred_std = pred.std(dim=-1)
        target_std = target.std(dim=-1)
            
        l1_mean = F.l1_loss(pred_mean, target_mean)
        l1_std = F.l1_loss(pred_std, target_std)
        
        return l1_mean + l1_std


# ------------------------------------------------------------------
# 3b. Band-Stats Speaker Loss  — direct per-band mel statistics L1
# ------------------------------------------------------------------
class BandStatsSpeakerLoss(nn.Module):
    """
    Compares per-frequency-band mean and std of the PREDICTED mel against
    the TARGET speaker's ground-truth mel statistics.

    No trainable layers. Gradient flows directly through the comparison
    of mel statistics back into the decoder and mapping network.

    Three perceptually-motivated bands (for 80 mel-band log-mel):
      Low  [0 :20]: captures F0 / pitch region (80-400 Hz) — primary gender cue
      Mid  [20:60]: captures formants/resonance  (400-3000 Hz)
      High [60:80]: captures breathiness/sibilance (3000-8000 Hz)

    Uses gt_lengths mask so padding silence does not corrupt the statistics.
    """
    LOW  = (0,  20)
    MID  = (20, 60)
    HIGH = (60, 80)

    def _masked_stats(self, mel: torch.Tensor, lengths: torch.Tensor):
        """
        Compute per-band mean and std, ignoring padded frames.
        mel:     (B, 80, T)
        lengths: (B,)  real frame counts
        Returns: dict of band -> (mean (B, band_size), std (B, band_size))
        """
        B, C, T = mel.shape
        # Build (B, 1, T) mask of real frames
        mask = (torch.arange(T, device=mel.device)[None, None, :]
                < lengths[:, None, None].float())       # (B, 1, T)
        n = lengths.float().clamp(min=1.0).unsqueeze(-1)  # (B, 1)

        stats = {}
        for band_name, (lo, hi) in [("low", self.LOW), ("mid", self.MID), ("high", self.HIGH)]:
            band = mel[:, lo:hi, :]                     # (B, band_size, T)
            band_mask = mask.expand_as(band)            # (B, band_size, T)

            masked = band * band_mask
            band_mean = masked.sum(dim=-1) / n          # (B, band_size)

            # Variance via E[X^2] - E[X]^2, masked
            sq_mean = (masked ** 2).sum(dim=-1) / n     # (B, band_size)
            band_var = (sq_mean - band_mean ** 2).clamp(min=0.0)
            band_std = torch.sqrt(band_var + 1e-6)      # (B, band_size)

            stats[band_name] = (band_mean, band_std)
        return stats

    def forward(
        self,
        pred_mel: torch.Tensor,       # (B, 80, T_pred)
        target_gt_mel: torch.Tensor,  # (B, 80, T_tgt)  — rolled TARGET speaker gt mel
        pred_lengths: torch.Tensor,   # (B,) real frames in pred_mel
        tgt_lengths: torch.Tensor,    # (B,) real frames in target_gt_mel
        src_gt_mel: torch.Tensor = None,  # (B, 80, T_src) optional: source mel for diagnostic
        src_lengths: torch.Tensor = None,
    ) -> torch.Tensor:
        pred_stats = self._masked_stats(pred_mel, pred_lengths)
        tgt_stats  = self._masked_stats(target_gt_mel, tgt_lengths)

        loss = 0.0
        for band in ("low", "mid", "high"):
            p_mean, p_std = pred_stats[band]
            t_mean, t_std = tgt_stats[band]
            loss = loss + F.l1_loss(p_mean, t_mean) + F.l1_loss(p_std, t_std)

        # --- Diagnostic: D(pred,tgt) - D(pred,src) ---
        # Should trend negative: pred moving towards target, away from source
        if src_gt_mel is not None and src_lengths is not None:
            src_stats = self._masked_stats(src_gt_mel, src_lengths)
            d_pred_tgt = 0.0
            d_pred_src = 0.0
            for band in ("low", "mid", "high"):
                p_mean, p_std = pred_stats[band]
                t_mean, t_std = tgt_stats[band]
                s_mean, s_std = src_stats[band]
                d_pred_tgt += (F.l1_loss(p_mean, t_mean) + F.l1_loss(p_std, t_std)).item()
                d_pred_src += (F.l1_loss(p_mean, s_mean) + F.l1_loss(p_std, s_std)).item()
            conversion_metric = d_pred_tgt - d_pred_src
            print(f"[BAND_STATS] D(pred,tgt)={d_pred_tgt:.4f} D(pred,src)={d_pred_src:.4f} "
                  f"metric={conversion_metric:.4f} ({'✅ towards target' if conversion_metric < 0 else '❌ towards source'})")

        return loss


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
        mel_encoder: Optional[nn.Module] = None,
        speaker_encoder: Optional[nn.Module] = None,   # kept for API compat, unused
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

        # 1. Mel-spectrogram L1 reconstruction loss
        self.mel_loss = MelSpectrogramLoss(weight_l1=0.7)

        # 2. Multi-resolution STFT loss (disabled in Colab path since pred_wave=None)
        self.stft_loss = MRSTFTLoss()

        # 3. Band-stats speaker loss — no trainable params, direct gradient path
        self.spk_emb_loss: Optional[BandStatsSpeakerLoss] = None
        if hasattr(cfg, 'lambda_spk') and cfg.lambda_spk > 0:
            self.spk_emb_loss = BandStatsSpeakerLoss()
        else:
            print("Warning: lambda_spk is 0 or unset. Speaker loss disabled.")


    def forward(
        self,
        pred_mel: torch.Tensor,              # (B, n_mel, T_mel)
        gt_mel: torch.Tensor,                # (B, n_mel, T_mel)  — source speaker gt mel
        pred_wave: torch.Tensor,             # (B, T_wav)  — None in Colab path
        gt_wave: torch.Tensor,               # (B, T_wav)  — None in Colab path
        gt_lengths: torch.Tensor = None,     # (B,) real mel frame counts for pred + src
        target_gt_mel: torch.Tensor = None,  # (B, n_mel, T_tgt) — TARGET speaker gt mel (rolled)
        target_gt_lengths: torch.Tensor = None, # (B,) real frame counts for target_gt_mel
        is_cross_speaker: bool = False,      # True → only speaker loss, no mel L1
    ) -> LossOutputs:
        outs = LossOutputs()

        # --- Mel L1 loss: ONLY on same-speaker (self-recon) batches ---
        if not is_cross_speaker:
            if hasattr(self.cfg, 'lambda_mel') and self.cfg.lambda_mel > 0:
                try:
                    l_mel = self.mel_loss(pred_mel, gt_mel, lengths=gt_lengths)
                    outs["mel"] = self.cfg.lambda_mel * l_mel
                except Exception as exc:
                    raise RuntimeError(f"Mel loss failed: {exc}") from exc

        # --- STFT loss: disabled when pred_wave is None ---
        if hasattr(self.cfg, 'lambda_rec') and self.cfg.lambda_rec > 0 and pred_wave is not None and gt_wave is not None:
            try:
                l_stft = self.stft_loss(pred_wave, gt_wave)
                outs["stft"] = self.cfg.lambda_rec * l_stft
            except Exception as exc:
                raise RuntimeError(f"STFT loss failed: {exc}") from exc

        # --- Band-stats speaker loss: ONLY on cross-speaker batches ---
        # Compares per-band (low/mid/high) mean and std of pred_mel
        # against the TARGET speaker's actual gt mel statistics (masked, no padding).
        if is_cross_speaker and target_gt_mel is not None and self.spk_emb_loss is not None:
            if hasattr(self.cfg, 'lambda_spk') and self.cfg.lambda_spk > 0:
                try:
                    # pred_lengths: use gt_lengths (same content length as prediction)
                    # src_gt_mel:   gt_mel is the SOURCE speaker mel — for diagnostic only
                    l_spk = self.spk_emb_loss(
                        pred_mel=pred_mel,
                        target_gt_mel=target_gt_mel,
                        pred_lengths=gt_lengths,
                        tgt_lengths=target_gt_lengths,
                        src_gt_mel=gt_mel,
                        src_lengths=gt_lengths,
                    )
                    outs["speaker"] = self.cfg.lambda_spk * l_spk
                except Exception as exc:
                    raise RuntimeError(f"Speaker loss failed: {exc}") from exc

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
    gt_wave = torch.randn_like(pred_wav) * 0.9

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

