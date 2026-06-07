# hubertvc/models/hubertvc_model.py
"""
HubertVCModel
=============

End-to-end generator that glues together

    • HuBERTEncoder                 (frozen – extracts content features)
    • MelEncoder                    (learns speaker tokens)
    • PositionAgnosticCrossAttention(fuses content + speaker)
    • MobileNetDecoder              (predicts 80-bin / 10 ms log-mels)

It produces a mel-spectrogram ready for a **frozen** HiFi-GAN
vocoder (loaded outside this class).  
The model itself contains *no* vocoder or discriminators so it
can be exported / fine-tuned independently.

All hyper-parameters are pulled from `hubertvc.config`, nothing
is hard-coded.
"""

from __future__ import annotations
from typing import List, Dict, Any, Optional, Tuple
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

# ------------------------------------------------------------------ #
#  Project modules                                                   #
# ------------------------------------------------------------------ #
from config import AudioConfig, ModelConfig, TrainingConfig
from data.audio_utils import extract_mel_spectrogram
from models.hubert_encoder import HuBERTEncoder
from models.mel_encoder import MelEncoder
from models.cross_attention import PositionAgnosticCrossAttention
from models.mobilenet_decoder import MobileNetDecoder

a_cfg = AudioConfig()
m_cfg = ModelConfig()
t_cfg = TrainingConfig()

class TemporalResampler(nn.Module):
    def __init__(self, channels=96):
        super().__init__()
        self.channels = channels
        self.default_ratio = 1.25  # Theoretical 20ms→16ms ratio
        
        # No-normalization residual refinement: y = interp(x) + 0.1·conv(interp(x))
        # Preserves FiLM modulation magnitude while smoothing 20ms→16ms.
        self.refine_conv = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        nn.init.xavier_uniform_(self.refine_conv.weight, gain=0.01)  # tiny init
        nn.init.zeros_(self.refine_conv.bias)
    
    def forward(self, x: torch.Tensor, target_length: Optional[int] = None) -> torch.Tensor:
        """
        Args:
            x: (T, C) or (B, T, C)
            target_length: Exact output length (use during training for alignment)
                          If None, uses default 1.25x ratio (inference)
        """
        input_is_2d = (x.dim() == 2)
        if input_is_2d:
            x = x.unsqueeze(0)
        
        B, T, C = x.shape
        assert C == self.channels, f"Expected {self.channels}D, got {C}D"
        
        # Training: match exact target | Inference: use ratio
        if target_length is not None:
            out_length = target_length
        else:
            out_length = round(T * self.default_ratio)
        
        x = x.transpose(1, 2)  # (B, T, C) → (B, C, T)
        x = F.interpolate(x, size=out_length, mode='linear', align_corners=False)
        x = x + 0.1 * self.refine_conv(x)  # residual: preserves scale, adds smoothness
        x = x.transpose(1, 2)  # (B, C, T) → (B, T, C)
        
        if input_is_2d:
            x = x.squeeze(0)
        
        return x



# ------------------------------------------------------------------ #
#  HubertVCModel                                                     #
# ------------------------------------------------------------------ #
class HubertVCModel(nn.Module):
    """
    The generator used for training **and** conversion.
    """

    def __init__(
        self,
        audio_cfg: AudioConfig | None = None,
        model_cfg: ModelConfig | None = None,
        training_cfg: TrainingConfig | None = None,
        num_speakers: Optional[int] = None,
    ):
        super().__init__()
        self.a_cfg = audio_cfg or a_cfg
        self.m_cfg = model_cfg or m_cfg
        self.t_cfg = training_cfg or t_cfg

        # 1. Sub-modules
        self.hubert = HuBERTEncoder(
            model_name=self.m_cfg.hubert_model_name,
            cache_dir=self.m_cfg.hubert_cache_dir,
            max_audio_length=self.m_cfg.max_audio_length,
            enable_caching=False,
        )
        self.hubert_encoder = self.hubert
        self.hubert.eval()                        # frozen
        for p in self.hubert.parameters():
            p.requires_grad = False

        #self.mel_encoder = MelEncoder()

        self.mel_encoder = MelEncoder()

        self.cross_attn = PositionAgnosticCrossAttention(enable_residual=True)

        # Content projection: HuBERT layer 9 (768D) → 3-layer MLP → 96D
        self.hubert_proj = nn.Sequential(
            nn.Linear(768, 384),
            nn.LayerNorm(384),
            nn.GELU(),
            nn.Linear(384, 192),
            nn.LayerNorm(192),
            nn.GELU(),
            nn.Linear(192, 96),
            nn.LayerNorm(96)
        )

        # Speaker projection: HuBERT layer 1 (768D) → 3-layer MLP → 96D
        # Layer 1 preserves acoustic speaker identity (F0, formant structure)
        # before semantic abstraction. Time-varying tokens enable per-frame
        # timbre modulation rather than static speaker embedding.
        # Same capacity as content projection — 3 layers with LayerNorm
        # gives the model enough representational power to separate speaker
        # identity from generic acoustic features in HuBERT L1.
        self.speaker_hubert_proj = nn.Sequential(
            nn.Linear(768, 384),
            nn.LayerNorm(384),
            nn.GELU(),
            nn.Linear(384, 192),
            nn.LayerNorm(192),
            nn.GELU(),
            nn.Linear(192, 96),
        )

        self.temporal_resampler = TemporalResampler(channels=96)
        self._verbose = True  # set False to suppress per-call debug prints

        # --------------------------------------------------------- #
        # CONTINUOUS INFORMATION BOTTLENECK (IB)                    #
        # --------------------------------------------------------- #
        # Since online VQ suffered from Codebook Collapse on random
        # initialization, we use a continuous bottleneck instead. 
        # Squeezing 96D to 32D strips high-level speaker identity while
        # preserving enough capacity for spectral texture (harmonics, formant
        # bandwidth, fricative noise) that 12D choked off.
        self.ib_dim = 32
        self.info_bottleneck = nn.Sequential(
            nn.Linear(self.m_cfg.cross_attention_dim, self.ib_dim),
            nn.LayerNorm(self.ib_dim),
            nn.GELU(),  # Replaced Tanh to prevent hard saturation and information loss
            nn.Linear(self.ib_dim, self.m_cfg.cross_attention_dim),
            nn.LayerNorm(self.m_cfg.cross_attention_dim)
        )

        self.decoder = MobileNetDecoder(config=m_cfg, return_feats=False)    

        # Speaker classifier head - per-frame Conv1d at decoder bottleneck
        if num_speakers is not None and num_speakers > 0:
            self.speaker_classifier = nn.Conv1d(192, num_speakers, kernel_size=1)
            nn.init.xavier_uniform_(self.speaker_classifier.weight)
            nn.init.zeros_(self.speaker_classifier.bias)
            print(f"[HubertVCModel] Speaker classifier: Conv1d(192, {num_speakers}, k=1) → {num_speakers * 193:,} params")

            # Block3 classifier: directly supervises block3_sum (96ch, pre-spk_film)
            # to teach FiLM MLP which channels encode target-speaker identity
            self.block3_classifier = nn.Conv1d(96, num_speakers, kernel_size=1)
            nn.init.xavier_uniform_(self.block3_classifier.weight)
            nn.init.zeros_(self.block3_classifier.bias)
            print(f"[HubertVCModel] Block3 classifier: Conv1d(96, {num_speakers}, k=1) → {num_speakers * 97:,} params")

            # Mel-output classifier: supervises mel_proj to retain speaker info
            self.mel_classifier = nn.Conv1d(80, num_speakers, kernel_size=1)
            nn.init.xavier_uniform_(self.mel_classifier.weight)
            nn.init.zeros_(self.mel_classifier.bias)
            print(f"[HubertVCModel] Mel classifier: Conv1d(80, {num_speakers}, k=1) → {num_speakers * 81:,} params")

            # Pooled mel-bias classifier: time-pooled → Linear(80, num_speakers) → CE
            # Operates on post-bias, pre-clamp mel to directly train MelSpeakerAffine.
            # Pooled (not per-frame) to match time-invariant bias structure.
            self.pooled_mel_classifier = nn.Linear(80, num_speakers)
            nn.init.xavier_uniform_(self.pooled_mel_classifier.weight, gain=0.2)
            nn.init.zeros_(self.pooled_mel_classifier.bias)
            print(f"[HubertVCModel] Pooled mel classifier: Linear(80, {num_speakers}) → {num_speakers * 81:,} params")

            # Spk_film classifier: supervises post-mel_proj features (80-dim mel)
            # to force mel_proj to preserve speaker information.
            # Previous 96-dim placement (pre-mel_proj) failed because the classifier
            # read pre-existing speaker signal from b3_id_film upstream as a shortcut,
            # leaving speaker_film near-identity. Post-mel_proj placement eliminates
            # the shortcut because mel_proj erases speaker info — the classifier MUST
            # push mel_proj to preserve it. Same classifier_weight=0.3 as block2/3.
            self.spk_film_classifier = nn.Conv1d(80, num_speakers, kernel_size=1)
            nn.init.xavier_uniform_(self.spk_film_classifier.weight)
            nn.init.zeros_(self.spk_film_classifier.bias)
            print(f"[HubertVCModel] Spk_film classifier: Conv1d(80, {num_speakers}, k=1) → {num_speakers * 81:,} params")
        else:
            self.speaker_classifier = None
            self.block3_classifier = None
            self.mel_classifier = None
            self.pooled_mel_classifier = None
            self.spk_film_classifier = None

        # 2. Sanity-check overall parameter budget
        trainables = sum(p.numel() for p in self.parameters()
                         if p.requires_grad)
        print(f"[HubertVCModel] trainable params: {trainables/1e3:,.1f} k")

    # ============================================================= #
    #  Helpers                                                      #
    # ============================================================= #

    def _log_parameter_counts(self) -> None:
        """
        Prints how many trainable parameters sit in every immediate
        sub-module of HubertVCModel.
        """
        print("\n[HubertVCModel] trainable parameters per sub-module")
        for name, module in self.named_children():          # <— PyTorch util[1]
            count = sum(p.numel() for p in module.parameters() if p.requires_grad)
            print(f"   • {name:<18s}: {count:,}")
        print()  # extra newline for readability


    @staticmethod
    def _pad_features(seqs: List[torch.Tensor],
                      pad_value: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Pads a list of (T_i, D) tensors to (B, T_max, D) and
        returns a Bool mask (B, T_max) where *True* marks real frames.
        """
        lengths = torch.tensor([s.size(0) for s in seqs], device=seqs[0].device)
        batch = pad_sequence(seqs, batch_first=True, padding_value=pad_value)
        mask = torch.arange(batch.size(1), device=batch.device)[None, :] < lengths[:, None]
        return batch, mask

    def _make_target_mel(self,
                         wav_or_mel: torch.Tensor,
                         sample_rate: int | None = None) -> torch.Tensor:
        """
        Accepts either a raw waveform (1-D Tensor) or a pre-computed
        mel (2-D Tensor).  Always returns a *log-mel* at 10 ms hop. 
        """
        if wav_or_mel.dim() == 1:   # raw audio
            return extract_mel_spectrogram(
                wav_or_mel,
                samplerate=sample_rate or self.a_cfg.sample_rate,
                frameshift_ms=self.a_cfg.frame_shift_reference,
                normalize=False,
            )
        elif wav_or_mel.dim() == 2:  # already mel
            return wav_or_mel
        raise ValueError("Unsupported target-mel input shape.")

    # ============================================================= #
    #  Public API                                                   #
    # ============================================================= #
    def forward(
    self,
    ref_audio: List[torch.Tensor] | torch.Tensor | None = None,
    content_audio: List[torch.Tensor] | None = None,
    gt_mels: Optional[List[torch.Tensor] | torch.Tensor] = None,
    compute_losses: bool = False,
    return_aux: bool = False,
    return_bottleneck: bool = False,
    precomputed_speaker_feats: Optional[torch.Tensor] = None,
    precomputed_content_feats: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor] | None, Dict[str, Any] | None]:
        """
        Parameters
        ----------
        ref_audio     : list or batch tensor of reference raw audio waveforms (samples,)
                        for HuBERT L1 speaker-token extraction
        content_audio : list of raw 16-kHz waveforms (variable length) for HuBERT
        gt_mels       : optional ground-truth mels for reconstruction loss
        compute_losses: if True, returns {"mel": …}
        return_aux    : if True, returns dict with intermediate tensors
        precomputed_speaker_feats: Optional cached HuBERT L1 [B, T_ref, 768]
                                   or already-projected speaker tokens [B, T_ref, 96]
        precomputed_content_feats: Optional precomputed HuBERT features [B, T, 96]

        Returns
        -------
        pred_mel  : (B, 80, T_out)  – predicted log-mel (10 ms hop)
        loss_dict : dict[str, tensor] or None
        aux       : dict[str, Any]   or None
        """
        device = next(self.parameters()).device
        
        # --------------------------------------------------------- #
        # 1. Speaker-token extraction (HuBERT layer 1 from ref audio) #
        # --------------------------------------------------------- #
        # Layer 1 provides per-frame acoustic speaker features (F0,
        # formant structure) — time-varying tokens instead of a single
        # pooled embedding. Enables fine-grained timbre modulation in
        # downstream cross-attention and FiLM.
        if precomputed_speaker_feats is not None:
            speaker_feats = precomputed_speaker_feats
            B = speaker_feats.shape[0]
            if speaker_feats.shape[-1] == self.m_cfg.hubert_features_dim:
                speaker_feats = self.speaker_hubert_proj(speaker_feats)
            elif speaker_feats.shape[-1] != self.m_cfg.speaker_projection_dim:
                raise ValueError(
                    f"precomputed_speaker_feats must end in "
                    f"{self.m_cfg.hubert_features_dim} or {self.m_cfg.speaker_projection_dim}, "
                    f"got {speaker_feats.shape[-1]}"
                )
        else:
            if ref_audio is None:
                raise ValueError("Either ref_audio or precomputed_speaker_feats must be provided")
            if isinstance(ref_audio, torch.Tensor) and ref_audio.dim() == 2:
                ref_audio = [ref_audio[i] for i in range(ref_audio.shape[0])]
            if not isinstance(ref_audio, (list, tuple)):
                ref_audio = [ref_audio]
            B = len(ref_audio)
            
            with torch.no_grad():
                speaker_feats = self.hubert.extract_speaker_layer(ref_audio)  # [B, T_ref, 768]
            speaker_feats = self.speaker_hubert_proj(speaker_feats)            # [B, T_ref, 96]
            
            if self._verbose:
                print(f"[HuBERT L1] Speaker feats shape: {speaker_feats.shape}")
                print(f"Speaker feats (96D): μ={speaker_feats.mean():.4f}, σ={speaker_feats.std():.4f}")

        # --------------------------------------------------------- #
        # 2. HuBERT content features                                #
        # --------------------------------------------------------- #
        if precomputed_content_feats is not None:
            content_feats = precomputed_content_feats
        else:
            if content_audio is None:
                raise ValueError("Either content_audio or precomputed_content_feats must be provided")
            if not isinstance(content_audio, (list, tuple)):
                raise ValueError("content_audio must be a list[Tensor]")
            if len(content_audio) != B:
                raise ValueError(f"Batch mismatch: ref_audio has {B} samples, content_audio has {len(content_audio)}")

            with torch.no_grad():
                content_feats = self.hubert(content_audio)    # [B, T, 768] (assuming padded)

            content_feats = self.hubert_proj(content_feats)  # [B, T, 768] → [B, T, 96]
            
        # ========================================================= #
        # CONTINUOUS INFORMATION BOTTLENECK                         #
        # ========================================================= #
        # Force the features through a 12-dimensional chokepoint. 
        # This violently strips out the background identity while maintaining
        # gradient flow and avoiding discrete codebook collapse.
        content_feats = self.info_bottleneck(content_feats)
        
        # REMOVED F.instance_norm (Fixes Whispering)
        # Normalizing over the temporal dimension completely destroyed the F0/pitch 
        # contour encoded in the variance of HuBERT features, causing the model to minimize
        # L1 loss by outputting a flat, unvoiced spectrum (whispering).
        # content_feats = content_feats.transpose(1, 2)  # [B, 96, T]
        # content_feats = F.instance_norm(content_feats)
        # content_feats = content_feats.transpose(1, 2)  # [B, T, 96]

        # --------------------------------------------------------- #
        # 3. Cross-attention fusion                                 #
        # --------------------------------------------------------- #
        
        # Process entire batch at once - each item uses its OWN speaker conditioning
        # content_feats: [B, T, 96], speaker_feats: [B, 64, 96]
        fused_features = self.cross_attn(
            content_feats,  # [B, T, 96]
            speaker_feats   # [B, 64, 96]
        )
        # fused_features: [B, T, 96]

        # Decoder runs at Mel frame rate (proper 1.25x interpolation via Conv1D)
        if gt_mels is not None:
            target_len = gt_mels.shape[-1] if isinstance(gt_mels, torch.Tensor) else gt_mels[0].shape[-1]
            resampled_features = self.temporal_resampler(fused_features, target_length=target_len)
        else:
            resampled_features = self.temporal_resampler(fused_features, target_length=None)
        if self._verbose:
            print(f"[DEBUG] Decoder input shape (Mel rate): {resampled_features.shape}")

            print(f"[DEBUG] Cross_Attention_Output shape: {fused_features.shape}, "
                  f"mean: {fused_features.mean().item():.4f}, "
                  f"std: {fused_features.std().item():.4f}")

        # --------------------------------------------------------- #
        # 4. Mel decoding                                           #
        # --------------------------------------------------------- #
        # Force FP32 for decoder to avoid NaN gradients
        with torch.amp.autocast(device_type=device.type, enabled=False):
            if return_bottleneck and (self.speaker_classifier is not None or self.pooled_mel_classifier is not None or self.spk_film_classifier is not None):
                pred_mel, intermediate = self.decoder(resampled_features.float(), return_intermediate=True, speaker_feats=speaker_feats)
            else:
                pred_mel = self.decoder(resampled_features.float(), speaker_feats=speaker_feats)  # [B, 80, T_hubert]

            # Upsample output from HuBERT rate → mel rate (deterministic, clean interpolation)
            if gt_mels is not None:
                if isinstance(gt_mels, torch.Tensor):
                    target_mel_frames = gt_mels.shape[-1]
                else:
                    target_mel_frames = gt_mels[0].shape[-1]
            else:
                target_mel_frames = pred_mel.shape[-1]  # Do not scale again

            pred_mel = torch.nn.functional.interpolate(
                pred_mel, size=target_mel_frames, mode='linear', align_corners=False
            )

            """
            print("FUSED INPUT  ➜  mean", fused_batch.mean().item(),
            "std", fused_batch.std().item(),
            "min", fused_batch.min().item(),
            "max", fused_batch.max().item(),
            "NaN?", torch.isnan(fused_batch).any().item(),
            "Inf?", torch.isinf(fused_batch).any().item())
            """
            

            if self._verbose:
                print(f"[DEBUG] MobileNet_Decoder_Output shape: {pred_mel.shape}, mean: {pred_mel.mean().item():.4f}, std: {pred_mel.std().item():.4f}")

        # --------------------------------------------------------- #
        # 5. Losses (optional)                                      #
        # --------------------------------------------------------- #
        loss_dict: Dict[str, torch.Tensor] | None = None
        if compute_losses:
            if gt_mels is None:
                raise ValueError("gt_mels must be provided when compute_losses=True")

            if isinstance(gt_mels, torch.Tensor):
                gt_mels = list(gt_mels)

            target_list = [self._make_target_mel(m) for m in gt_mels]
            tgt_batch, _ = self._pad_features(
                [t.transpose(0, 1) for t in target_list])      # (B, T', 80)
            tgt_batch = tgt_batch.transpose(1, 2)             # (B, 80, T')
            tgt_batch = tgt_batch.to(pred_mel.device)     # <— add this

            # Ensure same length (truncate both to minimum length)
            minlen = min(pred_mel.shape[-1], tgt_batch.shape[-1])
            pred_aligned = pred_mel[:, :, :minlen]
            tgt_aligned = tgt_batch[:, :, :minlen]

            l1_mel = F.l1_loss(pred_aligned, tgt_aligned) 
            #mse_mel = F.mse_loss(pred_aligned, tgt_aligned)

            loss_dict = {
                "mel": self.t_cfg.lambda_mel * l1_mel #!!!!!!!!!!!!
            }


        # --------------------------------------------------------- #
        # 6. Aux-dump (optional)                                    #
        # --------------------------------------------------------- #
        aux: Dict[str, Any] | None = None
        if return_aux:
            aux = {
              "speaker_tokens": speaker_feats,
              "content_length": fused_features.size(0),
          }

        # Classifier head: per-frame speaker ID logits from decoder bottleneck
        if return_bottleneck and self.speaker_classifier is not None:
            bottleneck = intermediate[2]  # Block 2 output: [B, 192, T]
            # Temporal smooth before classification: 3-frame AvgPool (~30ms)
            # prevents per-frame CE gradient from injecting high-frequency jitter
            bottleneck = F.avg_pool1d(bottleneck, kernel_size=3, stride=1, padding=1)
            classifier_logits = self.speaker_classifier(bottleneck)  # [B, num_speakers, T]
            if aux is None:
                aux = {}
            aux["classifier_logits"] = classifier_logits
            aux["bottleneck"] = bottleneck

            # Block3 classifier: supervise block3_sum (pre-SpeakerFiLM) to teach
            # b3_id_film which channels carry target-speaker identity
            if self.block3_classifier is not None:
                block3_sum = intermediate[3]  # [B, 96, T] — after residual sum, before spk_film
                block3_sum = F.avg_pool1d(block3_sum, kernel_size=3, stride=1, padding=1)
                block3_logits = self.block3_classifier(block3_sum)  # [B, num_speakers, T]
                aux["block3_classifier_logits"] = block3_logits

            # Mel-output classifier: supervise mel_proj to retain speaker info
            if self.mel_classifier is not None:
                mel_feats = pred_mel  # [B, 80, T] — clamp saturation is 0%, so effectively unclamped
                mel_feats = F.avg_pool1d(mel_feats, kernel_size=3, stride=1, padding=1)
                mel_logits = self.mel_classifier(mel_feats)  # [B, num_speakers, T]
                aux["mel_classifier_logits"] = mel_logits

            # Pooled mel-bias classifier: gated CE on bias-only component
            # bias_only = postbias_mel - spk_film_mel isolates the affine delta.
            # Detach on spk_film_mel prevents CE gradient from leaking into
            # mel_proj / out_scale — gradient reaches ONLY mel_speaker_affine MLP.
            # intermediate: [-4]=prebias(raw), [-3]=variance(post-scale),
            #               [-2]=spk_film_mel, [-1]=postbias(post-affine)
            if self.pooled_mel_classifier is not None and len(intermediate) > 7:
                spk_film_mel_ce = intermediate[-2]   # [B, 80, T] pre-affine
                postbias_mel = intermediate[-1]       # [B, 80, T] post-affine
                bias_only = postbias_mel - spk_film_mel_ce  # delta from affine only
                mel_for_ce = spk_film_mel_ce.detach() + bias_only
                mel_pooled = mel_for_ce.mean(dim=-1)
                pooled_mel_logits = self.pooled_mel_classifier(mel_pooled)
                aux["pooled_mel_logits"] = pooled_mel_logits

            # Spk_film classifier: taps spk_film_mel (post-out_scale + speaker delta, pre-affine)
            # Now reads from the post-speaker-delta tensor so CE gradient trains
            # mel_proj_speaker directly (content path is detached upstream).
            if self.spk_film_classifier is not None and len(intermediate) > 7:
                spk_film_feats = intermediate[-2]  # [B, 80, T] spk_film_mel
                spk_film_feats = F.avg_pool1d(spk_film_feats, kernel_size=3, stride=1, padding=1)
                spk_film_logits = self.spk_film_classifier(spk_film_feats)
                aux["spk_film_classifier_logits"] = spk_film_logits

            # Three-path gradient separation: expose all intermediate mels
            #   prebias_mel   (raw, pre-scale)              → L1 target (mel_proj_content)
            #   variance_mel  (post-scale, pre-speaker-delta) → variance target (trains out_scale)
            #   spk_film_mel  (post-speaker-delta)           → CE target (trains mel_proj_speaker)
            #   postbias_mel  (post-scale, post-affine)      → speaker stats (detach protected)
            if len(intermediate) >= 8:
                aux["prebias_mel"]   = intermediate[-4]   # raw mel (content path) — L1 target
                aux["variance_mel"]  = intermediate[-3]   # post-scale, pre-speaker-delta — variance target
        
        return pred_mel, loss_dict, aux

    # ============================================================= #
    #  Convenience                                                  #
    # ============================================================= #
    def convert(
        self,
        ref_mel: torch.Tensor,
        src_audio: torch.Tensor,
        vocoder: Optional[nn.Module] = None,
        return_waveform: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Zero-shot conversion helper.
        Returns predicted mel and (optionally) waveform if
        a HiFi-GAN vocoder is supplied.
        """
        self.eval()
        with torch.inference_mode():
            pred_mel, _, _ = self.forward(
                ref_mels=[ref_mel.to(self.device)],
                content_audio=[src_audio.to(self.device)],
                compute_losses=False,
            )
            if return_waveform:
                if vocoder is None:
                    raise ValueError("vocoder must be provided "
                                     "when return_waveform=True")

                vocoder_input = (pred_mel * 2.0) - 4.1
                wav = vocoder(pred_mel).squeeze(1)  # vocoder expects (B,80,T)
                return pred_mel[0], wav[0]
        return pred_mel[0], None

    # ============================================================= #
    #  Utilities                                                    #
    # ============================================================= #
    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def freeze_submodule(self, name: str, freeze: bool = True) -> None:
        """
        Example:  model.freeze_submodule('mel_encoder', True)
        """
        mod = getattr(self, name, None)
        if mod is None:
            raise AttributeError(f"No such sub-module: {name}")
        for p in mod.parameters():
            p.requires_grad = not freeze

    def trainable_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ------------------------------------------------------------------ #
#  Stand-alone sanity test                                           #
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = HubertVCModel().to(device)

    # Dummy 1-s reference & 2-s source
    ref_mel = torch.randn(a_cfg.n_mel_bands, 100)          # 100 frames @10 ms
    src_wav = torch.randn(int(2 * a_cfg.sample_rate))      # 2-second audio

    pred, loss, aux = model(
        ref_mels=[ref_mel],
        content_audio=[src_wav],
        gt_mels=[ref_mel],                 # fake target
        compute_losses=True,
        return_aux=True,
    )

    print("Pred mel:", pred.shape)         # (1, 80, 200)
    if loss:
        print({k: v.item() for k, v in loss.items()})
    print("Aux keys:", list(aux.keys()))
