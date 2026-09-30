# hubertvc/models/hubertvc_model.py
"""
HubertVCModel: Sub-500k Parameter Zero-Shot Voice Conversion Generator
======================================================================

Orchestrates:
    • HuBERTEncoder                 (frozen – extracts continuous phonetic features)
    • MelEncoder                    (frozen ECAPA-TDNN + trainable projection to speaker tokens)
    • PositionAgnosticCrossAttention(fuses phonetic queries + speaker keys/values)
    • TemporalResampler             (aligns 50Hz semantic rate to target mel frame rate)
    • ConvNeXtDecoder               (direct mel synthesis with multi-scale FiLM conditioning)

All hyperparameters are loaded from `config.py`.
Trainable parameters strictly capped under 500,000 (~370k total).
"""

from __future__ import annotations
from typing import List, Dict, Any, Optional, Tuple
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from vc.config import AudioConfig, ModelConfig, TrainingConfig
from vc.data.audio_utils import extract_mel_spectrogram
from vc.content.hubert_encoder import HuBERTEncoder
from vc.speaker.mel_encoder import MelEncoder
from vc.fusion.cross_attention import PositionAgnosticCrossAttention, ReferenceFrameEncoder
from vc.decoder.convnext_decoder import ConvNeXtDecoder, TemporalResampler

a_cfg = AudioConfig()
m_cfg = ModelConfig()
t_cfg = TrainingConfig()


class HubertVCModel(nn.Module):
    """
    Sub-500k Zero-Shot Voice Conversion Generator.
    Directly synthesizes target speaker mel-spectrograms from disentangled
    phonetic features and target acoustic tokens.
    """

    def __init__(
        self,
        audio_cfg: AudioConfig | None = None,
        model_cfg: ModelConfig | None = None,
        training_cfg: TrainingConfig | None = None,
        num_speakers: Optional[int] = None,
        load_encoders: bool = False,
    ):
        super().__init__()
        self.a_cfg = audio_cfg or a_cfg
        self.m_cfg = model_cfg or m_cfg
        self.t_cfg = training_cfg or t_cfg

        # 1. Content Encoder: Frozen HuBERT (Base, 768-D) - Lazy-loaded
        self._hubert: Optional[HuBERTEncoder] = None
        if load_encoders:
            self._init_hubert()

        # Content projection & information bottleneck (768 -> 128 -> 64) ~107k params
        self.hubert_proj = nn.Sequential(
            nn.Linear(self.m_cfg.hubert_features_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, self.m_cfg.content_bottleneck_dim),
            nn.LayerNorm(self.m_cfg.content_bottleneck_dim),
        )

        # 2. Speaker Encoder: MelEncoder (ECAPA-TDNN + projection) - Lazy-loaded
        self.speaker_dim = getattr(self.m_cfg, "speaker_projection_dim", 64)
        self.num_speaker_tokens = getattr(self.m_cfg, "num_speaker_tokens", 4)
        self.ecapa_dim = 192  # Standard SpeechBrain ECAPA embedding dimension
        self._mel_encoder: Optional[MelEncoder] = None
        if load_encoders:
            self._init_mel_encoder()

        # Trainable projection from raw 192D ECAPA to tokens: 192 -> (num_tokens * speaker_dim) ~49.5k params
        self.speaker_proj = nn.Linear(
            self.ecapa_dim,
            self.num_speaker_tokens * self.speaker_dim
        )
        self.speaker_token_norm = nn.LayerNorm(self.speaker_dim)

        # Reference-frame timbre encoder (position-agnostic frame tokens)
        self.ref_frame_encoder = None
        if getattr(self.m_cfg, "use_ref_frame_encoder", True):
            self.ref_frame_encoder = ReferenceFrameEncoder(
                n_mels=self.a_cfg.n_mel_bands,
                dim=self.speaker_dim,
                stride=getattr(self.m_cfg, "ref_frame_stride", 4),
            )

        residual_rank = int(getattr(self.m_cfg, "residual_speaker_rank", 0))
        self.residual_speaker_proj = None
        if residual_rank > 0:
            self.residual_speaker_proj = nn.Sequential(
                nn.Linear(self.m_cfg.hubert_features_dim, residual_rank, bias=False),
                nn.GELU(),
                nn.Linear(residual_rank, self.speaker_dim, bias=False),
            )

        # 3. Cross-Attention Fusion (~16.7k params)
        self.cross_attn = PositionAgnosticCrossAttention(
            content_dim=self.m_cfg.content_bottleneck_dim,
            speaker_dim=self.speaker_dim,
            d_model=self.m_cfg.cross_attention_dim,
            num_heads=self.m_cfg.cross_attention_heads,
            dropout=self.m_cfg.cross_attention_dropout,
        )

        # 4. Temporal Resampler (50Hz HuBERT -> target mel rate) ~12.3k params
        self.temporal_resampler = TemporalResampler(
            channels=self.m_cfg.cross_attention_dim,
            default_ratio=1.25
        )

        # 5. Direct Mel Decoder (~200k params)
        decoder_channels = getattr(self.m_cfg, "decoder_channels", 96)
        num_blocks = getattr(self.m_cfg, "decoder_num_blocks", 4)
        self.decoder = ConvNeXtDecoder(
            in_channels=self.m_cfg.cross_attention_dim,
            decoder_channels=decoder_channels,
            speaker_dim=self.speaker_dim,
            out_channels=self.a_cfg.n_mel_bands,
            num_blocks=num_blocks,
            expansion=getattr(self.m_cfg, "decoder_expansion", 2),
            kernel_size=getattr(self.m_cfg, "decoder_kernel_size", 7),
            speaker_token_routing=getattr(self.m_cfg, "speaker_token_routing", False),
            num_speaker_tokens=self.num_speaker_tokens,
        )

        # 6. Audit parameter budget (Strictly enforce sub-500k constraint)
        self.trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"\n{'='*60}")
        print(f"[HubertVCModel] Initialized Sub-500k Model")
        print(f"  • Total Trainable Parameters: {self.trainable_params:,} ({self.trainable_params / 1e3:.1f}k)")
        print(f"  • HuBERT Projection:          {sum(p.numel() for p in self.hubert_proj.parameters() if p.requires_grad):,}")
        print(f"  • Speaker Projection:         {sum(p.numel() for p in self.speaker_proj.parameters() if p.requires_grad) + sum(p.numel() for p in self.speaker_token_norm.parameters() if p.requires_grad):,}")
        ref_params = (sum(p.numel() for p in self.ref_frame_encoder.parameters() if p.requires_grad)
                      if self.ref_frame_encoder is not None else 0)
        print(f"  • Ref Frame Encoder:          {ref_params:,}")
        residual_params = (sum(p.numel() for p in self.residual_speaker_proj.parameters() if p.requires_grad)
                           if self.residual_speaker_proj is not None else 0)
        print(f"  • Residual Speaker Adapter:   {residual_params:,}")
        print(f"  • Cross Attention:            {sum(p.numel() for p in self.cross_attn.parameters() if p.requires_grad):,}")
        print(f"  • Temporal Resampler:         {sum(p.numel() for p in self.temporal_resampler.parameters() if p.requires_grad):,}")
        print(f"  • ConvNeXt Mel Decoder:       {sum(p.numel() for p in self.decoder.parameters() if p.requires_grad):,}")
        print(f"{'='*60}\n")

        assert self.trainable_params < getattr(self.m_cfg, "max_trainable_params", 500_000), (
            f"Trainable parameter budget exceeded: {self.trainable_params:,} >= 500,000"
        )

    def _init_hubert(self):
        if self._hubert is None:
            self._hubert = HuBERTEncoder(
                model_name=self.m_cfg.hubert_model_name,
                cache_dir=self.m_cfg.hubert_cache_dir,
                max_audio_length=self.m_cfg.max_audio_length,
                enable_caching=False,
            )
            self._hubert.eval()
            for p in self._hubert.parameters():
                p.requires_grad = False
        return self._hubert

    @property
    def hubert(self) -> HuBERTEncoder:
        return self._init_hubert()

    def _init_mel_encoder(self):
        if self._mel_encoder is None:
            self._mel_encoder = MelEncoder(
                output_dim=self.speaker_dim,
                num_speaker_tokens=self.num_speaker_tokens
            )
            self._mel_encoder.eval()
            for p in self._mel_encoder.parameters():
                p.requires_grad = False
        return self._mel_encoder

    @property
    def mel_encoder(self) -> MelEncoder:
        return self._init_mel_encoder()

    # ============================================================= #
    #  Helpers                                                      #
    # ============================================================= #
    @staticmethod
    def _pad_features(seqs: List[torch.Tensor], pad_value: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor]:
        lengths = torch.tensor([s.size(0) for s in seqs], device=seqs[0].device)
        batch = pad_sequence(seqs, batch_first=True, padding_value=pad_value)
        mask = torch.arange(batch.size(1), device=batch.device)[None, :] < lengths[:, None]
        return batch, mask

    def _make_target_mel(self, wav_or_mel: torch.Tensor, sample_rate: int | None = None) -> torch.Tensor:
        if wav_or_mel.dim() == 1:
            return extract_mel_spectrogram(
                wav_or_mel,
                sample_rate=sample_rate or self.a_cfg.sample_rate,
                normalize=False,
            )
        elif wav_or_mel.dim() == 2:
            return wav_or_mel
        raise ValueError("Unsupported target-mel input shape.")

    # ============================================================= #
    #  Forward                                                      #
    # ============================================================= #
    def forward(
        self,
        ref_audio: List[torch.Tensor] | torch.Tensor | None = None,
        content_audio: List[torch.Tensor] | torch.Tensor | None = None,
        gt_mels: Optional[List[torch.Tensor] | torch.Tensor] = None,
        compute_losses: bool = False,
        return_aux: bool = False,
        precomputed_speaker_feats: Optional[torch.Tensor] = None,  # [B, 192] raw ECAPA
        precomputed_content_feats: Optional[torch.Tensor] = None,  # [B, T, 768] raw HuBERT
        precomputed_ref_mel: Optional[torch.Tensor] = None,        # [B, 80, T_ref] reference mel
        precomputed_speaker_residual: Optional[torch.Tensor] = None, # [B, 768] utterance-stable HuBERT residual
        frames_only: bool = False,                                  # use ref frames as sole timbre path
        target_lengths: Optional[torch.Tensor] = None,              # [B] per-sample true mel lengths
        content_lengths: Optional[torch.Tensor] = None,             # [B] per-sample true content lengths
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]], Optional[Dict[str, Any]]]:
        """
        Args:
            ref_audio: raw waveform(s) for speaker conditioning
            content_audio: raw waveform(s) for phonetic content
            gt_mels: ground truth mel spectrograms for loss computation
            compute_losses: whether to compute reconstruction losses
            return_aux: whether to return internal activations
            precomputed_speaker_feats: pre-extracted 192D ECAPA embeddings [B, 192]
            precomputed_content_feats: pre-extracted 768D HuBERT features [B, T, 768]
        Returns:
            pred_mel: [B, 80, T_out]
            loss_dict: dictionary of losses (if compute_losses=True)
            aux: auxiliary diagnostic tensors (if return_aux=True)
        """
        device = next(self.parameters()).device

        # --------------------------------------------------------- #
        # 1. Speaker Representations (Tokens)                       #
        # --------------------------------------------------------- #
        if precomputed_speaker_feats is not None:
            raw_spk = precomputed_speaker_feats.to(device)
            B = raw_spk.shape[0]
            spk_proj = self.speaker_proj(raw_spk)  # [B, num_tokens * speaker_dim]
            speaker_tokens = spk_proj.view(B, self.num_speaker_tokens, self.speaker_dim)
            speaker_tokens = self.speaker_token_norm(speaker_tokens)  # [B, 4, 64]
        else:
            if ref_audio is None:
                raise ValueError("Either ref_audio or precomputed_speaker_feats must be provided.")
            if isinstance(ref_audio, list):
                ref_audio = torch.stack(ref_audio).to(device)
            elif isinstance(ref_audio, torch.Tensor) and ref_audio.dim() == 1:
                ref_audio = ref_audio.unsqueeze(0).to(device)
            else:
                ref_audio = ref_audio.to(device)
            B = ref_audio.shape[0]

            with torch.no_grad():
                raw_spk = self.mel_encoder._safe_encode(ref_audio, torch.ones(B, device=device))  # [B, 192]
            spk_proj = self.speaker_proj(raw_spk)
            speaker_tokens = spk_proj.view(B, self.num_speaker_tokens, self.speaker_dim)
            speaker_tokens = self.speaker_token_norm(speaker_tokens)

        if precomputed_speaker_residual is not None:
            if self.residual_speaker_proj is None:
                raise ValueError("precomputed_speaker_residual supplied, but residual_speaker_rank is disabled")
            residual = self.residual_speaker_proj(precomputed_speaker_residual.to(device).float())
            # A shared residual offset keeps the four ECAPA-derived token roles
            # intact while providing the decoder with a target-side residual cue.
            speaker_tokens = speaker_tokens + residual.unsqueeze(1)

        # --------------------------------------------------------- #
        # 2. Content Features (HuBERT + Bottleneck)                 #
        # --------------------------------------------------------- #
        if precomputed_content_feats is not None:
            content_raw = precomputed_content_feats.to(device).float()
        else:
            if content_audio is None:
                raise ValueError("Either content_audio or precomputed_content_feats must be provided.")
            with torch.no_grad():
                content_raw = self.hubert(content_audio)  # [B, T, 768]
            content_raw = content_raw.to(device)

        # Project through information bottleneck: [B, T, 768] -> [B, T, 64]
        content_feats = self.hubert_proj(content_raw)

        # --------------------------------------------------------- #
        # 3. Cross-Attention Fusion                                 #
        # --------------------------------------------------------- #
        ref_frames = None
        if precomputed_ref_mel is not None:
            if self.ref_frame_encoder is None:
                raise ValueError("precomputed_ref_mel supplied, but the reference-frame encoder is disabled")
            ref_mel = precomputed_ref_mel.to(device)
            if ref_mel.dim() == 2:
                ref_mel = ref_mel.unsqueeze(0)
            ref_frames = self.ref_frame_encoder(ref_mel)

        frames_active = ref_frames is not None
        attn_speaker = None if (frames_only and frames_active) else speaker_tokens
        decoder_speaker = ref_frames.mean(dim=1) if (frames_only and frames_active) else speaker_tokens

        if return_aux:
            fused_features, attn_weights = self.cross_attn(
                content_feats, attn_speaker, ref_frames=ref_frames, return_attention=True
            )
        else:
            fused_features = self.cross_attn(content_feats, attn_speaker, ref_frames=ref_frames)
            attn_weights = None

        # --------------------------------------------------------- #
        # 4. Temporal Resampling & Mel Decoding                     #
        # --------------------------------------------------------- #
        target_len = None
        if target_lengths is None and gt_mels is not None:
            if isinstance(gt_mels, torch.Tensor):
                target_len = gt_mels.shape[-1]
            elif isinstance(gt_mels, list) and len(gt_mels) > 0:
                target_len = gt_mels[0].shape[-1]

        resampled_features = self.temporal_resampler(
            fused_features, target_length=target_len, target_lengths=target_lengths,
            source_lengths=content_lengths
        )

        # ConvNeXt Mel Decoder: [B, T_resampled, 64] -> [B, 80, T_mel]
        pred_mel = self.decoder(resampled_features, speaker_emb=decoder_speaker)

        # Ensure exact frame match if target_len is specified (legacy shared-length path)
        if target_lengths is None and target_len is not None and pred_mel.shape[-1] != target_len:
            pred_mel = F.interpolate(pred_mel, size=target_len, mode='linear', align_corners=False)

        # --------------------------------------------------------- #
        # 5. Losses                                                 #
        # --------------------------------------------------------- #
        loss_dict: Optional[Dict[str, torch.Tensor]] = None
        if compute_losses:
            if gt_mels is None:
                raise ValueError("gt_mels required when compute_losses=True")
            if isinstance(gt_mels, list):
                tgt_batch, _ = self._pad_features([self._make_target_mel(m).transpose(0, 1) for m in gt_mels])
                tgt_batch = tgt_batch.transpose(1, 2).to(pred_mel.device)
            else:
                tgt_batch = gt_mels.to(pred_mel.device)

            min_len = min(pred_mel.shape[-1], tgt_batch.shape[-1])
            p_aligned = pred_mel[..., :min_len]
            t_aligned = tgt_batch[..., :min_len]

            # Mel L1 Reconstruction Loss
            l1_loss = F.l1_loss(p_aligned, t_aligned)

            # Spectral Convergence Loss on Mel
            spec_conv = torch.norm(t_aligned - p_aligned, p="fro") / (torch.norm(t_aligned, p="fro") + 1e-6)

            total_mel_loss = self.t_cfg.lambda_mel * l1_loss + 2.0 * spec_conv

            loss_dict = {
                "mel": total_mel_loss,
                "mel_l1": l1_loss,
                "spec_conv": spec_conv,
            }

        # --------------------------------------------------------- #
        # 6. Aux Diagnostics                                        #
        # --------------------------------------------------------- #
        aux: Optional[Dict[str, Any]] = None
        if return_aux:
            aux = {
                "content_feats": content_feats,
                "speaker_tokens": speaker_tokens,
                "ref_frames": ref_frames,
                "fused_features": fused_features,
                "resampled_features": resampled_features,
                "attention_weights": attn_weights,
            }

        return pred_mel, loss_dict, aux
