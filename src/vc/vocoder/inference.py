# inference.py
"""
CasiVC 2.0 Voice Conversion & Inference Engine
=============================================

Production-ready zero-shot voice conversion pipeline.
Extracts semantic content from a source utterance and applies the speaker timbre
from a reference utterance to generate converted speech.

Usage:
  python inference.py --source input.wav --target target_speaker.wav --output converted.wav
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Optional, Union, Tuple

import torch
import soundfile as sf
from torch import nn

# Ensure cross-platform compat and symlink support on Windows
import vc.data.compat
from vc.config import AudioConfig, ModelConfig, TrainingConfig
from vc.data.audio_utils import load_audio, extract_mel_spectrogram
from vc.model import HubertVCModel

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


class HiFiGANVocoder(nn.Module):
    """
    Wrapper for SpeechBrain's 16kHz HiFi-GAN neural vocoder.
    """
    def __init__(self, device: str = "cpu", trainable: bool = False):
        super().__init__()
        self.device = str(device)
        self.trainable = trainable

        print(f"[HiFiGAN] Loading SpeechBrain 16kHz Vocoder on {self.device}...")
        try:
            from speechbrain.inference.vocoders import HIFIGAN

            self.vocoder = HIFIGAN.from_hparams(
                source="speechbrain/tts-hifigan-libritts-16kHz",
                savedir="pretrained_models/hifigan-16k",
                run_opts={"device": self.device}
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load SpeechBrain HiFiGAN: {e}")

        if not trainable:
            self.vocoder.eval()
            for p in self.vocoder.parameters():
                p.requires_grad = False

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mel: (B, 80, T) log-mel spectrogram
        Returns:
            waveform: (B, T_wav) float32
        """
        if mel.device != torch.device(self.device):
            mel = mel.to(self.device)

        wav = self.vocoder.mods.generator(mel)
        if wav.dim() == 3:
            wav = wav.squeeze(1)
        return wav


def load_vocoder(checkpoint_path: Optional[str] = None, device: str = "cpu", trainable: bool = False) -> HiFiGANVocoder:
    """Loads 16kHz SpeechBrain HiFi-GAN vocoder."""
    return HiFiGANVocoder(device=device, trainable=trainable)


def load_casivc_model(
    checkpoint_path: str = "checkpoints/best.ckpt",
    device: str = "cpu"
) -> HubertVCModel:
    """Instantiates the CasiVC sub-500k model and loads weights from checkpoint."""
    a_cfg = AudioConfig()
    m_cfg = ModelConfig()
    t_cfg = TrainingConfig()

    model = HubertVCModel(a_cfg, m_cfg, t_cfg).to(device)
    model.eval()

    if checkpoint_path and os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        state_dict = ckpt.get("model_state", ckpt)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"[CasiVC] Loaded checkpoint: {checkpoint_path} (Missing: {len(missing)}, Unexpected: {len(unexpected)})")
    else:
        print(f"[CasiVC] Warning: Checkpoint '{checkpoint_path}' not found. Initialized with fresh weights.")

    return model


def convert_voice(
    source_audio_path: str,
    target_speaker_path: str,
    output_audio_path: Optional[str] = None,
    checkpoint_path: str = "checkpoints/best.ckpt",
    model: Optional[HubertVCModel] = None,
    vocoder: Optional[HiFiGANVocoder] = None,
    device: Optional[str] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Performs end-to-end zero-shot voice conversion.

    Args:
        source_audio_path: Path to content/source audio (.wav)
        target_speaker_path: Path to target timbre reference audio (.wav)
        output_audio_path: Optional output path to save converted .wav
        checkpoint_path: Path to trained CasiVC checkpoint
        model: Optional pre-loaded HubertVCModel
        vocoder: Optional pre-loaded HiFiGANVocoder
        device: 'cuda' or 'cpu' (auto-detected if None)

    Returns:
        converted_waveform: (T_wav,) float tensor
        predicted_mel: (80, T_mel) float tensor
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    a_cfg = AudioConfig()

    # 1. Load Subsystems
    if model is None:
        model = load_casivc_model(checkpoint_path, device=device)
    model.eval()

    if vocoder is None:
        vocoder = load_vocoder(None, device=device)
    vocoder.eval()

    # 2. Load and preprocess audio
    src_wav = load_audio(source_audio_path, sample_rate=a_cfg.sample_rate).to(device)
    tgt_wav = load_audio(target_speaker_path, sample_rate=a_cfg.sample_rate).to(device)

    print(f"\n[Conversion] Source: {source_audio_path} ({src_wav.shape[0] / a_cfg.sample_rate:.2f}s)")
    print(f"[Conversion] Target: {target_speaker_path} ({tgt_wav.shape[0] / a_cfg.sample_rate:.2f}s)")

    # 3. Model Forward Pass
    with torch.no_grad():
        pred_mel, _, aux = model(
            ref_audio=tgt_wav.unsqueeze(0),
            content_audio=[src_wav],
            return_aux=True
        )

        # 4. Neural Vocoding
        pred_mel = pred_mel.to(device)
        converted_wav = vocoder(pred_mel).squeeze(0).cpu()

    # 5. Save Output
    if output_audio_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_audio_path)), exist_ok=True)
        sf.write(output_audio_path, converted_wav.numpy(), a_cfg.sample_rate)
        print(f"[PASS] Converted audio saved to: {output_audio_path}\n")

    return converted_wav, pred_mel.squeeze(0).cpu()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Zero-Shot Voice Conversion Inference with CasiVC 2.0")
    parser.add_argument("--source", type=str, required=True, help="Path to source audio file (.wav)")
    parser.add_argument("--target", type=str, required=True, help="Path to target speaker reference audio (.wav)")
    parser.add_argument("--output", type=str, default="converted.wav", help="Path to save converted audio (.wav)")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best.ckpt", help="Path to model checkpoint")
    parser.add_argument("--device", type=str, default=None, help="Device ('cuda' or 'cpu')")
    args = parser.parse_args()

    # Fallback to last or quick_test checkpoint if best does not exist yet
    ckpt = args.checkpoint
    if not os.path.exists(ckpt):
        for candidate in ["checkpoints/best.ckpt", "checkpoints/last.ckpt", "checkpoints/quick_test.ckpt"]:
            if os.path.exists(candidate):
                ckpt = candidate
                break

    convert_voice(
        source_audio_path=args.source,
        target_speaker_path=args.target,
        output_audio_path=args.output,
        checkpoint_path=ckpt,
        device=args.device
    )
    os._exit(0)