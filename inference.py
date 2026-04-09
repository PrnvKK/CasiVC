# hubertvc/inference.py

import torch
from torch import nn
import os
import sys
import torchaudio

# --- MONKEYPATCH START ---
# This restores the missing function that old SpeechBrain versions look for
if not hasattr(torchaudio, "list_audio_backends"):
    # We just tell it that 'soundfile' is available (standard for Linux/Colab)
    torchaudio.list_audio_backends = lambda: ["soundfile"]
# --- MONKEYPATCH END ---


class HiFiGANVocoder(nn.Module):
    """
    Wrapper for SpeechBrain's 16kHz HiFi-GAN.
    
    CRITICAL ARCHITECTURAL NOTE:
    This model is trained on LibriTTS at 16000Hz. 
    It REQUIRES the following Mel-Spectrogram settings in your config:
      - Sample Rate: 16000
      - Hop Length: 256 (approx 16ms frame shift)
      - Win Length: 1024
      - n_mels: 80
      - f_min: 0, f_max: 8000
    """
    def __init__(self, device="cpu", trainable=False):
        super().__init__()
        self.device = str(device) # Force string since SpeechBrain expects a string
        self.trainable = trainable
        
        print(f"[HiFiGAN] Loading SpeechBrain 16kHz Vocoder on {self.device}...")
        
        try:
            from speechbrain.inference.vocoders import HIFIGAN
            
            # Load the correct 16k LibriTTS model
            # Source: https://huggingface.co/speechbrain/tts-hifigan-libritts-16kHz
            self.vocoder = HIFIGAN.from_hparams(
                source="speechbrain/tts-hifigan-libritts-16kHz",
                savedir="pretrained_models/hifigan-16k",
                run_opts={"device": self.device}
            )
            
        except ImportError:
            print("\nCRITICAL ERROR: 'speechbrain' not found.")
            print("Please run: pip install speechbrain\n")
            raise
        except Exception as e:
            raise RuntimeError(f"Failed to load SpeechBrain HiFiGAN: {e}")

        # SpeechBrain models are wrappers. We can control gradients if needed,
        # but usually, we just use them in eval mode for zero-shot VC.
        if not trainable:
            self.vocoder.eval()
            for p in self.vocoder.parameters():
                p.requires_grad = False
        else:
            self.vocoder.train()
            for p in self.vocoder.parameters():
                p.requires_grad = True

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mel: (B, 80, T) log-mel spectrogram
        Returns:
            waveform: (B, T_wav) float32
        """
        # Ensure input is on correct device
        if mel.device != torch.device(self.device):
            mel = mel.to(self.device)
            
        # SpeechBrain's decode_batch expects (B, 80, T) or similar
        # and handles the conversion internally.
        wav = self.vocoder.mods.generator(mel)
        
        # Ensure output is (B, T)
        if wav.dim() == 3:
            wav = wav.squeeze(1)
            
        return wav


class SpeakerEncoderSimple(nn.Module):
    """
    Lightweight speaker encoder using resemblyzer or simple CNN.
    """
    def __init__(self, device="cpu"):
        super().__init__()
        self.device = str(device) # Force string
        
        # Try resemblyzer first (most reliable)
        try:
            import subprocess
            # Only install if import fails to avoid slowing down every run
            try:
                import resemblyzer
            except ImportError:
                subprocess.run(
                    ["pip", "install", "-q", "resemblyzer"],
                    check=True,
                    capture_output=True
                )
            
            from resemblyzer import VoiceEncoder
            self.encoder = VoiceEncoder(device=device)
            self.encoder_type = "resemblyzer"
            print("[SpeakerEncoder] Loaded Resemblyzer encoder")
            
        except Exception as e:
            print(f"[SpeakerEncoder] Resemblyzer not available: {e}")
            print("[SpeakerEncoder] Using simple CNN encoder (untrained - for testing only)")
            
            # Fallback: Simple CNN-based encoder
            class SimpleCNNEncoder(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.conv_layers = nn.Sequential(
                        nn.Conv1d(1, 64, kernel_size=5, stride=2, padding=2),
                        nn.BatchNorm1d(64),
                        nn.ReLU(),
                        nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
                        nn.BatchNorm1d(128),
                        nn.ReLU(),
                        nn.Conv1d(128, 256, kernel_size=5, stride=2, padding=2),
                        nn.BatchNorm1d(256),
                        nn.ReLU(),
                        nn.AdaptiveAvgPool1d(1)
                    )
                
                def forward(self, x):
                    if x.dim() == 2:
                        x = x.unsqueeze(1)  # (B, T) -> (B, 1, T)
                    x = self.conv_layers(x)  # (B, 256, 1)
                    return x.squeeze(-1)  # (B, 256)
            
            self.encoder = SimpleCNNEncoder().to(device)
            self.encoder_type = "simple"
            
            # Initialize with Xavier initialization
            for m in self.encoder.modules():
                if isinstance(m, nn.Conv1d):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
        
        self.eval()
        for p in self.encoder.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def encode_batch(self, waveforms: torch.Tensor) -> torch.Tensor:
        """
        Extract speaker embeddings from waveforms.
        
        Args:
            waveforms: (B, T) float tensor of audio waveforms
            
        Returns:
            embeddings: (B, 1, D) speaker embeddings
        """
        self.eval()
        
        if waveforms.dim() == 1:
            waveforms = waveforms.unsqueeze(0)
        
        # Ensure waveforms are on the correct device for the check, 
        # but resemblyzer might need CPU numpy
        waveforms = waveforms.to(self.device)
        
        if self.encoder_type == "resemblyzer":
            # Resemblyzer expects numpy on CPU
            wav_np = waveforms.cpu().numpy()
            embeddings = []
            # VoiceEncoder processes one by one or list
            # We process list to be efficient
            try:
                # The map logic in resemblyzer might be slower than batch inference 
                # if available, but standard usage is per-utterance
                emb_list = [self.encoder.embed_utterance(w) for w in wav_np]
                embeddings = torch.tensor(emb_list, device=self.device)
            except:
                 # Fallback loop
                embeddings = []
                for w in wav_np:
                    emb = self.encoder.embed_utterance(w)
                    embeddings.append(torch.from_numpy(emb))
                embeddings = torch.stack(embeddings).to(self.device)
                
        else:
            # Simple encoder
            embeddings = self.encoder(waveforms)
        
        if embeddings.dim() == 2:
            embeddings = embeddings.unsqueeze(1)
        
        return embeddings


def load_vocoder(checkpoint_path=None, device="cpu", trainable=False):
    """
    Loads 16kHz SpeechBrain HiFi-GAN vocoder.
    
    Args:
        checkpoint_path: Path to custom checkpoint (ignored for SpeechBrain auto-load)
        device: "cuda" or "cpu"
        trainable: If True, vocoder is trainable
        
    Returns:
        HiFiGANVocoder instance
    """
    if checkpoint_path is not None and checkpoint_path != "":
        print(f"[Warning] Custom checkpoint path '{checkpoint_path}' provided but ignored in favor of SpeechBrain hub.")
    
    vocoder = HiFiGANVocoder(device=device, trainable=trainable)
    return vocoder


def load_speaker_encoder(checkpoint_path, device="cpu"):
    """
    Loads speaker encoder for speaker identity loss.
    
    Args:
        checkpoint_path: Path to custom checkpoint (ignored)
        device: "cuda" or "cpu"
        
    Returns:
        SpeakerEncoderSimple instance
    """
    if checkpoint_path is not None and checkpoint_path != "":
        print(f"[Warning] Custom checkpoint path '{checkpoint_path}' provided but ignored.")
    
    encoder = SpeakerEncoderSimple(device=device)
    return encoder