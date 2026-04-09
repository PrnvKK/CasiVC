# hubertvc/data/audio_utils.py
import os
import torch
import torchaudio
from torchaudio.transforms import MelSpectrogram, Resample
from typing import Optional, Tuple, Dict, Any
import warnings
import soundfile as sf


from config import AudioConfig
config = AudioConfig()


def load_audio(file_path: str, sample_rate: Optional[int] = None) -> torch.Tensor:
    """
    Load audio file using soundfile directly to bypass Torchaudio backend issues.
    """
    if sample_rate is None:
        sample_rate = config.sample_rate
        
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Audio file not found: {file_path}")
    
    try:
        # Direct SoundFile read. Bypasses torchaudio.load() complexity completely.
        wav_numpy, orig_sr = sf.read(file_path)
        
        # Convert to Tensor
        waveform = torch.from_numpy(wav_numpy).float()
        
        # Handle shapes: SoundFile returns (Time, Channels) or just (Time,)
        # PyTorch expects (Channels, Time)
        if waveform.ndim == 1:
            # (Time,) -> (1, Time)
            waveform = waveform.unsqueeze(0)
        else:
            # (Time, Channels) -> (Channels, Time)
            waveform = waveform.transpose(0, 1)
        # --- NUCLEAR FIX END ---

    except Exception as e:
        raise RuntimeError(f"Failed to load audio file {file_path}: {str(e)}")
    
    # Validate audio
    if waveform.numel() == 0:
        raise ValueError(f"Audio file is empty: {file_path}")
    
    # Convert to mono if multichannel
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    # Resample if needed
    if orig_sr != sample_rate:
        # Note: Resample expects (..., time), which we now have.
        resampler = Resample(orig_freq=orig_sr, new_freq=sample_rate)
        waveform = resampler(waveform)
    
    return waveform.squeeze(0)  # Return 1D tensor

def extract_mel_spectrogram(
    waveform: torch.Tensor,
    sample_rate: Optional[int] = None,
    n_mels: Optional[int] = None,
    f_min: Optional[int] = None,
    f_max: Optional[int] = None,
    normalize = False
) -> torch.Tensor:
    """
    Extract mel-spectrogram compatible with HiFi-GAN vocoder.
    
    Args:
        waveform: 1D audio tensor
        sample_rate: Sample rate (default from config)
        
    Returns:
        Log-mel-spectrogram (n_mels, time) - NO per-utterance normalization
    """
    if sample_rate is None:
        sample_rate = config.sample_rate

    n_mels = n_mels if n_mels is not None else config.n_mel_bands
    f_min = f_min if f_min is not None else config.mel_fmin
    f_max = f_max if f_max is not None else config.mel_fmax
    
    if waveform.ndim != 1:
        raise ValueError("Input waveform must be 1D tensor")
    
    if waveform.numel() == 0:
        raise ValueError("Input waveform is empty")
    
    # Calculate parameters
    hop_length = 256
    win_length = 1024
    n_fft = 1024
    
    # Pad short waveforms
    if waveform.shape[0] < n_fft:
        pad_length = n_fft - waveform.shape[0]
        waveform = torch.nn.functional.pad(waveform, (0, pad_length), mode='constant', value=0)

        if waveform.shape[0] < n_fft:
            print(f"[extract_mel] Padded from {original_length} to {waveform.shape[0]} samples")

    mel_transform = MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        f_min=f_min,
        f_max=f_max,
        n_mels=n_mels,
        window_fn=torch.hann_window,
        power=1.0,
        normalized=True,
        center=False,
        pad_mode='constant',
        mel_scale='slaney'  #
    )
    
    mel_transform = mel_transform.to(waveform.device)
    mel_spec = mel_transform(waveform)

    # Clamp and log (dynamic range compression)
    mel_spec = torch.clamp(mel_spec, min=1e-5)
    mel_spec = torch.log(mel_spec)  # Natural log

    if normalize:
        mean = mel_spec.mean()
        std = mel_spec.std()
        # Safe division to prevent NaNs on silent audio
        mel_spec = (mel_spec - mean) / (std + 1e-5)

    # Error checking
    if torch.isnan(mel_spec).any() or torch.isinf(mel_spec).any():
        raise RuntimeError("NaN/Inf in mel-spectrogram")
    
    return mel_spec



def process_audio_chunk(
    waveform: torch.Tensor,
    chunk_length: float = 4.0,
    overlap: float = 0.5,
    sample_rate: Optional[int] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Process audio for mel-spectrogram extraction.
    
    Args:
        waveform (torch.Tensor): 1D audio tensor
        chunk_length (float): Deprecated, kept for API compatibility
        overlap (float): Deprecated, kept for API compatibility
        sample_rate (int, optional): Sample rate
        
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Mel-spectrogram and boundaries
    """
    if sample_rate is None:
        sample_rate = config.sample_rate
    
    # Simply process the entire audio at once
    # Modern GPUs handle even 30+ second audio without issues
    mel = extract_mel_spectrogram(waveform, sample_rate)
    
    # Return full mel with boundary indicating entire waveform
    boundaries = torch.tensor([[0, waveform.shape[0]]], dtype=torch.long)
    
    return mel, boundaries



def split_utterance_for_training(
    waveform: torch.Tensor,
    ref_length_range: Tuple[float, float] = (2.0, 3.0),
    sample_rate: Optional[int] = None,
    min_content_length: float = 1.0,
    deterministic=False
) -> Tuple[torch.Tensor, torch.Tensor]:
    
    if sample_rate is None:
        sample_rate = config.sample_rate

    min_content_samples = int(min_content_length * sample_rate)
    
    # --- DETERMINISTIC BRANCH ---
    if deterministic:
      
        g = torch.Generator(device='cpu')
        g.manual_seed(0)


        valid_start = int(0.1 * waveform.shape[0])
        valid_end = int(0.9 * waveform.shape[0])
        valid_waveform = waveform[valid_start:valid_end]

        ref_samples = int(ref_length_range[0] * sample_rate)

        if valid_waveform.shape[0] < ref_samples + min_content_samples:
            valid_waveform = waveform  # Fallback to full waveform
            if valid_waveform.shape[0] < ref_samples + min_content_samples:
                raise ValueError(
                    f"Audio too short: {waveform.shape[0]/sample_rate:.2f}s, "
                    f"need {(ref_samples + min_content_samples)/sample_rate:.2f}s"
                )

        max_split_index = valid_waveform.shape[0] - min_content_samples
        min_split_index = ref_samples
        
        split_point = torch.randint(
            min_split_index, max_split_index + 1, (1,), generator=g
        ).item()

        # Randomly swap ref/content sides so model sees ALL positions as content
        swap = torch.randint(0, 2, (1,), generator=g).item()
        
        if swap == 0:
            # Original: ref before content
            ref_audio = valid_waveform[split_point - ref_samples : split_point]
            content_audio = valid_waveform[split_point:]
        else:
            # Swapped: content before ref
            content_audio = valid_waveform[:split_point - ref_samples]
            ref_audio = valid_waveform[split_point - ref_samples : split_point]
            # Ensure content is long enough after swap
            if content_audio.shape[0] < min_content_samples:
                # Fallback to original order
                ref_audio = valid_waveform[split_point - ref_samples : split_point]
                content_audio = valid_waveform[split_point:]
            
        return ref_audio, content_audio
    
    # --- RANDOM BRANCH ---
    min_ref_samples = int(ref_length_range[0] * sample_rate)
    max_ref_samples = int(ref_length_range[1] * sample_rate)
    
    total_samples = waveform.shape[0]
    
    # This allows shorter utterances to be used if they can fit min_ref + min_content
    min_required_samples = min_ref_samples + min_content_samples
    
    if total_samples < min_required_samples:
        raise ValueError(
            f"Audio too short: {total_samples/sample_rate:.2f}s, "
            f"need at least {min_required_samples/sample_rate:.2f}s"
        )
    
    # Randomly choose reference length within valid range
    max_allowed_ref = min(max_ref_samples, total_samples - min_content_samples)
    ref_samples = torch.randint(min_ref_samples, max_allowed_ref + 1, (1,)).item()
    
    # Randomly pick split point
    max_split_point = total_samples - min_content_samples
    split_point = torch.randint(ref_samples, max_split_point + 1, (1,)).item()
    
    # Randomly swap ref/content sides so model sees ALL positions as content
    if torch.rand(1).item() < 0.5:
        # Original: ref before content
        ref_segment = waveform[split_point - ref_samples : split_point]
        content_segment = waveform[split_point:]
    else:
        # Swapped: content is the beginning, ref is after it
        content_segment = waveform[:split_point - ref_samples]
        ref_segment = waveform[split_point - ref_samples : split_point]
        # Ensure content is long enough after swap
        if content_segment.shape[0] < min_content_samples:
            # Fallback to original order
            ref_segment = waveform[split_point - ref_samples : split_point]
            content_segment = waveform[split_point:]
    
    return ref_segment, content_segment






# Standalone testing
if __name__ == "__main__":
    print("This is audio_utils.py....")
