# hubertvc/models/mel_encoder.py
"""
ECAPA-TDNN based frozen speaker encoder for zero-shot voice conversion.
Replaces the trainable mel encoder with a frozen pre-trained ECAPA-TDNN model.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Union, Dict, Any
import warnings
from pathlib import Path

# ============================================================================
# MONKEYPATCH: Fix torchaudio backend compatibility
# ============================================================================
try:
    import torchaudio
    if not hasattr(torchaudio, 'list_audio_backends'):
        def _mock_list_audio_backends():
            try:
                import torchaudio.backend
                return ["sox_io", "soundfile"]
            except (ImportError, AttributeError):
                return ["soundfile"]
        torchaudio.list_audio_backends = _mock_list_audio_backends
except ImportError:
    warnings.warn("torchaudio not available", ImportWarning)
# ============================================================================

import os
import huggingface_hub

# Save the original function
_orig_hf_hub_download = huggingface_hub.hf_hub_download

def _patched_hf_hub_download(*args, **kwargs):
    # Rename deprecated argument
    if 'use_auth_token' in kwargs:
        kwargs['token'] = kwargs.pop('use_auth_token')

    # Extract the filename being requested
    filename = kwargs.get('filename')
    if filename is None and len(args) > 1:
        filename = args[1]
        
    try:
        # Try to download normally
        return _orig_hf_hub_download(*args, **kwargs)
    except Exception as e:
        # If it's custom.py and we get a 404 (or similar missing file error)
        if filename == 'custom.py' and ('404' in str(e) or 'Not Found' in str(e)):
            
            # --- NEW FIX: Bulletproof the cache_dir fallback ---
            cache_dir = kwargs.get('cache_dir')
            if cache_dir is None:
                cache_dir = '/tmp/speechbrain_dummy'
            # ---------------------------------------------------
            
            dummy_path = os.path.join(cache_dir, 'custom.py')
            os.makedirs(os.path.dirname(dummy_path), exist_ok=True)
            with open(dummy_path, 'w') as f:
                f.write('') # Empty file
            return dummy_path
            
        # If it's a different error or file, raise it normally
        raise e

# Apply the patch
huggingface_hub.hf_hub_download = _patched_hf_hub_download

try:
    from speechbrain.pretrained import EncoderClassifier
    SPEECHBRAIN_AVAILABLE = True
except ImportError:
    SPEECHBRAIN_AVAILABLE = False
    warnings.warn("speechbrain not installed. Install with: pip install speechbrain")

try:
    from config import AudioConfig, ModelConfig
    audio_config = AudioConfig()
    model_config = ModelConfig()
except ImportError:
    warnings.warn("Could not import configs")


class MelEncoder(nn.Module):
    """
    Frozen ECAPA-TDNN speaker encoder with manual device enforcement.
    """
    
    def __init__(
        self,
        model_name: str = "speechbrain/spkrec-ecapa-voxceleb",
        output_dim: int = None,
        num_speaker_tokens: int = 64,
        device: str = None
    ):
        super().__init__()
        
        if not SPEECHBRAIN_AVAILABLE:
            raise ImportError("speechbrain is required. Install with: pip install speechbrain")
        
        self.output_dim = output_dim or model_config.mel_encoder_output_dim
        self.num_speaker_tokens = 8  # Fix: Project into 8 tokens to enable multi-token attention
        self.model_name = model_name
        
        # Audio parameters
        self.input_dim = audio_config.n_mel_bands
        self.sample_rate = audio_config.sample_rate
        
        print(f"Loading frozen ECAPA-TDNN model: {model_name}")
        self.ecapa_model = self._load_ecapa_model(model_name, device)
        
        # Freeze parameters
        self._freeze_ecapa()
        
        # Get embedding dimension
        self.ecapa_dim = self._get_ecapa_dim()
        print(f"ECAPA-TDNN embedding dimension: {self.ecapa_dim}D")
        
        # Projection
        self.projection = nn.Sequential(
            nn.Linear(self.ecapa_dim, self.num_speaker_tokens * self.output_dim),
            nn.Tanh()  # bounds output to [-1, 1] while preserving inter-speaker amplitude differences
        )

        self._init_projection()
        
        # Initial device move
        if device:
            self.to(torch.device(device))

    def _load_ecapa_model(self, model_name: str, device: Optional[str] = None) -> EncoderClassifier:
        try:
            run_opts = {"device": device} if device else None
            return EncoderClassifier.from_hparams(
                source=model_name,
                run_opts=run_opts,
                savedir=f"pretrained_models/{model_name.replace('/', '_')}"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load ECAPA-TDNN model: {e}")

    def _freeze_ecapa(self):
        for param in self.ecapa_model.parameters():
            param.requires_grad = False
        self.ecapa_model.eval()

    def _get_ecapa_dim(self) -> int:
        device = next(self.ecapa_model.parameters()).device
        dummy_wav = torch.randn(1, self.sample_rate).to(device)
        dummy_lens = torch.ones(1).to(device)
        with torch.no_grad():
            # Use manual pipeline for dimension check too
            emb = self._safe_encode(dummy_wav, dummy_lens)
        return emb.shape[-1]

    def _init_projection(self):
        for module in self.projection.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.03)  # reduced from 0.1 to prevent Tanh saturation (max → 0.86, not 0.9997)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _safe_encode(self, wavs: torch.Tensor, wav_lens: torch.Tensor) -> torch.Tensor:
        """
        Manually execute SpeechBrain pipeline to enforce device placement.
        Bypasses 'encode_batch' to fix the CPU fallback issues.
        """
        device = wavs.device
        
        # Access internal modules from SpeechBrain
        # Note: 'mods' can be a dict or SimpleNamespace depending on version
        mods = self.ecapa_model.mods
        if hasattr(mods, 'compute_features'):
            compute_features = mods.compute_features
            mean_var_norm = mods.mean_var_norm
            embedding_model = mods.embedding_model
        else:
            # Fallback for older versions or dict structure
            compute_features = mods['compute_features']
            mean_var_norm = mods['mean_var_norm']
            embedding_model = mods['embedding_model']

        # 1. Compute Features (e.g. Fbank)
        feats = compute_features(wavs)
        if feats.device != device:
            feats = feats.to(device)

        # 2. Normalization
        # InputNormalization in SpeechBrain can be finicky about devices
        feats = mean_var_norm(feats, wav_lens)
        if feats.device != device:
            feats = feats.to(device)

        # 3. Embedding
        # The internal try/except for lengths happens here, but since feats 
        # is now explicitly on GPU, the fallback conv1d will succeed.
        embeddings = embedding_model(feats, wav_lens)
        
        return embeddings

    def extract_speaker_features(self, ref_audio: torch.Tensor, apply_projection: bool = True) -> torch.Tensor:
        if not isinstance(ref_audio, torch.Tensor):
            raise ValueError("ref_audio must be a torch.Tensor")
        
        if ref_audio.dim() == 1:
            ref_audio = ref_audio.unsqueeze(0)
        
        batch_size = ref_audio.shape[0]
        
        # 1. Get Model Device
        device = next(self.ecapa_model.parameters()).device
        
        # 2. Ensure Inputs are on Device
        if ref_audio.device != device:
            ref_audio = ref_audio.to(device)
            
        # 3. Create lengths explicitly on Device
        wav_lens = torch.ones(batch_size, device=device)
        
        with torch.no_grad():
            self.ecapa_model.eval()
            ecapa_embeddings = self._safe_encode(ref_audio, wav_lens)
        
        # ============ DEBUG: RAW ECAPA EMBEDDINGS ============
        print(f"\n[ECAPA RAW EMBEDDINGS]")
        print(f"  Shape: {ecapa_embeddings.shape}")  # Should be (B, 192)
        print(f"  Mean: {ecapa_embeddings.mean().item():.4f}")
        print(f"  Std: {ecapa_embeddings.std().item():.4f}")
        print(f"  Min: {ecapa_embeddings.min().item():.4f}")
        print(f"  Max: {ecapa_embeddings.max().item():.4f}")
        print(f"  L2 Norm: {torch.norm(ecapa_embeddings, p=2, dim=-1).mean().item():.4f}")
        
        # Dimensionality coverage
        dim_std = ecapa_embeddings.std(dim=0)  # Std across batch for each dim
        active_dims = (dim_std > 0.01).sum().item()
        print(f"  Active dimensions (std > 0.01): {active_dims}/{ecapa_embeddings.shape[-1]}")
        
        # Check for dead dimensions (all near zero)
        dead_dims = (ecapa_embeddings.abs().mean(dim=0) < 0.001).sum().item()
        print(f"  Dead dimensions (|mean| < 0.001): {dead_dims}/{ecapa_embeddings.shape[-1]}")
        
        if apply_projection:
            projected = self.projection(ecapa_embeddings)
            
            # ============ DEBUG: AFTER PROJECTION ============
            print(f"\n[AFTER PROJECTION (before reshape)]")
            print(f"  Shape: {projected.shape}")  # Should be (B, 1*96 = 96)
            print(f"  Mean: {projected.mean().item():.4f}")
            print(f"  Std: {projected.std().item():.4f}")
            print(f"  Min: {projected.min().item():.4f}")
            print(f"  Max: {projected.max().item():.4f}")
            
            speaker_features = projected.view(
                batch_size, 
                self.num_speaker_tokens, 
                self.output_dim
            )
            
            # ============ DEBUG: SPEAKER FEATURES (for cross-attention) ============
            print(f"\n[SPEAKER FEATURES - FINAL]")
            print(f"  Shape: {speaker_features.shape}")  # Should be (B, 64, 96)
            print(f"  Mean: {speaker_features.mean().item():.4f}")
            print(f"  Std: {speaker_features.std().item():.4f}")
            print(f"  Min: {speaker_features.min().item():.4f}")
            print(f"  Max: {speaker_features.max().item():.4f}")
            
            # Token diversity (are tokens different from each other?)
            token_means = speaker_features.mean(dim=-1)  # (B, 64) - mean of each token
            token_diversity = token_means.std(dim=-1).mean().item()
            print(f"  Token diversity (std of token means): {token_diversity:.4f}")
            
            # Dimension coverage across tokens
            dim_std_per_token = speaker_features.std(dim=1).mean(dim=0)  # (96,) - std per dim across tokens
            active_dims_tokens = (dim_std_per_token > 0.01).sum().item()
            print(f"  Active dims across tokens (std > 0.01): {active_dims_tokens}/96")
            
            # Check for collapsed tokens (all tokens identical)
            token_pairwise_dist = torch.cdist(speaker_features[0], speaker_features[0], p=2)  # (64, 64)
            avg_token_dist = token_pairwise_dist.mean().item()
            print(f"  Avg pairwise token distance (L2): {avg_token_dist:.4f}")
            
            # Rank check (effective dimensionality)
            U, S, V = torch.svd(speaker_features[0])  # SVD on first sample
            explained_variance = (S ** 2) / (S ** 2).sum()
            cumsum_var = explained_variance.cumsum(dim=0)
            dims_for_95 = (cumsum_var < 0.95).sum().item() + 1
            print(f"  Effective rank (95% variance): {dims_for_95}/96")
            print(f"  Singular values (top 5): {S[:5].tolist()}")
            
            # NaN/Inf check
            has_nan = torch.isnan(speaker_features).any().item()
            has_inf = torch.isinf(speaker_features).any().item()
            print(f"  Contains NaN: {has_nan}")
            print(f"  Contains Inf: {has_inf}")
            
        else:
            speaker_features = ecapa_embeddings
        
        return speaker_features

    def forward(self, ref_audio: Union[torch.Tensor, List[torch.Tensor]]) -> torch.Tensor:
        device = next(self.ecapa_model.parameters()).device
        
        if isinstance(ref_audio, torch.Tensor):
            ref_audio = ref_audio.to(device)
            return self.extract_speaker_features(ref_audio)
        
        if not isinstance(ref_audio, (list, tuple)):
            raise ValueError("ref_audio must be Tensor or List[Tensor]")
        
        max_samples = max(audio.shape[0] if audio.dim() == 1 else audio.shape[-1] for audio in ref_audio)
        
        padded_audio = []
        for audio in ref_audio:
            audio = audio.to(device)
            if audio.dim() == 1:
                if audio.shape[0] < max_samples:
                    audio = F.pad(audio, (0, max_samples - audio.shape[0]))
                padded_audio.append(audio)
            elif audio.dim() == 2:
                audio = audio.squeeze(0)
                if audio.shape[0] < max_samples:
                    audio = F.pad(audio, (0, max_samples - audio.shape[0]))
                padded_audio.append(audio)
                
        batched_audio = torch.stack(padded_audio, dim=0)
        return self.extract_speaker_features(batched_audio)

    def to(self, device: torch.device):
        """
        Custom to() method to strictly move SpeechBrain internals.
        """
        super().to(device)
        self.ecapa_model = self.ecapa_model.to(device)
        self.projection = self.projection.to(device)
        
        # Aggressively move modules inside SpeechBrain's 'mods' container
        if hasattr(self.ecapa_model, 'mods'):
            mods = self.ecapa_model.mods
            # Handle both dict and SimpleNamespace
            iterator = mods.items() if isinstance(mods, dict) else vars(mods).items()
            
            for name, module in iterator:
                if isinstance(module, torch.nn.Module):
                    module.to(device)
                    # Force buffers if they missed the bus
                    for buffer_name, buffer in module.named_buffers():
                        setattr(module, buffer_name, buffer.to(device))
                        
        return self

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def estimate_parameters(self) -> Dict[str, int]:
        ecapa_params = sum(p.numel() for p in self.ecapa_model.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        projection_params = sum(p.numel() for p in self.projection.parameters())
        return {
            "ecapa_tdnn_frozen": ecapa_params,
            "projection": projection_params,
            "trainable": trainable_params,
            "total": ecapa_params + projection_params
        }

def create_mel_encoder(**kwargs) -> MelEncoder:
    return MelEncoder(**kwargs)

if __name__ == "__main__":
    print("Testing frozen ECAPA-TDNN mel encoder...")