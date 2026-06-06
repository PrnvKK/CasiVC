import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import HubertModel, HubertConfig
from typing import Optional, Tuple, Dict, Any, List, Union
import warnings
from pathlib import Path

from config import AudioConfig, ModelConfig

audio_config = AudioConfig()
model_config = ModelConfig()


# ══════════════════════════════════════════════════════════════════
# HUBERT LAYER SELECTION CONSTANTS
# ══════════════════════════════════════════════════════════════════
# Model: facebook/hubert-base-ls960 
#
# HuBERT hidden_states indexing:
#   hidden_states[0]  = CNN embedding layer output
#   hidden_states[1]  = Transformer layer 1
#   hidden_states[2]  = Transformer layer 2
#   ...
#   hidden_states[12] = Transformer layer 12 (= last_hidden_state)
#
# Model specs (from verification):
#   - Hidden size: 768
#   - Num layers: 12 (+ 1 CNN = 13 hidden_states)
#   - Actual stride: ~327 samples/frame (~49 frames per second at 16kHz)
#   - Attention mask: Not required (return_attention_mask=None)
#
# Research rationale (from Soft-VC paper):
#   - Layer 12 (final): Too abstract, discards prosody
#   - Layer 9: "Goldilocks zone" - retains phonetic + prosody
#   - Earlier layers: Too much speaker-specific info
# ══════════════════════════════════════════════════════════════════

HUBERT_TRANSFORMER_LAYER = 9  # Which transformer layer (1-12)
HUBERT_HIDDEN_STATE_INDEX = 9  # Index into hidden_states tuple (9 for layer 9)

# Sanity check at module load time
if not (1 <= HUBERT_TRANSFORMER_LAYER <= 12):
    raise ValueError(
        f"HUBERT_TRANSFORMER_LAYER must be 1-12, got {HUBERT_TRANSFORMER_LAYER}"
    )




class HuBERTEncoder(nn.Module):
    """
    HuBERT encoder for extracting continuous semantic features.
    
    Implements zero-shot voice conversion content extraction using:
    - Frozen HuBERT-large model for speaker-invariant features
    - Continuous feature extraction (no quantization)
    - 20ms frame rate for semantic consistency
    - Direct integration with cross-attention mechanism
    """
    
    def __init__(
        self,
        model_name: str = None,
        cache_dir: Optional[str] = None,
        max_audio_length: float = None,
        enable_caching: bool = True
    ):
        """
        Initialize HuBERT encoder.
        
        Args:
            model_name: HuBERT model identifier
            cache_dir: Directory for model caching
            max_audio_length: Maximum audio length in seconds
            enable_caching: Whether to enable model caching
        """
        super().__init__()
        
        # Configuration
        self.model_name = model_name or model_config.hubert_model_name
        self.cache_dir = cache_dir or model_config.hubert_cache_dir
        self.max_audio_length = max_audio_length or model_config.max_audio_length
        self.enable_caching = enable_caching
        
        # Audio processing parameters
        self.sample_rate = audio_config.sample_rate
        self.frame_shift_ms = audio_config.frame_shift_semantic  # 20ms
        self.hop_length = int(self.sample_rate * self.frame_shift_ms / 1000)  # 320 samples
        
        # Maximum sequence length (samples)
        self.max_samples = int(self.max_audio_length * self.sample_rate)
        
        # Initialize HuBERT model
        self._initialize_model()
        
        # Feature dimensions
        self.feature_dim = model_config.hubert_features_dim

        actual_dim = self.hubert_model.config.hidden_size
        if actual_dim != self.feature_dim:
            raise ValueError(
                f"HuBERT model dimension mismatch: "
                f"config expects {self.feature_dim}, "
                f"but model '{self.model_name}' has {actual_dim}"
            )
            
    def _initialize_model(self):
        """Initialize and freeze HuBERT model."""
        try:
            # Setup cache directory
            if self.enable_caching:
                os.makedirs(self.cache_dir, exist_ok=True)
                cache_dir = self.cache_dir
            else:
                cache_dir = None
            
            print(f"Loading HuBERT model: {self.model_name}")
            
            # Load HuBERT model
            self.hubert_model = HubertModel.from_pretrained(
                self.model_name,
                cache_dir=cache_dir,
                torch_dtype=torch.float32,  # Ensure float32 for stability
                output_hidden_states=True,  # We need final hidden states
                return_dict=True
            )
            
            # Freeze all parameters
            for param in self.hubert_model.parameters():
                param.requires_grad = False
            
            # Set to evaluation mode
            self.hubert_model.eval()

            # Verify all parameters are actually frozen
            trainable_params = sum(p.numel() for p in self.hubert_model.parameters() if p.requires_grad)
            if trainable_params > 0:
                raise RuntimeError(f"HuBERT should be frozen but has {trainable_params} trainable parameters")

            print(f"✓ HuBERT model loaded and frozen ({sum(p.numel() for p in self.hubert_model.parameters())/1e6:.1f}M parameters)")
                         
        except Exception as e:
            raise RuntimeError(f"Failed to initialize HuBERT model: {str(e)}")

        # VERIFY HuBERT stride matches assumption
        with torch.no_grad():
            # Test with known input length
            test_audio = torch.randn(16000)  # 1 second at 16kHz
            test_output = self.hubert_model(
                test_audio.unsqueeze(0).to(self.device),
                return_dict=True
            )

            test_features = test_output.hidden_states[HUBERT_HIDDEN_STATE_INDEX]
            
            expected_frames = 16000 // self.hop_length  # Should be ~50 frames
            actual_frames = test_features.shape[1]
            
            # Allow small difference due to padding/boundary effects
            if abs(expected_frames - actual_frames) > 2:
                raise RuntimeError(
                    f"HuBERT stride mismatch! Expected ~{expected_frames} frames, "
                    f"got {actual_frames}. hop_length={self.hop_length} may be incorrect."
                )
            
            print(
                f"✓ HuBERT Transformer layer {HUBERT_TRANSFORMER_LAYER} "
                f"(hidden_states[{HUBERT_HIDDEN_STATE_INDEX}]) stride verified: "
                f"{actual_frames} frames for 1s audio (expected ~{expected_frames})"
            )
    
    def _validate_audio(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Validate and preprocess input audio.
        
        Args:
            audio: Input audio tensor
            
        Returns:
            Validated and preprocessed audio
            
        Raises:
            ValueError: If audio is invalid or too short for HuBERT processing
        """
        if not isinstance(audio, torch.Tensor):
            raise ValueError("Audio must be a torch.Tensor")
        
        if audio.dim() == 0 or audio.numel() == 0:
            raise ValueError("Audio tensor is empty")
        
        # Convert to 1D if needed
        if audio.dim() > 1:
            if audio.shape[0] == 1:
                audio = audio.squeeze(0)
            else:
                # Convert multi-channel to mono
                audio = torch.mean(audio, dim=0)
        
        # Check for NaN/Inf
        if torch.isnan(audio).any():
            raise ValueError("Audio contains NaN values")
        if torch.isinf(audio).any():
            raise ValueError("Audio contains Inf values")
        
        # Ensure reasonable length
        if audio.shape[0] > self.max_samples:
            audio = audio[:self.max_samples]
        
        # Minimum length for HuBERT processing
        # HuBERT has strided convolutions (total stride ~320)
        # 400 samples (~25ms at 16kHz) ensures at least 1 output frame
        min_hubert_samples = 400
        extremely_short_threshold = 16  # ~1ms - clearly invalid
        
        audio_length = audio.shape[0]
        original_valid_length = audio_length
        
        if audio_length < extremely_short_threshold:
            raise ValueError(
                f"Audio extremely short ({audio_length} samples = "
                f"{audio_length/self.sample_rate*1000:.1f}ms). "
                f"Minimum {extremely_short_threshold} samples required."
            )
        
        if audio_length < min_hubert_samples:
            padding = min_hubert_samples - audio_length
            audio = F.pad(audio, (0, padding), mode='constant', value=0.0)
            warnings.warn(
                f"Short audio ({audio_length} samples = {audio_length/self.sample_rate*1000:.1f}ms) "
                f"padded to {min_hubert_samples} samples ({min_hubert_samples/self.sample_rate*1000:.1f}ms)"
            )
        
        return audio, original_valid_length



    def _extract_layer_features(self, audio: torch.Tensor, original_valid_length: int, layer_idx: int = HUBERT_HIDDEN_STATE_INDEX) -> torch.Tensor:
        """
        Extract features from a specific HuBERT layer.
        
        Args:
            audio: Preprocessed audio tensor [samples]
            layer_idx: Index into hidden_states (1-12 for transformer layers)
        """
        try:
            device = next(self.hubert_model.parameters()).device
            audio = audio.to(device)
            max_val = audio.abs().max()
            if max_val > 1e-8:
                if max_val > 1.0 + 1e-5:
                    warnings.warn(f"Audio unnormalized (max: {max_val:.4f}), auto-normalizing")
                audio = audio / max_val
            else:
                warnings.warn("Audio is near-silent, may produce poor features")
            outputs = self.hubert_model(
                input_values=audio.unsqueeze(0),
                output_hidden_states=True,
                return_dict=True
            )
            features = outputs.hidden_states[layer_idx].squeeze(0)
            if features.dim() != 2:
                raise RuntimeError(f"Unexpected feature shape from layer {layer_idx}: {features.shape}")
            return features
        except Exception as e:
            raise RuntimeError(f"HuBERT layer {layer_idx} extraction failed: {str(e)}")

    def _extract_continuous_features(self, audio: torch.Tensor, original_valid_length: int) -> torch.Tensor:
        """
        Extract continuous semantic features from HuBERT.
        
        Args:
            audio: Preprocessed audio tensor [samples]
            
        Returns:
            Continuous semantic features [time, features]
        """
        try:
            # Move to same device as model
            device = next(self.hubert_model.parameters()).device
            audio = audio.to(device)


            max_val = audio.abs().max()
            if max_val > 1e-8:  # Avoid division by zero
                if max_val > 1.0:
                    # Only warn if actually unnormalized (not just slightly over due to float precision)
                    if max_val > 1.0 + 1e-5:
                        warnings.warn(f"Audio unnormalized (max: {max_val:.4f}), auto-normalizing")
                audio = audio / max_val
            else:
                warnings.warn("Audio is near-silent, may produce poor features")


            # Extract features using HuBERT
            # Input shape: [batch_size, sequence_length]
            outputs = self.hubert_model(
                input_values=audio.unsqueeze(0),  # Add batch dimension
                output_hidden_states=True,
                return_dict=True
            )

            continuous_features = outputs.hidden_states[HUBERT_HIDDEN_STATE_INDEX].squeeze(0)

            if continuous_features.dim() != 2:
                raise RuntimeError(
                    f"Unexpected feature shape from layer {HUBERT_TRANSFORMER_LAYER}: "
                    f"{continuous_features.shape}"
                )

            if continuous_features.shape[1] != self.feature_dim:
                raise RuntimeError(
                    f"Feature dimension mismatch at layer {HUBERT_TRANSFORMER_LAYER}: "
                    f"{continuous_features.shape[1]} != {self.feature_dim}"
                )
            
            return continuous_features
                
        except Exception as e:
            raise RuntimeError(f"HuBERT feature extraction failed: {str(e)}")

    
    def extract_features(self, content_audio: torch.Tensor, return_attention_mask: bool = True) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Extract semantic features from content audio.
        
        Args:
            content_audio: Content audio tensor [samples]
            return_attention_mask: Whether to return attention mask
            
        Returns:
            Semantic features [time, feature_dim] or tuple with attention mask
        """
        # Validate input - returns tuple
        audio, original_valid_length = self._validate_audio(content_audio)
        
        # Extract continuous features with valid length info
        features = self._extract_continuous_features(audio, original_valid_length)
        
        if return_attention_mask:
          # Calculate actual valid frames based on HuBERT's downsampling
          # HuBERT stride is approximately 320 samples (20ms at 16kHz)
          valid_feature_frames = (original_valid_length + self.hop_length - 1) // self.hop_length
          
          # Clamp to actual output length (in case of rounding differences)
          valid_feature_frames = min(valid_feature_frames, features.shape[0])
          
          # Create attention mask
          attention_mask = torch.zeros(features.shape[0], dtype=torch.bool, device=features.device)
          attention_mask[:valid_feature_frames] = True
          
          return features, attention_mask

          return features, attention_mask

    
        return features


    
    def forward(self, content_audio, return_attention_mask=False):
        """Forward pass for batch processing."""
        
        # Handle single tensor input - convert to list for uniform processing
        if isinstance(content_audio, torch.Tensor):
            content_audio = [content_audio]
        
        # Handle batch of variable-length tensors
        if not isinstance(content_audio, (list, tuple)):
            raise ValueError("content_audio must be Tensor or List[Tensor]")
        
        if len(content_audio) == 0:
            raise ValueError("Empty batch provided")
        
        # Process each audio tensor
        batch_features = []
        batch_masks = []
        
        for audio in content_audio:
            if return_attention_mask:
                features, mask = self.extract_features(audio, return_attention_mask=True)
                batch_masks.append(mask)
            else:
                features = self.extract_features(audio, return_attention_mask=False)
            
            batch_features.append(features)
        
        
        if len(batch_features) == 1:
            # Add batch dimension for consistency
            padded_features = batch_features[0].unsqueeze(0)  # [1, seq_len, feature_dim]
            if return_attention_mask:
                padded_masks = batch_masks[0].unsqueeze(0)  # [1, seq_len]
                return padded_features, padded_masks
            return padded_features
        
        # Manual padding
        max_len = max(f.shape[0] for f in batch_features)
        batch_size = len(batch_features)
        feature_dim = batch_features[0].shape[1]
        device = batch_features[0].device
        
        # Initialize with padding value
        padded_features = torch.zeros(
        batch_size, max_len, feature_dim, 
        dtype=batch_features[0].dtype,
        device=device
        )
        
        # Fill in actual features
        for i, features in enumerate(batch_features):
            length = features.shape[0]
            padded_features[i, :length] = features
        
        if return_attention_mask:
            padded_masks = torch.zeros(
                batch_size, max_len, 
                dtype=torch.bool, 
                device=device
            )

            for i, mask in enumerate(batch_masks):
                padded_masks[i, :mask.shape[0]] = mask
        
            return padded_features, padded_masks
        
        # Return as list of tensors
        return padded_features

    @torch.no_grad()
    def extract_speaker_layer(self, audio_batch):
        """Extract per-frame speaker features from HuBERT layer 1.
        
        Layer 1 preserves acoustic speaker identity (F0, formants) before
        semantic abstraction. Returns time-varying tokens instead of a
        single pooled embedding, enabling per-frame timbre modulation
        in downstream cross-attention.
        
        Args:
            audio_batch: list of 1D tensors [T_i] or batched tensor (B, T)
        Returns:
            padded features [B, T_max, 768] — time-varying speaker tokens
        """
        if isinstance(audio_batch, torch.Tensor):
            if audio_batch.dim() == 2:
                audio_batch = [audio_batch[i] for i in range(audio_batch.shape[0])]
            else:
                audio_batch = [audio_batch]
        
        if not isinstance(audio_batch, (list, tuple)):
            raise ValueError("audio_batch must be Tensor or List[Tensor]")
        
        batch_features = []
        for audio in audio_batch:
            audio_proc, valid_len = self._validate_audio(audio)
            feats = self._extract_layer_features(audio_proc, valid_len, layer_idx=1)
            batch_features.append(feats)
        
        if len(batch_features) == 1:
            return batch_features[0].unsqueeze(0)
        
        max_len = max(f.shape[0] for f in batch_features)
        padded = torch.zeros(
            len(batch_features), max_len, self.feature_dim,
            dtype=batch_features[0].dtype, device=batch_features[0].device
        )
        for i, features in enumerate(batch_features):
            padded[i, :features.shape[0]] = features
        return padded


    @property
    def device(self) -> torch.device:
        """Get model device."""
        return next(self.hubert_model.parameters()).device
    
    def to(self, device: torch.device):
        """Move model to device."""
        super().to(device)
        self.hubert_model.to(device)
        return self
    
    def get_output_dim(self) -> int:
        """Get output feature dimension."""
        return self.feature_dim
    
    def get_frame_rate(self) -> float:
        """Get frame rate in Hz."""
        return 1000.0 / self.frame_shift_ms
    
    def estimate_output_length(self, audio_length_seconds: float) -> int:
        """
        Estimate output sequence length for given audio duration.
        
        Args:
            audio_length_seconds: Input audio duration in seconds
            
        Returns:
            Estimated output sequence length
        """
        audio_samples = int(audio_length_seconds * self.sample_rate)
        # HuBERT downsamples by hop_length
        return audio_samples // self.hop_length

# Utility functions for integration
def create_hubert_encoder(**kwargs) -> HuBERTEncoder:
    """
    Factory function to create HuBERT encoder.
    
    Args:
        **kwargs: Arguments passed to HuBERTEncoder
        
    Returns:
        Initialized HuBERTEncoder
    """
    return HuBERTEncoder(**kwargs)


def batch_extract_features(
    encoder: HuBERTEncoder,
    audio_list: List[torch.Tensor],
    device: Optional[torch.device] = None
) -> List[torch.Tensor]:
    """
    Extract features from batch of variable-length audio tensors.
    
    Args:
        encoder: HuBERT encoder instance
        audio_list: List of audio tensors
        device: Target device
        
    Returns:
        List of feature tensors
    """
    if device is not None:
        encoder = encoder.to(device)
    
    return encoder(audio_list)


# Standalone testing
if __name__ == "__main__":
  print("This is hubert_encoder.py")

