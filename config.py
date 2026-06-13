# hubertvc/config.py
import os
from dataclasses import dataclass
from typing import List, Optional
from typing import List, Tuple
from dataclasses import field


@dataclass
class AudioConfig:

  n_mel_bends: int = 80
  sample_rate: int = 16000
  mel_fmin: int = 0
  mel_fmax: int = 8000
  n_mel_bands: int = 80
  frame_shift_reference: int = 16  # ms - for reference mel-spectrograms (speaker identity)
  frame_shift_semantic: int = 20   # ms - for semantic token processing (HuBERT)

  @property
  def frame_shift(self) -> int:
      """Default frame shift for alignment and configuration hashing."""
      return self.frame_shift_semantic  # Use semantic as default (20ms)
  
  @property  
  def default_frame_shift(self) -> int:
      """Alias for frame_shift property for clarity."""
      return self.frame_shift_semantic


@dataclass
class DataConfig:
    # Dataset configuration
    dataset_name: str = "libritts"  # "libritts", "vctk", "custom"
    dataset_root: str = "/content/LibriTTS"  # Change this to your actual path
    
    # Development mode - use dev-clean split by speakers
    use_dev_subset: bool = True  # Set True for development, False for full dataset
    dev_subset_name: str = "dev-clean"
    
    # Speaker-level splits (percentages)
    train_speaker_ratio: float = 0.6  # 60% speakers for training
    val_speaker_ratio: float = 0.2    # 20% speakers for validation  
    test_speaker_ratio: float = 0.2   # 20% speakers for testing
    
    # Full dataset paths (when use_dev_subset=False)
    train_subsets: List[str] = field(default_factory=lambda: ["train-clean-100", "train-clean-360"])
    val_subsets: List[str] = field(default_factory=lambda: ["dev-clean"])
    test_subsets: List[str] = field(default_factory=lambda: ["test-clean"])
    
    # Cache configuration
    cache_dir: str = "cache/"
    enable_caching: bool = True
    cache_format: str = "hdf5"
    
    # Data filtering
    min_duration: float = 0.5   # Minimum utterance duration (seconds)
    max_duration: float = 30.0  # Maximum utterance duration (seconds)
    
    # Training data preparation
    ref_length_range: Tuple[float, float] = (1.0, 2.0)  # Reference segment length range
    min_content_length: float = 0.5  # Minimum content segment length
    
    # Random seed for reproducible speaker splits
    speaker_split_seed: int = 42

    max_items: Optional[int] = 4000
    
    def get_dataset_path(self) -> str:
        """Get the appropriate dataset path based on configuration."""
        if self.use_dev_subset:
            return os.path.join(self.dataset_root, self.dev_subset_name)
        else:
            return self.dataset_root
    
    def get_speaker_split_ratios(self) -> Tuple[float, float, float]:
        """Get speaker split ratios, ensuring they sum to 1.0."""
        total = self.train_speaker_ratio + self.val_speaker_ratio + self.test_speaker_ratio
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Speaker split ratios must sum to 1.0, got {total}")
        return (self.train_speaker_ratio, self.val_speaker_ratio, self.test_speaker_ratio)

@dataclass  
class ModelConfig:
    # HuBERT Configuration - FIX: Match actual implementation
    hubert_model_name: str = "facebook/hubert-base-ls960"  # NOT large
    hubert_features_dim: int = 768  # Base model is 768
    hubert_cache_dir: str = "./cache/hubert"
    max_audio_length: float = 30.0

    # Speaker encoder - ADD: Missing configuration
    speaker_projection_dim: int = 96 
    
    # Cross-attention - ADD: Missing configuration  
    cross_attention_dim: int = 96  # ADD: Actual d_model value
    cross_attention_heads: int = 3  # ADD: Actual num_heads
    cross_attention_dropout: float = 0.1  # ADD: Dropout rate

    # Keep existing for backward compatibility
    kmeans_clusters: int = 2000
    hubert_model_path: str = "facebook/hubert-base-ls960"  # Update to match
    attention_heads: int = 2
    attention_dim: int = 184
    mel_encoder_kernel_size: int = 5 

    mel_encoder_output_dim: int =  96
    mel_encoder_hidden_dims: List[int] = field(default_factory=lambda: [96])


    # ============ MobileNetV3 Decoder Configuration ============
    # Input: cross_attention_dim (96)
    # Output: 80 mel bands
    
    mobilenet_input_dim: int = 96  # Must match cross_attention_dim
    mobilenet_channel_progression: List[int] = field(
        default_factory=lambda: [96, 128, 160, 192, 96]
    )
    mobilenet_expand_ratios: List[int] = field(
        default_factory=lambda: [4, 3, 3, 2]
    )
    mobilenet_use_se: List[bool] = field(
        default_factory=lambda: [False, False, True, True]  # Reverted: SE on block3 provides some speaker-dependent reweighting; removing it made b3_body worse
    )

    block3_residual_scale: float = 0.35  # Reduced from 0.5: body branch (cent_cos 0.83) dilutes identity-FiLM path (cent_cos 0.51). 0.35 preserves body gradient while reducing dilution.

    mobilenet_upsample_stages: List[bool] = field(
    default_factory=lambda: [False, False, False, False]
)

    
    mobilenet_kernel_size_first: int = 5
    mobilenet_kernel_size: int = 3
    mobilenet_norm: str = "gn"  
    mobilenet_upsample_first: bool = True  
         
            



@dataclass
class TrainingConfig:
  # Loss coefficients for the Generator (HiFi-GAN proven weights)
  """
  lambda_mel: float = 10.0     # Weight for the Mel-spectrogram L1 reconstruction loss.
  lambda_rec: float = 2.0      # Weight for the Multi-Resolution STFT loss.
  lambda_spk: float = 1.0      # Weight for the new Speaker Identity loss.
  """

  lambda_mel: float = 45.0     # Phase 1: Heavy L1 on final output
  lambda_rec: float = 3.0      # STFT (if active)
  lambda_spk: float = 1.0      # Re-enabled for Phase 2
  lambda_var: float = 0.0      # Disabled for Phase 1
  lambda_entropy: float = 0.0  # Disabled for Phase 1

  # Cross-pair training
  cross_pair_prob: float = 0.5           # Phase 2 Step 2: cross-pair training
  cross_pair_stats_weight: float = 6.0   # Balance with L1 gradient magnitude

  # Speaker classifier head at decoder bottleneck
  classifier_weight: float = 0.0         # Bottleneck CE disabled (FiLM modules inactive)
  mel_classifier_weight: float = 0.0   # Per-frame mel classifier disabled
  pooled_mel_ce_weight: float = 2.0    # Pooled mel-bias CE on speaker delta
  spk_film_ce_weight: float = 2.0      # Per-frame CE on mel_speaker delta (mel_proj_speaker)

  # Set unused loss weights to zero
  lambda_aux: float = 0.0
  lambda_adv: float = 0.0
  lambda_feat: float = 0.0

  # Training (HiFi-GAN proven hyperparameters)
  learning_rate: float = 5e-4  # Bumped from 1e-4 for faster baseline convergence
  adam_beta1: float = 0.8
  adam_beta2: float = 0.99
  lr_decay_factor: float = 0.999
  lr_decay_steps: int = 1
  batch_size: int = 32

  stft_fft_sizes:  Tuple[int, int, int] = (512, 1024, 2048)
  stft_hop_sizes:  Tuple[int, int, int] = (128, 256, 512)
  stft_win_lengths: Tuple[int, int, int] = (512, 1024, 2048)

  grad_clip: float = 1.0   # Max gradient norm — prevents instability with 2+ speakers





@dataclass
class PathConfig:
  """File paths and directories"""
  data_dir: str = "data/"
  model_dir: str = "models/"
  checkpoint_dir: str = "checkpoints/"
  vocoder_path: str = ""
  speaker_encoder_path: str = "speechbrain/spkrec-ecapa-voxceleb"

@dataclass
class InferenceConfig:
  """Inference-specific settings"""
  batch_size: int = 8
  max_length: int = 1000  # Maximum sequence length
  device: str = "cuda"  # or "cpu"
