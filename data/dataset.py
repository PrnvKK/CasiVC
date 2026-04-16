# hubertvc/data/dataset.py
import os
import torch
import json
import hashlib
import pickle
import h5py
import random
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from collections import defaultdict
from dataclasses import dataclass
import numpy as np
from torch.utils.data import Dataset
import torch.nn.functional as F   # <-- add this

# Import project modules
try:
    from data.audio_utils import (
        load_audio, extract_mel_spectrogram, split_utterance_for_training,
        process_audio_chunk
    )
    from data.alignment import align_mel_sequences, DTWAligner
    from config import AudioConfig, DataConfig, TrainingConfig
except ImportError:
    # Fallback for standalone testing
    from data.audio_utils import (
        load_audio, extract_mel_spectrogram, split_utterance_for_training,
        process_audio_chunk
    )
    from data.alignment import align_mel_sequences, DTWAligner
    from config import AudioConfig, DataConfig, TrainingConfig


@dataclass
class TrainingPair:
    """Container for a complete training pair."""
    ref_audio: Optional[torch.Tensor]         # ADD THIS - Raw reference audio for ECAPA-TDNN (samples,)
    ref_mel: Optional[torch.Tensor]           # Reference mel-spectrogram (n_mels, time_ref)
    content_audio: Optional[torch.Tensor]     # Content audio for HuBERT processing (samples,)
    content_mel: Optional[torch.Tensor]       # Content mel-spectrogram (n_mels, time_content)
    alignment_quality: float        # DTW alignment quality score
    speaker_id: str                # Speaker identifier
    utterance_id: str              # Unique utterance identifier
    ref_duration: float            # Reference segment duration (seconds)
    content_duration: float        # Content segment duration (seconds)
    content_feats: Optional[torch.Tensor] = None # Added for precomputed HuBERT
    speaker_feats: Optional[torch.Tensor] = None # Added for precomputed ECAPA


@dataclass
class CacheEntry:
    """Container for cached data."""
    audio_hash: str
    config_hash: str
    timestamp: float
    data: Any


class VoiceConversionCache:
    """
    Hierarchical caching system for voice conversion dataset.
    Supports both memory and disk caching with automatic invalidation.
    """
    
    def __init__(self, cache_dir: str, max_memory_size: int = 1000):
        """
        Initialize cache system.
        
        Args:
            cache_dir (str): Directory for disk cache
            max_memory_size (int): Maximum number of items in memory cache
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Memory cache (LRU-like behavior)
        self.memory_cache: Dict[str, CacheEntry] = {}
        self.max_memory_size = max_memory_size
        self.access_order: List[str] = []
        
        # Cache file paths
        self.audio_cache_dir = self.cache_dir / "audio"
        self.mel_cache_dir = self.cache_dir / "mel"
        self.pairs_cache_dir = self.cache_dir / "pairs"
        self.metadata_file = self.cache_dir / "metadata.json"
        
        # Create cache directories
        for cache_dir in [self.audio_cache_dir, self.mel_cache_dir, self.pairs_cache_dir]:
            cache_dir.mkdir(exist_ok=True)
        
        # Load metadata
        self.metadata = self._load_metadata()
    
    def _load_metadata(self) -> Dict[str, Any]:
        """Load cache metadata."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                warnings.warn(f"Failed to load cache metadata: {e}")
        return {"cache_version": "1.0", "entries": {}}
    
    def _save_metadata(self):
        """Save cache metadata."""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2)
        except Exception as e:
            warnings.warn(f"Failed to save cache metadata: {e}")
    
    def _compute_hash(self, data: Union[str, Dict]) -> str:
        """Compute hash for cache key."""
        if isinstance(data, str):
            return hashlib.md5(data.encode()).hexdigest()
        elif isinstance(data, dict):
            data_str = json.dumps(data, sort_keys=True)
            return hashlib.md5(data_str.encode()).hexdigest()
        else:
            data_str = str(data)
            return hashlib.md5(data_str.encode()).hexdigest()
    
    def _manage_memory_cache(self):
        """Manage memory cache size using LRU policy."""
        while len(self.memory_cache) > self.max_memory_size:
            if self.access_order:
                oldest_key = self.access_order.pop(0)
                self.memory_cache.pop(oldest_key, None)
    
    def _update_access_order(self, key: str):
        """Update access order for LRU policy."""
        if key in self.access_order:
            self.access_order.remove(key)
        self.access_order.append(key)
    
    def get_audio_cache(self, file_path: str, config_hash: str) -> Optional[torch.Tensor]:
        """Get cached audio."""
        cache_key = f"audio_{self._compute_hash(file_path)}_{config_hash}"
        
        # Check memory cache
        if cache_key in self.memory_cache:
            self._update_access_order(cache_key)
            return self.memory_cache[cache_key].data
        
        # Check disk cache
        cache_file = self.audio_cache_dir / f"{cache_key}.pt"
        if cache_file.exists():
            try:
                audio = torch.load(cache_file, map_location='cpu')
                # Store in memory cache
                entry = CacheEntry(
                    audio_hash=self._compute_hash(file_path),
                    config_hash=config_hash,
                    timestamp=cache_file.stat().st_mtime,
                    data=audio
                )
                self.memory_cache[cache_key] = entry
                self._update_access_order(cache_key)
                self._manage_memory_cache()
                return audio
            except Exception as e:
                warnings.warn(f"Failed to load cached audio {cache_file}: {e}")
        
        return None
    
    def set_audio_cache(self, file_path: str, config_hash: str, audio: torch.Tensor):
        """Set cached audio."""
        cache_key = f"audio_{self._compute_hash(file_path)}_{config_hash}"
        cache_file = self.audio_cache_dir / f"{cache_key}.pt"
        
        try:
            # Save to disk
            torch.save(audio, cache_file)
            
            # Store in memory cache
            entry = CacheEntry(
                audio_hash=self._compute_hash(file_path),
                config_hash=config_hash,
                timestamp=cache_file.stat().st_mtime,
                data=audio
            )
            self.memory_cache[cache_key] = entry
            self._update_access_order(cache_key)
            self._manage_memory_cache()
            
        except Exception as e:
            warnings.warn(f"Failed to cache audio {cache_file}: {e}")
    
    def get_training_pair_cache(self, utterance_id: str, config_hash: str) -> Optional[TrainingPair]:
        """Get cached training pair."""
        cache_key = f"pair_{self._compute_hash(utterance_id)}_{config_hash}"
        
        # Check memory cache
        if cache_key in self.memory_cache:
            self._update_access_order(cache_key)
            return self.memory_cache[cache_key].data
        
        # Check disk cache
        cache_file = self.pairs_cache_dir / f"{cache_key}.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    training_pair = pickle.load(f)
                
                # Store in memory cache
                entry = CacheEntry(
                    audio_hash=self._compute_hash(utterance_id),
                    config_hash=config_hash,
                    timestamp=cache_file.stat().st_mtime,
                    data=training_pair
                )
                self.memory_cache[cache_key] = entry
                self._update_access_order(cache_key)
                self._manage_memory_cache()
                return training_pair
                
            except Exception as e:
                warnings.warn(f"Failed to load cached training pair {cache_file}: {e}")
        
        return None
    
    def set_training_pair_cache(self, utterance_id: str, config_hash: str, training_pair: TrainingPair):
        """Set cached training pair."""
        cache_key = f"pair_{self._compute_hash(utterance_id)}_{config_hash}"
        cache_file = self.pairs_cache_dir / f"{cache_key}.pkl"
        
        try:
            # Save to disk
            with open(cache_file, 'wb') as f:
                pickle.dump(training_pair, f)
            
            # Store in memory cache
            entry = CacheEntry(
                audio_hash=self._compute_hash(utterance_id),
                config_hash=config_hash,
                timestamp=cache_file.stat().st_mtime,
                data=training_pair
            )
            self.memory_cache[cache_key] = entry
            self._update_access_order(cache_key)
            self._manage_memory_cache()
            
        except Exception as e:
            warnings.warn(f"Failed to cache training pair {cache_file}: {e}")
    
    def clear_cache(self):
        """Clear all cache."""
        import shutil
        try:
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self.memory_cache.clear()
            self.access_order.clear()
            self.metadata = {"cache_version": "1.0", "entries": {}}
        except Exception as e:
            warnings.warn(f"Failed to clear cache: {e}")


class VoiceConversionDataset(Dataset):
    """
    PyTorch Dataset for zero-shot voice conversion training.
    
    Implements the SEF-VC training strategy:
    - Single utterance split into reference (2-3s) + content segments
    - Speaker-level splits for zero-shot evaluation
    - DTW alignment between reference and content mel-spectrograms
    - Quality filtering based on alignment scores
    """
    
    def __init__(
        self,
        split: str = "train",
        audio_config: Optional[AudioConfig] = None,
        data_config: Optional[DataConfig] = None,
        training_config: Optional[TrainingConfig] = None,
        enable_caching: bool = True,
        quality_threshold: float = 0.3,
        max_items: Optional[int] = None
    ):
        """
        Initialize dataset.
        
        Args:
            split (str): Dataset split ('train', 'val', 'test')
            audio_config: Audio processing configuration
            data_config: Data loading configuration
            training_config: Training configuration
            enable_caching (bool): Whether to enable caching
            quality_threshold (float): Minimum alignment quality threshold
            max_items (int, optional): Maximum number of items (for debugging)
        """
        self.split = split
        self.audio_config = audio_config or AudioConfig()
        self.data_config = data_config or DataConfig()
        self.training_config = training_config or TrainingConfig()
        self.quality_threshold = quality_threshold
        self.max_items = max_items
        
        # Initialize caching
        self.enable_caching = enable_caching and self.data_config.enable_caching
        if self.enable_caching:
            cache_dir = os.path.join(self.data_config.cache_dir, split)
            self.cache = VoiceConversionCache(cache_dir)
        else:
            self.cache = None

        # Compute configuration hash for cache validation
        self.config_hash = self._compute_config_hash()
        
        # Discover and split speakers
        self.speakers = self._discover_speakers()
        self.split_speakers = self._split_speakers()
        
        # Get utterances for current split
        self.utterances = self._get_split_utterances()

        # Filter valid utterances
        self.valid_utterances = self._filter_valid_utterances()
        
        # When debugging with small max_items, interleave by speaker
        # so max_items grabs from diverse speakers instead of just the first one
        if self.max_items is not None:
            self.valid_utterances = self._reorder_for_speaker_diversity(self.valid_utterances)
        
        print(f"Initialized {split} dataset:")

        print(f"  Speakers: {len(self.split_speakers[split])}")
        print(f"  Total utterances: {len(self.utterances)}")
        print(f"  Valid utterances: {len(self.valid_utterances)}")
        if self.max_items:
            print(f"  Limited to: {min(self.max_items, len(self.valid_utterances))} items")
    
    def _compute_config_hash(self) -> str:
        """Compute hash of configuration for cache validation."""
        config_dict = {
            'audio': {
                'sample_rate': self.audio_config.sample_rate,
                'n_mel_bands': self.audio_config.n_mel_bands,
                'frame_shift_reference': self.audio_config.frame_shift_reference,
                'frame_shift_semantic': self.audio_config.frame_shift_semantic,
                'mel_fmin': self.audio_config.mel_fmin,
                'mel_fmax': self.audio_config.mel_fmax,
            },
            'data': {
                'ref_length_range': self.data_config.ref_length_range,
                'min_content_length': self.data_config.min_content_length,
                'min_duration': self.data_config.min_duration,
                'max_duration': self.data_config.max_duration
            },
            'quality_threshold': self.quality_threshold
        }
        config_str = json.dumps(config_dict, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()
    
    def _discover_speakers(self) -> Dict[str, List[str]]:
        """
        Discover all speakers and their utterances.
        
        Returns:
            Dict mapping speaker_id to list of utterance file paths
        """
        speakers = defaultdict(list)
        
        dataset_path = Path(self.data_config.get_dataset_path())
        
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")
        
        # LibriTTS structure: dataset_root/subset/speaker_id/chapter_id/speaker_id_chapter_id_utterance_id.wav
        if self.data_config.use_dev_subset:
            # Use dev-clean only
            subset_path = dataset_path
            if subset_path.exists():
                for speaker_dir in subset_path.iterdir():
                    if speaker_dir.is_dir() and speaker_dir.name.isdigit():
                        speaker_id = speaker_dir.name
                        for chapter_dir in speaker_dir.iterdir():
                            if chapter_dir.is_dir():
                                for audio_file in chapter_dir.glob("*.wav"):
                                    speakers[speaker_id].append(str(audio_file))
        else:
            # Use full dataset with multiple subsets
            for subset in self.data_config.train_subsets + self.data_config.val_subsets + self.data_config.test_subsets:
                subset_path = dataset_path / subset
                if subset_path.exists():
                    for speaker_dir in subset_path.iterdir():
                        if speaker_dir.is_dir() and speaker_dir.name.isdigit():
                            speaker_id = speaker_dir.name
                            for chapter_dir in speaker_dir.iterdir():
                                if chapter_dir.is_dir():
                                    for audio_file in chapter_dir.glob("*.wav"):
                                        speakers[speaker_id].append(str(audio_file))
        
        # Sort utterances for reproducibility
        for speaker_id in speakers:
            speakers[speaker_id].sort()
        
        print(f"Discovered {len(speakers)} speakers with {sum(len(utterances) for utterances in speakers.values())} utterances")
        
        return dict(speakers)
    
    def _split_speakers(self) -> Dict[str, List[str]]:
        """
        Split speakers into train/val/test based on configuration.
        
        Returns:
            Dict mapping split to list of speaker IDs
        """
        speaker_ids = list(self.speakers.keys())
        speaker_ids.sort()  # Ensure reproducible ordering
        
        # Set random seed for reproducible splits
        random.seed(self.data_config.speaker_split_seed)
        random.shuffle(speaker_ids)
        
        # Get split ratios
        train_ratio, val_ratio, test_ratio = self.data_config.get_speaker_split_ratios()
        
        total_speakers = len(speaker_ids)
        train_size = int(total_speakers * train_ratio)
        val_size = int(total_speakers * val_ratio)
        
        splits = {
            'train': speaker_ids[:train_size],
            'val': speaker_ids[train_size:train_size + val_size],
            'test': speaker_ids[train_size + val_size:]
        }
        
        print(f"Speaker splits:")
        for split_name, split_speakers in splits.items():
            print(f"  {split_name}: {len(split_speakers)} speakers")
        
        return splits
    
    def _get_split_utterances(self) -> List[Tuple[str, str]]:
        """
        Get utterances for current split.
        
        Returns:
            List of (speaker_id, utterance_path) tuples
        """
        utterances = []
        split_speakers = self.split_speakers.get(self.split, [])
        
        for speaker_id in split_speakers:
            if speaker_id in self.speakers:
                for utterance_path in self.speakers[speaker_id]:
                    utterances.append((speaker_id, utterance_path))
        
        return utterances
    
    def _filter_valid_utterances(self) -> List[Tuple[str, str]]:
        """
        Filter utterances based on duration and other criteria.
        
        Returns:
            List of valid (speaker_id, utterance_path) tuples
        """
        valid_utterances = []
        min_total_duration = max(self.data_config.ref_length_range) + self.data_config.min_content_length
        
        for speaker_id, utterance_path in self.utterances:
            try:
                # Use torchaudio.info with proper error handling for newer versions
                import torchaudio
                
                # Try the new API first (torchaudio 2.2+)
                try:
                    metadata = torchaudio.info(utterance_path)
                    duration = metadata.num_frames / metadata.sample_rate
                except AttributeError:
                    # Fallback for older torchaudio versions
                    import soundfile as sf
                    info = sf.info(utterance_path)
                    duration = info.duration
                
                # Check duration constraints
                if self.data_config.min_duration <= duration <= self.data_config.max_duration:
                    if duration >= min_total_duration:
                        valid_utterances.append((speaker_id, utterance_path))
                        
            except Exception as e:
                warnings.warn(f"Failed to validate utterance {utterance_path}: {e}")
                continue
        
        return valid_utterances
    
    def _reorder_for_speaker_diversity(self, utterances: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        """
        Reorder utterances using round-robin across speakers.
        Result: [spk_A_utt_0, spk_B_utt_0, spk_C_utt_0, spk_A_utt_1, ...]
        Ensures max_items=N picks from N different speakers when possible.
        """
        from collections import defaultdict
        speaker_buckets = defaultdict(list)
        for spk, path in utterances:
            speaker_buckets[spk].append((spk, path))
        
        speakers = list(speaker_buckets.keys())  # order from _split_speakers (seeded shuffle)
        max_utts = max(len(v) for v in speaker_buckets.values())
        
        reordered = []
        for i in range(max_utts):
            for spk in speakers:
                if i < len(speaker_buckets[spk]):
                    reordered.append(speaker_buckets[spk][i])
        
        return reordered

    
    def _load_and_process_audio(self, utterance_path: str) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]]:

      try:

        # 1. Load Audio         
        audio = load_audio(utterance_path, self.audio_config.sample_rate)

        # 2. Split Audio
        ref_audio, content_audio = split_utterance_for_training(
            audio,
            ref_length_range=self.data_config.ref_length_range,
            sample_rate=self.audio_config.sample_rate,
            min_content_length=self.data_config.min_content_length,
            deterministic=(self.max_items is not None and self.max_items <= 5),
        )


        # 3. Extract Mel (still needed for compatibility/ground truth)
        ref_mel = extract_mel_spectrogram(
            ref_audio,
            sample_rate=self.audio_config.sample_rate, 
        )

        return (
            ref_audio,      # ADD THIS - raw reference audio for ECAPA-TDNN
            ref_mel,        
            content_audio,  
            None,           
            1.0             
        )
                        
      except Exception as e:
          print(f"!!! CRITICAL FAILURE processing {utterance_path} !!!")
          print(f"Error Type: {type(e)}")
          print(f"Error Message: {e}")
          import traceback
          traceback.print_exc() # Print full stack trace
          warnings.warn(f"Failed to process audio {utterance_path}: {e}")
          return None
    
    def __len__(self) -> int:
        """Return dataset size."""
        length = len(self.valid_utterances)
        if self.max_items:
            length = min(length, self.max_items)
        return length



    def __getitem__(self, idx: int) -> TrainingPair:
        speaker_id, utterance_path = self.valid_utterances[idx]
        utterance_id = Path(utterance_path).stem

        # Try cache first
        cache_file = Path(f"/content/hubertvc_cache/{utterance_id}.pt")
        if cache_file.exists():
            try:
                cached = torch.load(cache_file, map_location='cpu')
                speaker_feats = cached.get('speaker_feats')
                expected_tokens = getattr(model_cfg, "num_speaker_tokens", 8)
                if speaker_feats is not None and speaker_feats.dim() == 2 and speaker_feats.shape[0] != expected_tokens:
                    raise ValueError(
                        f"Cache token mismatch for {utterance_id}: "
                        f"got {speaker_feats.shape[0]}, expected {expected_tokens}"
                    )
                return TrainingPair(
                    ref_audio=cached.get('ref_audio'),
                    ref_mel=None, 
                    content_audio=cached['gt_wave'],
                    content_mel=cached['gt_mel'],
                    alignment_quality=1.0,
                    speaker_id=speaker_id,
                    utterance_id=utterance_id,
                    ref_duration=0.0,
                    content_duration=0.0,
                    content_feats=cached['content_feats'],
                    speaker_feats=None
                )
            except Exception as e:
                pass # Fall through to local processing if cache corrupted

        result = self._load_and_process_audio(utterance_path)

        if result is None:
            raise RuntimeError(f"Failed to process utterance {utterance_id} from {utterance_path}")

        ref_audio, ref_mel, content_audio, _, alignment_quality = result  # UNPACK ref_audio

        training_pair = TrainingPair(
            ref_audio=ref_audio,      # ADD THIS
            ref_mel=ref_mel,
            content_audio=content_audio,
            content_mel=None,
            alignment_quality=alignment_quality,
            speaker_id=speaker_id,
            utterance_id=utterance_id,
            ref_duration=ref_mel.shape[1] * self.audio_config.frame_shift_reference / 1000.0,
            content_duration=len(content_audio) / self.audio_config.sample_rate
        )

        return training_pair


    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get dataset statistics.
        
        Returns:
            Dictionary with dataset statistics
        """
        if not hasattr(self, '_statistics'):
            self._compute_statistics()
        return self._statistics
    
    def _compute_statistics(self):
        """Compute dataset statistics."""
        durations = []
        quality_scores = []
        speakers_count = len(self.split_speakers.get(self.split, []))
        
        # Sample a subset for statistics to avoid processing entire dataset
        sample_size = min(100, len(self))
        sample_indices = random.sample(range(len(self)), sample_size)
        
        for idx in sample_indices:
            try:
                pair = self[idx]
                durations.append(pair.ref_duration + pair.content_duration)
                quality_scores.append(pair.alignment_quality)
            except Exception:
                continue
        
        self._statistics = {
            'total_utterances': len(self.valid_utterances),
            'speakers': speakers_count,
            'mean_duration': np.mean(durations) if durations else 0.0,
            'std_duration': np.std(durations) if durations else 0.0,
            'mean_alignment_quality': np.mean(quality_scores) if quality_scores else 0.0,
            'std_alignment_quality': np.std(quality_scores) if quality_scores else 0.0,
            'config_hash': self.config_hash
        }


def create_datasets(
    audio_config: Optional[AudioConfig] = None,
    data_config: Optional[DataConfig] = None,
    training_config: Optional[TrainingConfig] = None,
    quality_threshold: float = 0.3,
    max_items_per_split: Optional[int] = None
) -> Dict[str, VoiceConversionDataset]:
    """
    Create train/val/test datasets.
    
    Args:
        audio_config: Audio configuration
        data_config: Data configuration  
        training_config: Training configuration
        quality_threshold: Minimum alignment quality threshold
        max_items_per_split: Maximum items per split (for debugging)
        
    Returns:
        Dictionary mapping split names to datasets
    """
    datasets = {}
    
    for split in ['train', 'val', 'test']:
        datasets[split] = VoiceConversionDataset(
            split=split,
            audio_config=audio_config,
            data_config=data_config,
            training_config=training_config,
            quality_threshold=quality_threshold,
            max_items=max_items_per_split
        )
    
    return datasets


from config import AudioConfig, ModelConfig, TrainingConfig

audio_cfg = AudioConfig()
model_cfg = ModelConfig()

def _pad_2d(tensors: List[torch.Tensor]) -> torch.Tensor:
    """Pads a list of 2-D tensors (C, T) on the time axis to max-T."""
    if not tensors or tensors[0] is None:
        return None
    c = tensors[0].size(0)
    max_t = max(t.size(1) for t in tensors)
    
    LOG_MEL_SILENCE = 0
    
    out = []
    for t in tensors:
        if t.size(1) < max_t:
            pad = F.pad(t, (0, max_t - t.size(1)), value=LOG_MEL_SILENCE)
            out.append(pad)
        else:
            out.append(t)

    # Stack FIRST, then print shape
    return torch.stack(out)


def _pad_2d_feats(tensors: List[torch.Tensor]) -> torch.Tensor:
    """Pads a list of 2-D feature tensors (T, C)."""
    from torch.nn.utils.rnn import pad_sequence
    if not tensors or tensors[0] is None:
        return None
    return pad_sequence(tensors, batch_first=True)

def _pad_1d(tensors: List[torch.Tensor]) -> torch.Tensor:
    """Pads a list of 1-D tensors (T,) on the time axis to max-T."""
    if not tensors or tensors[0] is None:
        return None
    max_t = max(t.size(0) for t in tensors)
    out = [F.pad(t, (0, max_t - t.size(0))) for t in tensors]

    # Stack FIRST, then print shape
    return torch.stack(out)



def collate_training_pairs(batch: List["TrainingPair"]) -> Dict[str, Any]:
    """
    Custom collate fn that returns **tensors** for everything the trainer
    needs (ref_audio, ref_mel, content_audio, gt_mel, gt_wave) while keeping
    the per-sample metadata as lists.
    
    Updated to support ECAPA-TDNN encoder by including raw reference audio.
    """

    # ---- gather per-sample objects ---------------------------------
    ref_audios = []              # NEW: Raw reference audio for ECAPA-TDNN
    ref_mels = []                # Reference mel-spectrograms
    content_audios = []          # Content audio for HuBERT
    content_mels = []            # Content mel (usually None)
    content_feats_list = []
    speaker_feats_list = []
    alignment_qualities = []
    speaker_ids = []
    utterance_ids = []
    ref_durs = []
    content_durs = []

    for pair in batch:
        ref_audios.append(pair.ref_audio)           # NEW: Collect raw audio
        ref_mels.append(pair.ref_mel)
        content_audios.append(pair.content_audio)
        content_mels.append(pair.content_mel)
        content_feats_list.append(pair.content_feats)
        speaker_feats_list.append(pair.speaker_feats)
        alignment_qualities.append(pair.alignment_quality)
        speaker_ids.append(pair.speaker_id)
        utterance_ids.append(pair.utterance_id)
        ref_durs.append(pair.ref_duration)
        content_durs.append(pair.content_duration)
    
    # ---- COMPUTE LENGTHS BEFORE PADDING -----------------------------
    ref_audio_lengths = torch.tensor([a.size(0) for a in ref_audios if a is not None], dtype=torch.long)  # NEW
    ref_lengths = torch.tensor([m.size(1) for m in ref_mels if m is not None], dtype=torch.long)
    audio_lengths = torch.tensor([a.size(0) for a in content_audios if a is not None], dtype=torch.long)
    
    # ---- pad / stack ------------------------------------------------
    batched_ref_audios = _pad_1d(ref_audios)       # NEW: (B, T_ref_wav)
    batched_ref_mels = _pad_2d(ref_mels)           # (B, 80, T_ref)
    batched_audio = _pad_1d(content_audios)        # (B, T_wav)
    batched_content_feats = _pad_2d_feats(content_feats_list)
    batched_speaker_feats = torch.stack(speaker_feats_list) if speaker_feats_list[0] is not None else None

    # ---- ground-truth mel for loss ----------------------------------
    if all(m is None for m in content_mels):
        gt_mels = [
            extract_mel_spectrogram(
                wav, 
                sample_rate=audio_cfg.sample_rate,
            )
            for wav in content_audios
        ]
    else:
        gt_mels = [
            m if m is not None else extract_mel_spectrogram(
                wav, 
                sample_rate=audio_cfg.sample_rate,
            )
            for m, wav in zip(content_mels, content_audios)
        ]
    
    # Debug: Mel extraction parameters
    print(f"[Consistency Check] Mel extraction parameters:")
    print(f"  sample_rate: {audio_cfg.sample_rate}")
    print(f"  hop_length: 256")
    print(f"  win_length: 1024")
    print(f"  n_fft: 1024")
    print(f"  n_mels: {audio_cfg.n_mel_bands}")
    print(f"  f_min: {audio_cfg.mel_fmin}")
    print(f"  f_max: {audio_cfg.mel_fmax}")
    print(f"  center: False")

    # Debug: GT mel shapes
    for i, gt_mel in enumerate(gt_mels):
        print(f"  GT mel {i}: shape={gt_mel.shape}")
    
    batched_gt_mels = _pad_2d(gt_mels)              # (B, 80, T_gt)
  
    # ---- COMPUTE GT LENGTHS AFTER MEL EXTRACTION --------------------
    gt_lengths = torch.tensor([m.size(1) for m in gt_mels], dtype=torch.long)

    # -----------------------------------------------------------------
    # RETURN COMPLETE BATCH DICTIONARY
    # -----------------------------------------------------------------
    return_dict = {
        # ---- AUDIO/MEL TENSORS --------------------------------------
        "ref_audio":         batched_ref_audios,       # NEW: (B, T_ref_wav) - for ECAPA-TDNN
        "ref_mel":           batched_ref_mels,         # (B, 80, T_ref) - kept for compatibility
        "content_audio":     batched_audio,            # (B, T_wav) - for HuBERT
        "content_feats":     batched_content_feats,    # (B, T, 96) - precomputed HuBERT
        "speaker_feats":     batched_speaker_feats,    # (B, 64, 96) - precomputed ECAPA
        "gt_mel":            batched_gt_mels,          # (B, 80, T_gt) - for reconstruction loss
        "gt_wave":           batched_audio,            # (B, T_wav) - for vocoder loss
        
        # ---- LENGTH TENSORS FOR MASKING -----------------------------
        "ref_audio_lengths": ref_audio_lengths,        # NEW: (B,) - samples in ref_audio
        "ref_lengths":       ref_lengths,              # (B,) - frames in ref_mel
        "gt_lengths":        gt_lengths,               # (B,) - frames in gt_mel
        "audio_lengths":     audio_lengths,            # (B,) - samples in content_audio
        
        # ---- METADATA -----------------------------------------------
        "content_mel":       content_mels,             # list – untouched (usually None)
        "alignment_quality": torch.tensor(alignment_qualities, dtype=torch.float32),
        "speaker_id":        speaker_ids,              # list of speaker IDs
        "utterance_id":      utterance_ids,            # list of utterance IDs
        "ref_duration":      torch.tensor(ref_durs, dtype=torch.float32),
        "content_duration":  torch.tensor(content_durs, dtype=torch.float32),
    }
    
    return return_dict




# Standalone testing
if __name__ == "__main__":
    print("Running dataset.py.....")

