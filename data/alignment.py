# hubertvc/data/alignment.py
import os
import torch
import numpy as np
from typing import Tuple, Optional, Dict, List
import warnings
from dataclasses import dataclass
import matplotlib.pyplot as plt


# Import config with fallback
try:
    from config import AudioConfig, TrainingConfig
    audio_config = AudioConfig()
    training_config = TrainingConfig()
except ImportError:
    warnings.warn("Could not import configs!!!!!")

@dataclass
class DTWResult:
    """Container for DTW alignment results."""
    alignment_path: np.ndarray  # Shape: (path_length, 2) - (i, j) indices
    dtw_cost: float            # Total DTW cost
    normalized_cost: float     # Cost normalized by path length
    is_good_alignment: bool    # Quality assessment
    cumulative_cost_matrix: Optional[np.ndarray] = None  # For debugging


@dataclass
class AlignedPair:
    """Container for aligned sequence pair."""
    sequence1_aligned: torch.Tensor  # First sequence after alignment
    sequence2_aligned: torch.Tensor  # Second sequence after alignment
    alignment_quality: float         # Quality score [0, 1]
    alignment_method: str           # Method used: 'dtw' or 'linear'
    dtw_result: Optional[DTWResult] = None


@dataclass
class AlignmentStats:
    """Statistics from batch alignment processing."""
    total_pairs: int
    successful_alignments: int
    failed_alignments: int
    mean_alignment_quality: float
    quality_distribution: Dict[str, int]  # 'good', 'fair', 'poor'
    mean_dtw_cost: float
    processing_time: float


class DTWAligner:
    """
    Dynamic Time Warping aligner with Sakoe-Chiba band constraint.
    Optimized for mel-spectrogram alignment in voice conversion.
    """
    
    def __init__(
        self,
        band_radius_pct: float = 0.15,
        distance_metric: str = 'euclidean',
        good_threshold: float = 0.5,
        poor_threshold: float = 1.0
    ):
        """
        Initialize DTW aligner.
        
        Args:
            band_radius_pct (float): Sakoe-Chiba band radius as percentage of sequence length
            distance_metric (str): Distance metric ('euclidean')
            good_threshold (float): Normalized cost threshold for good alignment
            poor_threshold (float): Normalized cost threshold for poor alignment
        """
        self.band_radius_pct = band_radius_pct
        self.distance_metric = distance_metric
        self.good_threshold = good_threshold
        self.poor_threshold = poor_threshold
        
        if distance_metric != 'euclidean':
            raise ValueError("Currently only 'euclidean' distance is supported")
    
    def _normalize_sequence(self, sequence: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Apply z-score normalization per utterance.
        
        Args:
            sequence (torch.Tensor): Input sequence (features, time)
            
        Returns:
            Tuple[torch.Tensor, Dict]: Normalized sequence and statistics
        """
        if sequence.ndim != 2:
            raise ValueError("Sequence must be 2D tensor (features, time)")
        
        # Flatten for global statistics across all features and time
        flat_seq = sequence.flatten()
        
        mean_val = flat_seq.mean()
        std_val = flat_seq.std()
        

        if torch.isnan(mean_val) or torch.isinf(mean_val):
            raise ValueError(f"Mean computation failed: mean={mean_val.item()}")
        if torch.isnan(std_val) or torch.isinf(std_val):
            raise ValueError(f"Std computation failed: std={std_val.item()}")
    
        # Handle zero std case
        if std_val < 1e-8:
            warnings.warn("Very low standard deviation detected, using mean centering only")
            normalized = sequence - mean_val
            std_val_item = 1.0  # Store as float
        else:
            normalized = (sequence - mean_val) / std_val
            std_val_item = std_val.item()
        

        if torch.isnan(std_val) or torch.isinf(std_val):
            raise ValueError(f"Std computation failed: std={std_val.item()}")

        stats = {
            'mean': mean_val.item(),
            'std': std_val_item
        }
        
        return normalized, stats
    
    def _compute_distance_matrix(self, seq1: torch.Tensor, seq2: torch.Tensor) -> np.ndarray:
        if seq1.shape[0] != seq2.shape[0]:
            raise ValueError(f"Feature dimension mismatch: {seq1.shape[0]} vs {seq2.shape[0]}")
            
        # Transpose to (Batch, Time, Features) for torch.cdist
        # We add a batch dimension of 1 using unsqueeze(0)
        s1 = seq1.transpose(0, 1).unsqueeze(0)  # Shape: (1, time1, features)
        s2 = seq2.transpose(0, 1).unsqueeze(0)  # Shape: (1, time2, features)
        
        # Compute on GPU
        # Result shape: (1, time1, time2) -> squeeze to (time1, time2)
        dist_mat = torch.cdist(s1, s2, p=2.0).squeeze(0)
        
        # ONLY move to CPU/Numpy at the very end
        return dist_mat.detach().cpu().numpy()
    
    def _compute_sakoe_chiba_band(self, n1: int, n2: int) -> np.ndarray:
        """
        Compute Sakoe-Chiba band constraint mask.
        
        Args:
            n1 (int): Length of first sequence
            n2 (int): Length of second sequence
            
        Returns:
            np.ndarray: Boolean mask for valid cells (n1, n2)
        """
        max_length = max(n1, n2)
        band_radius = int(np.ceil(self.band_radius_pct * (max_length - 1)))
        
        mask = np.zeros((n1, n2), dtype=bool)
        
        for i in range(n1):
            for j in range(n2):
                # Compute distance from diagonal
                diagonal_pos = i * (n2 - 1) / (n1 - 1) if n1 > 1 else j
                distance_from_diagonal = abs(j - diagonal_pos)
                
                if distance_from_diagonal <= band_radius:
                    mask[i, j] = True
        
        return mask
    
    def _dtw_with_band(self, distance_matrix: np.ndarray, band_mask: np.ndarray) -> DTWResult:
        """
        Perform DTW with Sakoe-Chiba band constraint.
        
        Args:
            distance_matrix (np.ndarray): Pairwise distance matrix
            band_mask (np.ndarray): Valid cells mask
            
        Returns:
            DTWResult: Alignment result
        """
        n1, n2 = distance_matrix.shape
        
        # Initialize cumulative cost matrix
        cost_matrix = np.full((n1, n2), np.inf)
        cost_matrix[0, 0] = distance_matrix[0, 0] if band_mask[0, 0] else np.inf
        
        # Fill first row and column within band
        for i in range(1, n1):
            if band_mask[i, 0]:
                cost_matrix[i, 0] = cost_matrix[i-1, 0] + distance_matrix[i, 0]
        
        for j in range(1, n2):
            if band_mask[0, j]:
                cost_matrix[0, j] = cost_matrix[0, j-1] + distance_matrix[0, j]
        
        # Fill cost matrix with band constraint
        for i in range(1, n1):
            for j in range(1, n2):
                if band_mask[i, j]:
                    candidates = [
                        cost_matrix[i-1, j],      # Insertion
                        cost_matrix[i, j-1],      # Deletion
                        cost_matrix[i-1, j-1]     # Match
                    ]
                    cost_matrix[i, j] = min(candidates) + distance_matrix[i, j]
        
        # Check if path exists
        if np.isinf(cost_matrix[n1-1, n2-1]):
            raise RuntimeError("No valid DTW path found within Sakoe-Chiba band")
        
        # Backtrack to find optimal path
        path = self._backtrack_path(cost_matrix, band_mask)
        
        # Compute results
        dtw_cost = cost_matrix[n1-1, n2-1]
        path_length = len(path)
        normalized_cost = dtw_cost / path_length if path_length > 0 else np.inf
        is_good = bool(normalized_cost < self.good_threshold)
        
        return DTWResult(
            alignment_path=np.array(path),
            dtw_cost=dtw_cost,
            normalized_cost=normalized_cost,
            is_good_alignment=is_good,
            cumulative_cost_matrix=cost_matrix
        )
    
    def _backtrack_path(self, cost_matrix: np.ndarray, band_mask: np.ndarray) -> List[Tuple[int, int]]:
        """
        Backtrack to find optimal alignment path.
        
        Args:
            cost_matrix (np.ndarray): Cumulative cost matrix
            band_mask (np.ndarray): Valid cells mask
            
        Returns:
            List[Tuple[int, int]]: Alignment path as list of (i, j) indices
        """
        n1, n2 = cost_matrix.shape
        path = []
        i, j = n1 - 1, n2 - 1
        
        while i > 0 or j > 0:
            path.append((i, j))
            
            if i == 0:
                j -= 1
            elif j == 0:
                i -= 1
            else:
                # Find minimum cost predecessor within band
                candidates = []
                if band_mask[i-1, j]:
                    candidates.append((cost_matrix[i-1, j], i-1, j))
                if band_mask[i, j-1]:
                    candidates.append((cost_matrix[i, j-1], i, j-1))
                if band_mask[i-1, j-1]:
                    candidates.append((cost_matrix[i-1, j-1], i-1, j-1))
                
                if not candidates:
                    raise RuntimeError("Backtracking failed: no valid predecessors")
                
                _, i, j = min(candidates)
        
        path.append((0, 0))
        return list(reversed(path))
    
    def align_sequences(
        self,
        seq1: torch.Tensor,
        seq2: torch.Tensor,
        normalize: bool = False
    ) -> AlignedPair:
        """
        Align two sequences using DTW with fallback to linear interpolation.
        
        Args:
            seq1 (torch.Tensor): First sequence (features, time1)
            seq2 (torch.Tensor): Second sequence (features, time2)
            normalize (bool): Whether to apply z-score normalization
            
        Returns:
            AlignedPair: Aligned sequences and metadata
        """
        if seq1.shape[0] != seq2.shape[0]:
            raise ValueError("Sequences must have same number of features")

        # Add NaN/Inf validation
        if torch.isnan(seq1).any() or torch.isnan(seq2).any():
            raise ValueError("Input sequences contain NaN values")
        if torch.isinf(seq1).any() or torch.isinf(seq2).any():
            raise ValueError("Input sequences contain Inf values")
        
        try:
            # Preprocessing
            if normalize:
                seq1_norm, _ = self._normalize_sequence(seq1)
                seq2_norm, _ = self._normalize_sequence(seq2)
            else:
                seq1_norm, seq2_norm = seq1, seq2
            
            # Compute distance matrix
            distance_matrix = self._compute_distance_matrix(seq1_norm, seq2_norm)
            
            # Compute Sakoe-Chiba band
            n1, n2 = distance_matrix.shape
            band_mask = self._compute_sakoe_chiba_band(n1, n2)
            
            # Perform DTW
            dtw_result = self._dtw_with_band(distance_matrix, band_mask)
            
            # Check alignment quality
            if dtw_result.normalized_cost > self.poor_threshold:
                warnings.warn(f"Poor DTW alignment (cost={dtw_result.normalized_cost:.3f})")
                raise ValueError(f"DTW alignment quality unacceptable (cost={dtw_result.normalized_cost:.3f})")

            
            # Apply alignment using original (non-normalized) sequences
            aligned_seq1, aligned_seq2 = self._apply_alignment(seq1, seq2, dtw_result.alignment_path)
            
            # Compute quality score
            quality = max(0.0, min(1.0, (self.poor_threshold - dtw_result.normalized_cost) / self.poor_threshold))
            
            return AlignedPair(
                sequence1_aligned=aligned_seq1,
                sequence2_aligned=aligned_seq2,
                alignment_quality=quality,
                alignment_method='dtw',
                dtw_result=dtw_result
            )
            
        except Exception as e:
            warnings.warn(f"DTW alignment failed: {str(e)}")
    
    def _apply_alignment(
        self,
        seq1: torch.Tensor,
        seq2: torch.Tensor,
        alignment_path: np.ndarray
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        """
        Apply DTW alignment path to sequences using vectorized indexing.
        """
        
        # Convert numpy indices to torch tensors on the correct device
        i_indices = torch.from_numpy(alignment_path[:, 0]).long().to(seq1.device)
        j_indices = torch.from_numpy(alignment_path[:, 1]).long().to(seq2.device)
        
        # Vectorized selection (Instantaneous vs O(N) loop)
        aligned_seq1 = seq1[:, i_indices]
        aligned_seq2 = seq2[:, j_indices]
        
        return aligned_seq1, aligned_seq2
    

def align_mel_sequences(
    ref_mel: torch.Tensor,
    content_mel: torch.Tensor,
    band_radius_pct: float = 0.15,
    normalize: bool = False
) -> AlignedPair:
    """
    High-level function to align two mel-spectrogram sequences.
    
    Args:
        ref_mel (torch.Tensor): Reference mel-spectrogram (n_mels, time1)
        content_mel (torch.Tensor): Content mel-spectrogram (n_mels, time2)
        band_radius_pct (float): Sakoe-Chiba band radius percentage
        normalize (bool): Whether to apply z-score normalization
        
    Returns:
        AlignedPair: Aligned mel-spectrograms
    """
    aligner = DTWAligner(band_radius_pct=band_radius_pct)
    return aligner.align_sequences(ref_mel, content_mel, normalize=normalize)


def process_training_pairs(
    mel_pairs: List[Tuple[torch.Tensor, torch.Tensor]],
    output_dir: Optional[str] = None,
    quality_threshold: float = 0.3
) -> Tuple[List[AlignedPair], AlignmentStats]:  # <-- Return BOTH
    import time
    start_time = time.time()
    
    aligner = DTWAligner()
    successful = 0
    failed = 0
    quality_scores = []
    dtw_costs = []
    quality_distribution = {'good': 0, 'fair': 0, 'poor': 0}
    
    # NEW: Store aligned pairs
    aligned_pairs = []
    
    for i, (ref_mel, content_mel) in enumerate(mel_pairs):
        try:
            aligned_pair = aligner.align_sequences(ref_mel, content_mel)
            
            quality_scores.append(aligned_pair.alignment_quality)
            
            if aligned_pair.dtw_result:
                dtw_costs.append(aligned_pair.dtw_result.normalized_cost)
            
            # Categorize quality
            if aligned_pair.alignment_quality > 0.7:
                quality_distribution['good'] += 1
            elif aligned_pair.alignment_quality > 0.4:
                quality_distribution['fair'] += 1
            else:
                quality_distribution['poor'] += 1
            
            if aligned_pair.alignment_quality >= quality_threshold:
                successful += 1
                # NEW: Store the aligned pair if it meets quality threshold
                aligned_pairs.append(aligned_pair)
            else:
                failed += 1
                
            # Save visualization if requested
            if output_dir and i < 10:
                _save_alignment_visualization(aligned_pair, output_dir, f"alignment_{i:03d}.png")
                
        except Exception as e:
            warnings.warn(f"Failed to process pair {i}: {str(e)}")
            failed += 1
    
    processing_time = time.time() - start_time
    
    stats = AlignmentStats(
        total_pairs=len(mel_pairs),
        successful_alignments=successful,
        failed_alignments=failed,
        mean_alignment_quality=np.mean(quality_scores) if quality_scores else 0.0,
        quality_distribution=quality_distribution,
        mean_dtw_cost=np.mean(dtw_costs) if dtw_costs else 0.0,
        processing_time=processing_time
    )

    return aligned_pairs, stats



def _save_alignment_visualization(
    aligned_pair: AlignedPair,
    output_dir: str,
    filename: str
) -> None:
    """
    Save alignment visualization for debugging.
    
    Args:
        aligned_pair (AlignedPair): Alignment result
        output_dir (str): Output directory
        filename (str): Output filename
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if aligned_pair.dtw_result and aligned_pair.dtw_result.cumulative_cost_matrix is not None:
        plt.figure(figsize=(10, 6))
        
        # Plot cost matrix
        plt.subplot(1, 2, 1)
        plt.imshow(aligned_pair.dtw_result.cumulative_cost_matrix.T, 
                  origin='lower', aspect='auto', cmap='viridis')
        plt.title(f'DTW Cost Matrix\nCost: {aligned_pair.dtw_result.normalized_cost:.3f}')
        plt.xlabel('Sequence 1 Time')
        plt.ylabel('Sequence 2 Time')
        
        # Plot alignment path
        path = aligned_pair.dtw_result.alignment_path
        plt.plot(path[:, 0], path[:, 1], 'r-', linewidth=2, label='Alignment Path')
        plt.legend()
        
        # Plot aligned sequences
        plt.subplot(1, 2, 2)
        seq1_mean = aligned_pair.sequence1_aligned.mean(dim=0).cpu().numpy()
        seq2_mean = aligned_pair.sequence2_aligned.mean(dim=0).cpu().numpy()
        
        plt.plot(seq1_mean, label='Sequence 1 (aligned)', alpha=0.7)
        plt.plot(seq2_mean, label='Sequence 2 (aligned)', alpha=0.7)
        plt.title(f'Aligned Sequences\nQuality: {aligned_pair.alignment_quality:.3f}')
        plt.xlabel('Aligned Time')
        plt.ylabel('Mean Feature Value')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, filename), dpi=150, bbox_inches='tight')
        plt.close()


# Standalone testing
if __name__ == "__main__":
  print("This is alignment.py ..........")

