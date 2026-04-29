import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any, Union, List
import warnings
import math
import matplotlib.pyplot as plt
import os
import torch
from typing import Optional

try:
    from config import ModelConfig, AudioConfig
    model_config = ModelConfig()
    audio_config = AudioConfig()
except ImportError:
    print("Error....")



class PositionAgnosticCrossAttention(nn.Module):
    """
    Position-agnostic cross-attention mechanism for zero-shot voice conversion.
    
    Fuses HuBERT continuous semantic features (content) with mel encoder 
    speaker tokens (speaker characteristics) without positional encoding.
    
    Key Design:
    - Content features (HuBERT) serve as queries - preserve temporal structure
    - Speaker tokens serve as keys/values - position-agnostic representation
    - No positional encoding anywhere - following SEF-VC principle
    - Single fusion point for efficient content-speaker interaction
    """
    
    def __init__(
        self,
        enable_residual: bool = True,
        enable_output_projection: bool = True
    ):
        """
        Initialize position-agnostic cross-attention layer.
        
        Args:
            content_dim: HuBERT content feature dimension (default: 1024)
            speaker_dim: Speaker token dimension from mel encoder (default: 768)
            d_model: Model dimension for attention (default: 512)
            num_heads: Number of attention heads (default: 8)
            dropout: Dropout rate (default: 0.1)
            enable_residual: Whether to use residual connections
            enable_output_projection: Whether to apply output projection
        """
        super().__init__()
        
        
        # Configuration with fallbacks
        self.content_dim = 96
        #self.content_dim = getattr(model_config, 'hubert_features_dim') 
        self.speaker_dim = getattr(model_config, 'speaker_projection_dim')
        self.d_model = getattr(model_config, 'cross_attention_dim')
        self.num_heads = getattr(model_config, 'cross_attention_heads') 
        self.dropout_rate = getattr(model_config, 'cross_attention_dropout')

        self.enable_output_projection = enable_output_projection
                
        # Validate dimensions
        if self.d_model % self.num_heads != 0:
            raise ValueError(f"d_model ({self.d_model}) must be divisible by num_heads ({self.num_heads})")
        
        self.head_dim = self.d_model // self.num_heads

        #self.alpha = nn.Parameter(torch.tensor(0.1))
        self.alpha = nn.Parameter(torch.tensor(2.0))

        # Learnable attention temperature (softplus-constrained, starts ≈0.5 for moderate sharpening)
        self.raw_temperature = nn.Parameter(torch.tensor(-0.432))  # F.softplus(-0.432) ≈ 0.5

        # Multi-head attention (speaker features already at correct dimension)
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=self.d_model,
            num_heads=self.num_heads,
            dropout=self.dropout_rate,
            batch_first=True,
            bias=True
        )
        
        # Residual connection projection
        # Residual connection
        self.enable_residual = enable_residual          # keep the flag
        if self.enable_residual:
            if self.content_dim == self.d_model:        # no channel mismatch
                self.residual_proj = nn.Identity()      # 0 params
            else:                                       # mismatch → learnable 1×1
                self.residual_proj = nn.Linear(self.content_dim, self.d_model)

        self.content_proj = nn.Linear(self.content_dim, self.d_model)
        self.speaker_proj = nn.Linear(self.speaker_dim, self.d_model)   # keys
        self.speaker_val_proj = nn.Linear(self.speaker_dim, self.d_model)  # values (separate!)
        
        # Optional output projection  
        if self.enable_output_projection:
            self.output_proj = nn.Linear(self.d_model, self.d_model)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(self.d_model)
        
        # Dropout
        self.dropout = nn.Dropout(self.dropout_rate)
        
        # AdaIN Mapping Network
        # Converts the abstract target speaker token into independent scale (gamma) 
        # and shift (beta) statistics per channel.
        self.mapping_network = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.LeakyReLU(0.2),
            nn.Linear(self.d_model, self.d_model * 2)
        )
        
        # Initialize parameters
        self._init_parameters()
        
            
    def _init_parameters(self):
        """Initialize parameters to preserve semantic relationships."""

        nn.init.xavier_uniform_(self.content_proj.weight, gain=1.0)
        nn.init.zeros_(self.content_proj.bias)

        nn.init.xavier_uniform_(self.speaker_proj.weight, gain=0.5)
        nn.init.zeros_(self.speaker_proj.bias)

        nn.init.xavier_uniform_(self.speaker_val_proj.weight, gain=1.0)
        nn.init.zeros_(self.speaker_val_proj.bias)


        # Residual projection (if used)
        if self.enable_residual and isinstance(self.residual_proj, nn.Linear):
            nn.init.xavier_uniform_(self.residual_proj.weight)
            nn.init.zeros_(self.residual_proj.bias)

        # Output projection (optional)
        if self.enable_output_projection:
            nn.init.xavier_uniform_(self.output_proj.weight)
            nn.init.zeros_(self.output_proj.bias)

        # Mapping network initialization
        for m in self.mapping_network.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)


    

    @property
    def config(self) -> Dict[str, Any]:
        """Get runtime configuration for dynamic testing."""
        return {
            'content_dim': self.content_dim,
            'speaker_dim': self.speaker_dim, 
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'head_dim': self.head_dim,
            'output_dim': self.d_model,  # Alias for clarity
            'speaker_tokens': 64,  # Fixed from mel encoder
            'attention_dim': self.d_model
        }

    def get_integration_info(self) -> Dict[str, Any]:
        """Get information for module integration."""
        return {
            'expected_content_shape': f"(batch, time, {self.content_dim})",
            'expected_speaker_shape': f"(batch, 64, {self.speaker_dim})",
            'output_shape': f"(batch, time, {self.d_model})",
            'parameter_count': self.get_parameter_count()['total']
        }



    def validate_inputs(
        self, 
        content_features: torch.Tensor, 
        speaker_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Validate and preprocess input tensors.
        
        Args:
            content_features: HuBERT content features [B, T_content, content_dim]
            speaker_features: Speaker tokens [B, num_tokens, speaker_dim]
            
        Returns:
            Validated content and speaker features
            
        Raises:
            ValueError: If input dimensions are incorrect
        """
        # Validate content features
        if not isinstance(content_features, torch.Tensor):
            raise ValueError("Content features must be a torch.Tensor")
        
        if content_features.dim() != 3:
            raise ValueError(f"Content features must be 3D [B, T, D], got {content_features.dim()}D")
        
        if content_features.shape[2] != self.content_dim:
            raise ValueError(f"Content features dimension mismatch: expected {self.content_dim}, got {content_features.shape[2]}")
        
        # Validate speaker features
        if not isinstance(speaker_features, torch.Tensor):
            raise ValueError("Speaker features must be a torch.Tensor")
        
        if speaker_features.dim() != 3:
            raise ValueError(f"Speaker features must be 3D [B, num_tokens, D], got {speaker_features.dim()}D")
        
        if speaker_features.shape[2] != self.speaker_dim:
            raise ValueError(f"Speaker features dimension mismatch: expected {self.speaker_dim}, got {speaker_features.shape[2]}")
        
        # Validate batch sizes match
        if content_features.shape[0] != speaker_features.shape[0]:
            raise ValueError(f"Batch size mismatch: content {content_features.shape[0]} vs speaker {speaker_features.shape[0]}")
        
        # Check for NaN/Inf values
        if torch.isnan(content_features).any():
            raise ValueError("Content features contain NaN values")
        
        if torch.isinf(content_features).any():
            raise ValueError("Content features contain Inf values")
        
        if torch.isnan(speaker_features).any():
            raise ValueError("Speaker features contain NaN values")
        
        if torch.isinf(speaker_features).any():
            raise ValueError("Speaker features contain Inf values")
        
        return content_features, speaker_features
    
    def forward(self, content_features, speaker_features, return_attention=False):
        """
        Forward pass of position-agnostic cross-attention.
        """
        print("\n" + "=" * 80)
        print("[cross_attn] >>> ENTERING CROSS ATTENTION FORWARD <<<")

        # Input validation
        content_features, speaker_features = self.validate_inputs(content_features, speaker_features)

        batch_size, time_content, _ = content_features.shape

        print(f"[cross_attn] content_features shape: {content_features.shape}")
        print(f"[cross_attn] speaker_features shape: {speaker_features.shape}")

        print(f"[cross_attn] content stats: mean={content_features.mean():.4f}, "
              f"std={content_features.std():.4f}, "
              f"min={content_features.min():.4f}, "
              f"max={content_features.max():.4f}")

        print(f"[cross_attn] speaker stats: mean={speaker_features.mean():.4f}, "
              f"std={speaker_features.std():.4f}, "
              f"min={speaker_features.min():.4f}, "
              f"max={speaker_features.max():.4f}")

        # ------------------------------------------------------------------
        # Projection
        # ------------------------------------------------------------------
        queries = self.content_proj(content_features)
        keys    = self.speaker_proj(speaker_features)
        values  = self.speaker_val_proj(speaker_features)  # separate projection

        # Check query diversity (debug — always use first batch item only)
        queries_sample = queries[0]  # [T, 96] — safe for any batch size

        # Compute pairwise distances between queries
        query_diffs = queries_sample.unsqueeze(0) - queries_sample.unsqueeze(1)  # [T, T, 96]
        query_distances = torch.norm(query_diffs, dim=-1)  # [T, T]

        num_frames = query_distances.shape[0]
        mask = ~torch.eye(num_frames, dtype=torch.bool, device=queries.device)

        off_diag_distances = query_distances[mask]

        print(f"[DEBUG] Query distances: mean={off_diag_distances.mean():.4f}, "
              f"min={off_diag_distances.min():.4f}, "
              f"max={off_diag_distances.max():.4f}")

        # Check alignment between query and key spaces (first batch item only)
        avg_query = queries_sample.mean(dim=0, keepdim=True)  # [1, 96]
        avg_key = keys[0].mean(dim=0, keepdim=True)           # [1, 96]

        cos_sim = F.cosine_similarity(avg_query, avg_key, dim=-1)
        print(f"[DEBUG] Query-Key space alignment (cosine): {cos_sim.item():.4f}")
        

        print(f"[cross_attn] queries shape: {queries.shape}")
        print(f"[cross_attn] keys shape: {keys.shape}")

        print(f"[cross_attn] queries stats: mean={queries.mean():.4f}, std={queries.std():.4f}")
        print(f"[cross_attn] keys stats: mean={keys.mean():.4f}, std={keys.std():.4f}")

        # ------------------------------------------------------------------
        # Raw attention scores (diagnostic only)
        # ------------------------------------------------------------------
        scores = torch.matmul(queries, keys.transpose(-2, -1)) / math.sqrt(self.head_dim)

        print(f"[cross_attn] attention scores shape: {scores.shape}")
        print(f"[cross_attn] attention scores stats: mean={scores.mean():.4f}, "
              f"std={scores.std():.4f}, "
              f"min={scores.min():.4f}, "
              f"max={scores.max():.4f}")

        attn_probs = F.softmax(scores, dim=-1)
        entropy = -(attn_probs * torch.log(attn_probs + 1e-9)).sum(-1).mean()

        print(f"[cross_attn] attention probs std: {attn_probs.std():.4f}")
        print(f"[cross_attn] attention entropy: {entropy:.4f} "
              f"(uniform≈{math.log(attn_probs.shape[-1]):.2f})")

        # ------------------------------------------------------------------
        # MultiheadAttention (with Learnable Temperature Scaling)
        # ------------------------------------------------------------------
        temperature = F.softplus(self.raw_temperature) + 0.01  # learnable, starts ≈0.5, ensures >0.01
        queries_scaled = queries / temperature
        
        print(f"[cross_attn] attention temperature: {temperature.item():.4f}")
        
        attended_features, attention_weights = self.multihead_attn(
            query=queries_scaled,
            key=keys,
            value=values,
            need_weights=True,
            average_attn_weights=True
        )

        print(f"[cross_attn] attended_features shape: {attended_features.shape}")
        print(f"[cross_attn] attended_features stats: mean={attended_features.mean():.4f}, "
              f"std={attended_features.std():.4f}")

        print(f"[cross_attn] attention_weights shape: {attention_weights.shape}")
        print(f"[cross_attn] attention_weights stats: mean={attention_weights.mean():.4f}, "
              f"std={attention_weights.std():.4f}")

        # ------------------------------------------------------------------
        # Dropout
        # ------------------------------------------------------------------
        if self.training:
            attended_features = self.dropout(attended_features)
            print("[cross_attn] dropout applied")

        # ------------------------------------------------------------------
        # AdaIN-style Fusion
        # ------------------------------------------------------------------
        if self.enable_residual:
            residual = self.residual_proj(content_features)
            
            # STEP 1: ERASING THE SOURCE VOICE
            if self.training:
                residual = F.dropout(residual, p=0.4, training=True)
                
            residual_norm = F.instance_norm(residual.transpose(1, 2)).transpose(1, 2)
            
            # STEP 2: INJECTING THE TARGET VOICE via MAPPING NETWORK
            # Instead of manually scaling alpha, we let a non-linear mapping 
            # network decipher the spatial ECAPA token into formal mean/std channel statistics.
            pooled_spk = speaker_features.mean(dim=1, keepdim=True) # [B, 1, 96]
            style_stats = self.mapping_network(pooled_spk)          # [B, 1, 192]
            
            gamma, beta = style_stats.chunk(2, dim=-1)              # [B, 1, 96] each
            
            # Apply AdaIN: y = (x - mean)/std * gamma + beta
            # ADD to attended_features instead of overwriting!
            print(f"[cross_attn] FiLM gamma: mean={gamma.mean():.4f}, std={gamma.std():.4f}, "
                  f"min={gamma.min():.4f}, max={gamma.max():.4f}")
            attended_features = attended_features + (residual_norm * (1.0 + gamma) + beta)

        print("[cross_attn] <<< EXITING CROSS ATTENTION FORWARD >>>")
        print("=" * 80 + "\n")

        if self.training and self.alpha.grad is not None:
            print(f"[cross_attn] alpha grad: {self.alpha.grad.item():.6f}")
        else:
            print(f"[cross_attn] alpha grad: None")

        if return_attention:
            return attended_features, attention_weights
        return attended_features



def create_cross_attention(**kwargs) -> PositionAgnosticCrossAttention:
    """
    Factory function to create cross-attention layer.
    
    Args:
        **kwargs: Arguments passed to PositionAgnosticCrossAttention
        
    Returns:
        Initialized PositionAgnosticCrossAttention
    """
    return PositionAgnosticCrossAttention(**kwargs)

def batch_cross_attention(
    cross_attn: PositionAgnosticCrossAttention,
    content_list: List[torch.Tensor],
    speaker_list: List[torch.Tensor],
    device: Optional[torch.device] = None
) -> List[torch.Tensor]:
    """
    Apply cross-attention to batch of variable-length sequences.
    
    Args:
        cross_attn: Cross-attention layer
        content_list: List of content feature tensors
        speaker_list: List of speaker feature tensors  
        device: Target device
        
    Returns:
        List of fused feature tensors
    """
    if device is not None:
        cross_attn = cross_attn.to(device)
    
    results = []
    for content, speaker in zip(content_list, speaker_list):
        if device is not None:
            content = content.to(device)
            speaker = speaker.to(device)
        
        # Add batch dimension if needed
        if content.dim() == 2:
            content = content.unsqueeze(0)
        if speaker.dim() == 2:
            speaker = speaker.unsqueeze(0)
        
        fused = cross_attn(content, speaker)
        
        # Remove batch dimension if added
        if fused.shape[0] == 1:
            fused = fused.squeeze(0)
        
        results.append(fused)
    
    return results

# Testing and validation
if __name__ == "__main__":
    print("Cross attention ..... ")
