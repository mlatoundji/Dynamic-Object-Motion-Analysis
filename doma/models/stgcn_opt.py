import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union
from enum import Enum
from doma.models.stgcn import STGCN, STBlock


@dataclass(frozen=True)
class ModelConfig:
    num_classes: int
    num_keypoints: int = 21
    temporal_length: int = 478
    dropout: float = 0.5
    fusion_type: Union[str, "FusionType"] = "concat"
    use_lstm_for_track: bool = False
    use_lstm_for_optflow: bool = False


class FusionType(Enum):
    """Fusion types for combining branches"""
    CONCAT = "concat"
    WEIGHTED = "weighted"
    ATTENTION = "attention"
    GATED = "gated"


class BranchType(Enum):
    """Types of branches"""
    STGCN = "stgcn"
    OPTFLOW_CNN = "optflow_cnn"
    OPTFLOW_LSTM = "optflow_lstm"
    TRACK_CNN = "track_cnn"
    TRACK_LSTM = "track_lstm"


class BaseBranch(nn.Module):
    """Base class for all branches"""
    
    def __init__(self, output_dim: int):
        super().__init__()
        self.output_dim = output_dim
    
    def forward(self, x):
        raise NotImplementedError
    
    def get_output_dim(self) -> int:
        return self.output_dim


class STGCNBranch(BaseBranch):
    """STGCN branch for skeleton + motion"""
    
    def __init__(self, ks, kt, n, channel_config, p, Lk):
        # Calculate output dim from config
        output_dim = channel_config[1][2]  # Last channel in config
        super().__init__(output_dim)
        
        self.st_conv1 = STBlock(ks, kt, n, channel_config[0], p, Lk)
        self.st_conv2 = STBlock(ks, kt, n, channel_config[1], p, Lk)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
    def forward(self, skeleton, motion):
        # Combine skeleton and motion
        x = torch.cat([skeleton, motion], dim=1)  # (B, 6, T, n)
        
        # STGCN forward
        x = self.st_conv1(x)
        x = self.st_conv2(x)
        
        # Global pooling
        x = self.pool(x)
        x = x.squeeze(-1).squeeze(-1)  # (B, C)
        
        return x


class TemporalCNNBranch(BaseBranch):
    """Temporal CNN branch for track or optical flow"""
    
    def __init__(self, input_dim: int, hidden_dims: list, p: float):
        super().__init__(hidden_dims[-1])
        
        layers = []
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            kernel_size = 5 if i == 0 else 3
            padding = kernel_size // 2
            
            layers.extend([
                nn.Conv1d(prev_dim, hidden_dim, kernel_size, padding=padding),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(p)
            ])
            prev_dim = hidden_dim
        
        self.conv = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        
    def forward(self, x):
        # x: (B, T, input_dim)
        x = x.permute(0, 2, 1)  # (B, C, T)
        x = self.conv(x)
        x = self.pool(x)
        x = x.squeeze(-1)  # (B, output_dim)
        return x


class LSTMBranch(BaseBranch):
    """LSTM branch for track or optical flow"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, num_layers: int = 2, 
                 bidirectional: bool = True, p: float = 0.5):
        output_dim = hidden_dim * (2 if bidirectional else 1)
        super().__init__(output_dim)
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=p if num_layers > 1 else 0
        )
        
    def forward(self, x):
        # x: (B, T, input_dim)
        _, (hidden, _) = self.lstm(x)
        
        if self.lstm.bidirectional:
            # Take last hidden states from both directions
            forward = hidden[-2, :, :]
            backward = hidden[-1, :, :]
            feat = torch.cat([forward, backward], dim=1)
        else:
            feat = hidden[-1, :, :]
        
        return feat


class MultiBranchFusion(nn.Module):
    """
    Flexible multi-branch architecture with different fusion methods.
    Accepts batch= with skeleton, motion, track, optflow (same as ST-GCN dataloader).
    """

    def __init__(
        self,
        cfg: ModelConfig,
    ):
        super().__init__()
        self.cfg = cfg

        # Convert string to enum
        fusion_type = cfg.fusion_type
        if isinstance(fusion_type, str):
            fusion_type = FusionType(fusion_type)

        self.fusion_type = fusion_type

        # Create Lk adjacency matrix
        self.register_buffer('Lk', torch.eye(cfg.num_keypoints).unsqueeze(0))

        # STGCN config
        self.stgcn_config = [[6, 64, 64], [64, 64, 128]]

        # ==================== CREATE BRANCHES ====================
        self.branches = nn.ModuleDict()
        self.branch_output_dims = {}

        # Branch 1: STGCN (always present)
        self.branches['stgcn'] = STGCNBranch(
            ks=1, kt=3,
            n=cfg.num_keypoints,
            channel_config=self.stgcn_config,
            p=cfg.dropout,
            Lk=self.Lk
        )
        self.branch_output_dims['stgcn'] = self.branches['stgcn'].output_dim

        # Branch 2: Optical Flow
        if cfg.use_lstm_for_optflow:
            self.branches['optflow'] = LSTMBranch(
                input_dim=6,
                hidden_dim=128,
                num_layers=2,
                bidirectional=True,
                p=cfg.dropout
            )
        else:
            self.branches['optflow'] = TemporalCNNBranch(
                input_dim=6,
                hidden_dims=[32, 64, 128],
                p=cfg.dropout
            )
        self.branch_output_dims['optflow'] = self.branches['optflow'].output_dim

        # Branch 3: Track
        if cfg.use_lstm_for_track:
            self.branches['track'] = LSTMBranch(
                input_dim=9,
                hidden_dim=128,
                num_layers=2,
                bidirectional=True,
                p=cfg.dropout
            )
        else:
            self.branches['track'] = TemporalCNNBranch(
                input_dim=9,
                hidden_dims=[32, 64, 128],
                p=cfg.dropout
            )
        self.branch_output_dims['track'] = self.branches['track'].output_dim

        # ==================== FUSION MODULE ====================
        self.fusion_module = self._create_fusion_module(fusion_type, cfg.dropout)

        # ==================== CLASSIFIER ====================
        self.classifier = self._create_classifier(cfg.dropout, cfg.num_classes)

    def _create_fusion_module(self, fusion_type: FusionType, dropout: float):
        """Create fusion module based on type"""
        
        total_dim = sum(self.branch_output_dims.values())
        
        if fusion_type == FusionType.CONCAT:
            return ConcatFusion(self.branch_output_dims)
            
        elif fusion_type == FusionType.WEIGHTED:
            return WeightedFusion(self.branch_output_dims)
            
        elif fusion_type == FusionType.ATTENTION:
            return AttentionFusion(
                branch_dims=self.branch_output_dims,
                projection_dim=256,
                num_heads=4,
                dropout=dropout
            )
            
        elif fusion_type == FusionType.GATED:
            return GatedFusion(
                branch_dims=self.branch_output_dims,
                projection_dim=256,
                dropout=dropout
            )
            
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")
    
    def _create_classifier(self, dropout: float, num_classes: int):
        """Create classifier based on fusion output dimension"""
        
        fusion_dim = self.fusion_module.get_output_dim()
        
        return nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(256, num_classes)
        )
    
    def forward(
        self,
        skeleton: Optional[torch.Tensor] = None,
        motion: Optional[torch.Tensor] = None,
        track: Optional[torch.Tensor] = None,
        optflow: Optional[torch.Tensor] = None,
        *,
        batch: Optional[dict[str, Any]] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Any]]:
        """
        Args (either positional or via batch=):
            skeleton: (B, 3, T, n)
            motion: (B, 3, T, n)
            track: (B, T, 9)
            optflow: (B, T, 6)
        Returns:
            logits: (B, num_classes). If fusion returns aux (e.g. attention), only logits for training.
        """
        if batch is not None:
            skeleton = batch.get("skeleton", skeleton)
            motion = batch.get("motion", motion)
            track = batch.get("track", track)
            optflow = batch.get("optflow", optflow)
        if skeleton is None or motion is None or track is None or optflow is None:
            raise ValueError(
                "MultiBranchFusion requires skeleton, motion, track, optflow (or batch with them)"
            )

        # Process each branch
        branch_outputs = {}

        # STGCN branch
        branch_outputs['stgcn'] = self.branches['stgcn'](skeleton, motion)

        # Optical flow branch
        branch_outputs['optflow'] = self.branches['optflow'](optflow)

        # Track branch
        branch_outputs['track'] = self.branches['track'](track)

        # Fusion
        fused, aux_output = self.fusion_module(branch_outputs)

        # Classification
        logits = self.classifier(fused)

        if aux_output is not None and batch is None:
            return logits, aux_output
        return logits


# ==================== FUSION MODULES ====================

class ConcatFusion(nn.Module):
    """Simple concatenation fusion"""
    
    def __init__(self, branch_dims: Dict[str, int]):
        super().__init__()
        self.branch_dims = branch_dims
        self.output_dim = sum(branch_dims.values())
        
    def forward(self, branch_outputs):
        fused = torch.cat(list(branch_outputs.values()), dim=1)
        return fused, None
    
    def get_output_dim(self):
        return self.output_dim


class WeightedFusion(nn.Module):
    """Learnable weighted fusion"""
    
    def __init__(self, branch_dims: Dict[str, int]):
        super().__init__()
        self.branch_dims = branch_dims
        self.output_dim = sum(branch_dims.values())
        
        # Learnable weights for each branch
        self.weights = nn.Parameter(torch.ones(len(branch_dims)))
        
    def forward(self, branch_outputs):
        # Apply softmax to weights
        weights = F.softmax(self.weights, dim=0)
        
        weighted_outputs = []
        for i, (name, output) in enumerate(branch_outputs.items()):
            weighted_outputs.append(weights[i] * output)
        
        fused = torch.cat(weighted_outputs, dim=1)
        return fused, weights
    
    def get_output_dim(self):
        return self.output_dim


class AttentionFusion(nn.Module):
    """Attention-based fusion"""
    
    def __init__(self, branch_dims: Dict[str, int], projection_dim: int = 256, 
                 num_heads: int = 4, dropout: float = 0.5):
        super().__init__()
        
        self.branch_names = list(branch_dims.keys())
        self.num_branches = len(branch_dims)
        self.projection_dim = projection_dim
        self.output_dim = projection_dim
        
        # Project each branch to common dimension
        self.projections = nn.ModuleDict({
            name: nn.Linear(dim, projection_dim)
            for name, dim in branch_dims.items()
        })
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=projection_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout
        )
        
        self.norm = nn.LayerNorm(projection_dim)
        
    def forward(self, branch_outputs):
        # Project to common dimension
        projected = []
        for name in self.branch_names:
            proj = self.projections[name](branch_outputs[name])
            projected.append(proj.unsqueeze(1))  # (B, 1, proj_dim)
        
        # Stack: (B, num_branches, proj_dim)
        features = torch.cat(projected, dim=1)
        
        # Self-attention
        attended, attn_weights = self.attention(features, features, features)
        
        # Residual + norm
        attended = self.norm(features + attended)
        
        # Pool across branches
        fused = attended.mean(dim=1)  # (B, proj_dim)
        
        return fused, attn_weights
    
    def get_output_dim(self):
        return self.output_dim


class GatedFusion(nn.Module):
    """Gated fusion mechanism"""
    
    def __init__(self, branch_dims: Dict[str, int], projection_dim: int = 256, 
                 dropout: float = 0.5):
        super().__init__()
        
        self.branch_names = list(branch_dims.keys())
        self.num_branches = len(branch_dims)
        self.projection_dim = projection_dim
        
        # Project each branch
        self.projections = nn.ModuleDict({
            name: nn.Linear(dim, projection_dim)
            for name, dim in branch_dims.items()
        })
        
        # Gates for each branch
        self.gates = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(projection_dim, projection_dim // 2),
                nn.ReLU(),
                nn.Linear(projection_dim // 2, 1),
                nn.Sigmoid()
            )
            for name in branch_dims.keys()
        })
        
        # Fusion layer
        self.fusion_layer = nn.Linear(projection_dim * self.num_branches, projection_dim)
        
        self.output_dim = projection_dim
        
    def forward(self, branch_outputs):
        # Project and apply gates
        gated_features = []
        gates = {}
        
        for name in self.branch_names:
            proj = self.projections[name](branch_outputs[name])  # (B, proj_dim)
            gate = self.gates[name](proj)  # (B, 1)
            gated = gate * proj
            gated_features.append(gated)
            gates[name] = gate
        
        # Concatenate and fuse
        combined = torch.cat(gated_features, dim=1)  # (B, proj_dim * num_branches)
        fused = self.fusion_layer(combined)  # (B, proj_dim)
        
        return fused, gates
    
    def get_output_dim(self):
        return self.output_dim


# ==================== FACTORY FUNCTION ====================

def create_model(
    model_type: str = "default",
    num_classes: int = 10,
    num_keypoints: int = 21,
    temporal_length: int = 478,
    dropout: float = 0.5,
    **kwargs: Any,
) -> nn.Module:
    """
    Factory function to create different model variants.

    Args:
        model_type: One of:
            - "default": concat fusion, CNN branches
            - "attention": attention fusion, CNN branches
            - "lstm": weighted fusion, LSTM branches
            - "lstm_attention": attention fusion, LSTM branches
            - "lstm_gated": gated fusion, LSTM branches
    """
    configs = {
        "default": {
            "fusion_type": FusionType.CONCAT,
            "use_lstm_for_track": False,
            "use_lstm_for_optflow": False,
        },
        "attention": {
            "fusion_type": FusionType.ATTENTION,
            "use_lstm_for_track": False,
            "use_lstm_for_optflow": False,
        },
        "lstm": {
            "fusion_type": FusionType.WEIGHTED,
            "use_lstm_for_track": True,
            "use_lstm_for_optflow": True,
        },
        "lstm_attention": {
            "fusion_type": FusionType.ATTENTION,
            "use_lstm_for_track": True,
            "use_lstm_for_optflow": True,
        },
        "lstm_gated": {
            "fusion_type": FusionType.GATED,
            "use_lstm_for_track": True,
            "use_lstm_for_optflow": True,
        },
    }

    if model_type not in configs:
        raise ValueError(
            f"Unknown model type: {model_type}. Available: {list(configs.keys())}"
        )

    cfg = ModelConfig(
        num_classes=num_classes,
        num_keypoints=num_keypoints,
        temporal_length=temporal_length,
        dropout=dropout,
        **configs[model_type],
        **{k: v for k, v in kwargs.items() if k in ModelConfig.__dataclass_fields__},
    )
    return MultiBranchFusion(cfg)

