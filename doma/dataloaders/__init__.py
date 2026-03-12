"""
Data loaders for gesture recognition.

Public API:
- doma.dataloaders.build_dataloaders         — manifest (unified features, train/val/test)
- doma.dataloaders.build_dataloaders_cnn_lstm — same, with y/label_str for CNN-LSTM runner
- doma.dataloaders.collate_gesture_batch     — collate for manifest batches
- doma.dataloaders.gesture_features          — FeatureConfig, NormStats, read_manifest_rows, etc.
"""

from .dataloader import (
    build_dataloaders,
    build_dataloaders_cnn_lstm,
    collate_gesture_batch,
    collate_padded,
    CNNLSTMGestureDataset,
)
from .flow_dataloader import build_dataloaders as build_flow_dataloaders
from .stgcn_dataloader import build_dataloaders_stgcn
from .gesture_features import (
    FeatureConfig,
    NormStats,
    SampleRow,
    build_label_map,
    compute_norm_stats,
    read_manifest_rows,
)

__all__ = [
    "build_dataloaders",
    "build_dataloaders_cnn_lstm",
    "build_dataloaders_stgcn",
    "build_flow_dataloaders",
    "collate_gesture_batch",
    "collate_padded",
    "CNNLSTMGestureDataset",
    "FeatureConfig",
    "NormStats",
    "SampleRow",
    "build_label_map",
    "compute_norm_stats",
    "read_manifest_rows",
]
