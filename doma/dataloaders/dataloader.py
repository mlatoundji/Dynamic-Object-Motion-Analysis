"""
DataLoader: manifest.csv (pose/optflow NPZ). build_dataloaders and collate.
Uses unified features from gesture_features (same as CNN-LSTM pipeline).
Batch format: {x (B,T,F), lengths (B,), label (B,), sample_id}.
"""

from pathlib import Path
from typing import Any, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .gesture_features import (
    FeatureConfig,
    NormStats,
    SampleRow,
    build_label_map,
    compute_norm_stats,
    load_sample_features,
    read_manifest_rows,
)


MANIFEST_COLS = ["sample_id", "dataset", "split", "label", "pose_npz", "optflow_npz"]


def _split_rows(rows: list[SampleRow]) -> tuple[list[SampleRow], list[SampleRow], list[SampleRow]]:
    tr = [r for r in rows if r.split == "train"]
    va = [r for r in rows if r.split == "val"]
    te = [r for r in rows if r.split == "test"]
    return tr, va, te


class _GestureDataset(Dataset[dict[str, Any]]):
    """Dataset of gesture samples using unified features (pose + optflow, valid-frame filtering)."""

    def __init__(
        self,
        rows: list[SampleRow],
        *,
        label_to_idx: dict[str, int],
        feat_cfg: FeatureConfig,
        norm: NormStats | None = None,
    ) -> None:
        self._rows = [r for r in rows if r.label in label_to_idx]
        self._label_to_idx = dict(label_to_idx)
        self._feat_cfg = feat_cfg
        self._norm = norm

    def __len__(self) -> int:
        return len(self._rows)

    @property
    def num_classes(self) -> int:
        return len(self._label_to_idx)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        r = self._rows[int(idx)]
        x, valid = load_sample_features(r, feat_cfg=self._feat_cfg, norm=self._norm)
        if int(np.count_nonzero(valid)) > 0:
            x = x[valid]
        else:
            x = x[:0]
        y = int(self._label_to_idx[r.label])
        return {
            "x": x.astype(np.float32),
            "length": int(x.shape[0]),
            "label": y,
            "sample_id": r.sample_id,
        }


class CNNLSTMGestureDataset(Dataset[dict[str, Any]]):
    """
    Dataset for CNN-LSTM / runner: same features as _GestureDataset.
    Returns dict with x (T,F), y, sample_id, label (string) for compatibility with collate_padded.
    """

    def __init__(
        self,
        rows: list[SampleRow],
        *,
        label_to_idx: dict[str, int],
        feat_cfg: FeatureConfig,
        norm: NormStats | None = None,
    ) -> None:
        self._rows = [r for r in rows if r.label in label_to_idx]
        self._label_to_idx = dict(label_to_idx)
        self._feat_cfg = feat_cfg
        self._norm = norm

    @property
    def num_classes(self) -> int:
        return len(self._label_to_idx)

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        r = self._rows[int(idx)]
        x, valid = load_sample_features(r, feat_cfg=self._feat_cfg, norm=self._norm)
        if int(np.count_nonzero(valid)) > 0:
            x = x[valid]
        else:
            x = x[:0]
        y = int(self._label_to_idx[r.label])
        return {"x": x.astype(np.float32), "y": y, "sample_id": r.sample_id, "label": r.label}


def collate_padded(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate for CNN-LSTM runner: x (B,T,F), lengths, y, label (tensor), label_str, sample_id."""
    if not batch:
        return {}
    xs = [np.asarray(b["x"], dtype=np.float32) for b in batch]
    ys = torch.tensor([int(b["y"]) for b in batch], dtype=torch.long)
    sample_ids = [str(b["sample_id"]) for b in batch]
    labels_str = [str(b["label"]) for b in batch]
    lengths = torch.tensor([int(x.shape[0]) for x in xs], dtype=torch.long)
    max_len = int(lengths.max().item()) if lengths.numel() else 0
    feat_dim = int(xs[0].shape[1]) if xs and xs[0].ndim == 2 else 0
    x_pad = torch.zeros((len(xs), max_len, feat_dim), dtype=torch.float32)
    for i, x in enumerate(xs):
        if x.size == 0:
            continue
        t = int(x.shape[0])
        x_pad[i, :t] = torch.from_numpy(x)
    return {
        "x": x_pad,
        "lengths": lengths,
        "y": ys,
        "sample_id": sample_ids,
        "label": ys,
        "label_str": labels_str,
    }


def collate_gesture_batch(
    batch: list[dict[str, Any]],
    max_len: Optional[int] = None,
) -> dict[str, Any]:
    """
    Collate to padded batch. Returns dict: x (B,T,F), lengths (B,), label (B,), sample_id.
    """
    if not batch:
        return {}

    xs = [np.asarray(b["x"], dtype=np.float32) for b in batch]
    lengths = torch.tensor([b["length"] for b in batch], dtype=torch.long)
    labels = torch.tensor([int(b["label"]) for b in batch], dtype=torch.long)
    sample_ids = [str(b["sample_id"]) for b in batch]

    T_max = int(lengths.max().item())
    if max_len is not None:
        T_max = min(T_max, max_len)
    feat_dim = int(xs[0].shape[1]) if xs and xs[0].ndim == 2 else 0

    x_pad = torch.zeros((len(xs), T_max, feat_dim), dtype=torch.float32)
    for i, x in enumerate(xs):
        t = min(int(x.shape[0]), T_max)
        if t > 0:
            x_pad[i, :t] = torch.from_numpy(x[:t])

    return {
        "x": x_pad,
        "lengths": lengths,
        "label": labels,
        "sample_id": sample_ids,
    }


def build_dataloaders(
    manifest_path: str | Path,
    root_dir: str | Path,
    batch_size: int = 32,
    num_workers: int = 0,
    label_to_id: Optional[dict[str, int]] = None,
    max_len: Optional[int] = None,
    split_mode: str = "train_val_test",
    pin_memory: bool = True,
    generator: Optional[torch.Generator] = None,
    feature_cfg: Optional[FeatureConfig] = None,
):
    """
    Build DataLoaders from manifest.csv using unified gesture features (same as CNN-LSTM).
    Computes normalization stats from train split; all splits use the same norm.
    Returns (train_loader, val_loader, test_loader) or (train_loader, test_loader).
    Batch dict: x (B,T,F), lengths (B,), label (B,), sample_id.
    """
    from torch.utils.data import DataLoader

    manifest_path = Path(manifest_path)
    root_dir = Path(root_dir)
    feat_cfg = feature_cfg or FeatureConfig()

    rows = read_manifest_rows(manifest_path, repo_root=root_dir)
    if not rows:
        raise RuntimeError(f"No valid rows from manifest: {manifest_path}")

    # Use provided label map or build from manifest
    if label_to_id is not None:
        label_to_idx = dict(label_to_id)
    else:
        label_to_idx = build_label_map(rows)

    train_rows, val_rows, test_rows = _split_rows(rows)
    if not train_rows:
        raise RuntimeError("No training rows in manifest. Check manifest path and split column.")

    norm = compute_norm_stats(
        train_rows,
        feat_cfg=feat_cfg,
        label_to_idx=label_to_idx,
        max_samples=0,
    )

    def _loader(rows_split: list[SampleRow], shuffle: bool):
        if not rows_split:
            return None
        ds = _GestureDataset(
            rows_split,
            label_to_idx=label_to_idx,
            feat_cfg=feat_cfg,
            norm=norm,
        )
        if len(ds) == 0:
            return None
        collate = lambda b: collate_gesture_batch(b, max_len=max_len)
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate,
            pin_memory=pin_memory,
            generator=generator if shuffle else None,
        )

    if split_mode == "train_val_test":
        return (
            _loader(train_rows, shuffle=True),
            _loader(val_rows, shuffle=False),
            _loader(test_rows, shuffle=False),
        )
    if split_mode == "train_test":
        return (
            _loader(train_rows + val_rows, shuffle=True),
            _loader(test_rows, shuffle=False),
        )
    raise ValueError(
        f'split_mode must be "train_val_test" or "train_test", got {split_mode!r}'
    )


def build_dataloaders_cnn_lstm(
    manifest_path: Union[Path, str],
    root_dir: Union[Path, str],
    *,
    batch_size: int = 32,
    split_mode: str = "train_val_test",
    num_workers: int = 0,
    pin_memory: bool = False,
    generator: Optional[torch.Generator] = None,
    feature_cfg: Optional[FeatureConfig] = None,
) -> Union[
    Tuple[DataLoader, DataLoader, DataLoader],
    Tuple[DataLoader, DataLoader],
]:
    """
    Build train/val/test DataLoaders for CNN-LSTM from manifest (unified features).
    Batches: x (B,T,F), lengths (B,), label (B,) LongTensor, y, label_str, sample_id.
    Returns (train, val, test) or (train, test) depending on split_mode.
    """
    manifest_path = Path(manifest_path)
    root_dir = Path(root_dir)
    feat_cfg = feature_cfg or FeatureConfig()

    rows = read_manifest_rows(manifest_path, repo_root=root_dir)
    if not rows:
        raise RuntimeError(f"No valid rows from manifest: {manifest_path}")
    label_to_idx = build_label_map(rows)
    train_rows, val_rows, test_rows = _split_rows(rows)
    if not train_rows:
        raise RuntimeError("No training rows in manifest.")

    norm = compute_norm_stats(
        train_rows,
        feat_cfg=feat_cfg,
        label_to_idx=label_to_idx,
        max_samples=0,
    )

    def _make_loader(rows_split: list[SampleRow], shuffle: bool) -> DataLoader:
        ds = CNNLSTMGestureDataset(
            rows_split,
            label_to_idx=label_to_idx,
            feat_cfg=feat_cfg,
            norm=norm,
        )
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate_padded,
            drop_last=False,
            pin_memory=pin_memory,
            generator=generator if shuffle else None,
        )

    if split_mode == "train_val_test":
        return (
            _make_loader(train_rows, shuffle=True),
            _make_loader(val_rows, shuffle=False),
            _make_loader(test_rows, shuffle=False),
        )
    return (
        _make_loader(train_rows + val_rows, shuffle=True),
        _make_loader(test_rows, shuffle=False),
    )
