"""
DataLoader: manifest.csv (pose/optflow NPZ). build_dataloaders and collate_gesture_batch.
Dataset class is internal.
"""

from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from config.labels import LABEL_TO_ID


MANIFEST_COLS = ["sample_id", "dataset", "split", "label", "pose_npz", "optflow_npz"]


def _load_pose_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as z:
        return {k: np.asarray(z[k]) for k in z.files}


def _load_optflow_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as z:
        return {k: np.asarray(z[k]) for k in z.files}


def _pose_to_tensor(data: dict[str, np.ndarray]) -> tuple[torch.Tensor, torch.Tensor]:
    pos = np.asarray(data["track_pos_xyz"], dtype=np.float32)
    vel = np.asarray(data["track_vel_xyz"], dtype=np.float32)
    acc = np.asarray(data["track_acc_xyz"], dtype=np.float32)
    lm = np.asarray(data["landmarks_xyz"], dtype=np.float32)
    valid = np.asarray(data["valid"], dtype=bool)
    track = np.concatenate([pos, vel, acc], axis=1)
    landmarks_flat = lm.reshape(lm.shape[0], -1)
    pose = np.concatenate([track, landmarks_flat], axis=1)
    return torch.from_numpy(pose), torch.from_numpy(valid)


def _optflow_to_tensor(data: dict[str, np.ndarray]) -> tuple[torch.Tensor, torch.Tensor]:
    avg_speed = np.asarray(data["avg_speed"], dtype=np.float32)
    max_speed = np.asarray(data["max_speed"], dtype=np.float32)
    dominant_angle = np.asarray(data["dominant_angle_deg"], dtype=np.float32)
    concentration = np.asarray(data["direction_concentration"], dtype=np.float32)
    n_pixels = np.asarray(data["n_pixels"], dtype=np.float32)
    threshold = np.asarray(data["threshold"], dtype=np.float32)
    valid = np.asarray(data["valid"], dtype=bool)
    opt = np.stack([avg_speed, max_speed, dominant_angle, concentration, n_pixels, threshold], axis=1)
    return torch.from_numpy(opt), torch.from_numpy(valid)


class _GestureDataset(Dataset[dict[str, Any]]):
    """Internal. Dataset of gesture samples from manifest.csv."""

    def __init__(
        self,
        manifest_path: str | Path,
        root_dir: str | Path,
        split: Optional[Union[str, list[str]]] = None,
        label_to_id: Optional[dict[str, int]] = None,
        require_valid_paths: bool = True,
    ):
        self.root = Path(root_dir)
        self.manifest_path = Path(manifest_path)
        self.label_to_id = label_to_id if label_to_id is not None else LABEL_TO_ID
        self.require_valid_paths = require_valid_paths

        df = pd.read_csv(self.manifest_path)
        if not set(MANIFEST_COLS).issubset(df.columns):
            raise ValueError(f"Manifest must contain columns {MANIFEST_COLS}")
        df = df[MANIFEST_COLS].copy()

        if split is not None:
            if isinstance(split, str):
                split = [split]
            df = df[df["split"].isin(split)].reset_index(drop=True)

        if require_valid_paths:
            df = df.dropna(subset=["pose_npz", "optflow_npz"])
            df = df[
                df["pose_npz"].astype(str).str.strip().ne("")
                & df["optflow_npz"].astype(str).str.strip().ne("")
            ].reset_index(drop=True)

        known_labels = set(self.label_to_id)
        df = df[df["label"].isin(known_labels)].reset_index(drop=True)

        self._table = df
        self._num_classes = len(self.label_to_id)

    def __len__(self) -> int:
        return len(self._table)

    @property
    def num_classes(self) -> int:
        return self._num_classes

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self._table.iloc[idx]
        sample_id = str(row["sample_id"])
        label_str = str(row["label"]).strip()
        label = self.label_to_id[label_str]

        pose_path = self.root / str(row["pose_npz"]).strip()
        optflow_path = self.root / str(row["optflow_npz"]).strip()

        pose_t, valid_pose = _pose_to_tensor(_load_pose_npz(pose_path))
        optflow_t, valid_optflow = _optflow_to_tensor(_load_optflow_npz(optflow_path))

        T = pose_t.size(0)
        assert optflow_t.size(0) == T, "pose and optflow length mismatch"

        return {
            "pose": pose_t,
            "optflow": optflow_t,
            "valid_pose": valid_pose,
            "valid_optflow": valid_optflow,
            "label": label,
            "sample_id": sample_id,
            "length": T,
        }


def collate_gesture_batch(
    batch: list[dict[str, Any]],
    pad_value: float = 0.0,
    max_len: Optional[int] = None,
) -> dict[str, Any]:
    """
    Collate a list of gesture samples into a batch with padding.
    Returns dict: pose (B,T_max,72), optflow (B,T_max,6), valid_pose, valid_optflow, label, length, sample_id.
    """
    if not batch:
        return {}

    lengths = torch.tensor([s["length"] for s in batch], dtype=torch.long)
    T_max = int(lengths.max().item()) if max_len is None else min(int(lengths.max().item()), max_len)

    B = len(batch)
    pose_dim = batch[0]["pose"].size(1)
    optflow_dim = batch[0]["optflow"].size(1)

    pose_padded = torch.full((B, T_max, pose_dim), pad_value, dtype=torch.float32)
    optflow_padded = torch.full((B, T_max, optflow_dim), pad_value, dtype=torch.float32)
    valid_pose_padded = torch.zeros(B, T_max, dtype=torch.bool)
    valid_optflow_padded = torch.zeros(B, T_max, dtype=torch.bool)
    labels = torch.zeros(B, dtype=torch.long)
    sample_ids: list[str] = []

    for i, s in enumerate(batch):
        T = min(s["length"], T_max)
        pose_padded[i, :T] = s["pose"][:T]
        optflow_padded[i, :T] = s["optflow"][:T]
        valid_pose_padded[i, :T] = s["valid_pose"][:T]
        valid_optflow_padded[i, :T] = s["valid_optflow"][:T]
        labels[i] = s["label"]
        sample_ids.append(s["sample_id"])

    return {
        "pose": pose_padded,
        "optflow": optflow_padded,
        "valid_pose": valid_pose_padded,
        "valid_optflow": valid_optflow_padded,
        "label": labels,
        "length": lengths,
        "sample_id": sample_ids,
    }


def build_dataloaders(
    manifest_path: str | Path,
    root_dir: str | Path,
    batch_size: int = 32,
    num_workers: int = 0,
    label_to_id: Optional[dict[str, int]] = None,
    max_len: Optional[int] = None,
    pad_value: float = 0.0,
    split_mode: str = "train_val_test",
    pin_memory: bool = True,
    generator: Optional[torch.Generator] = None,
):
    """
    Build DataLoaders from manifest.csv.
    Returns (train_loader, val_loader, test_loader) or (train_loader, test_loader) depending on split_mode.
    If generator is provided, it is used for shuffle (reproducible runs).
    """
    from torch.utils.data import DataLoader

    collate = lambda b: collate_gesture_batch(b, pad_value=pad_value, max_len=max_len)

    def _loader(splits: Union[str, list[str]], shuffle: bool) -> Optional[DataLoader]:
        ds = _GestureDataset(
            manifest_path=manifest_path,
            root_dir=root_dir,
            split=splits,
            label_to_id=label_to_id,
        )
        if len(ds) == 0:
            return None
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
            _loader("train", shuffle=True),
            _loader("val", shuffle=False),
            _loader("test", shuffle=False),
        )
    if split_mode == "train_test":
        return (
            _loader(["train", "val"], shuffle=True),
            _loader("test", shuffle=False),
        )
    raise ValueError(
        f'split_mode must be "train_val_test" or "train_test", got {split_mode!r}'
    )
