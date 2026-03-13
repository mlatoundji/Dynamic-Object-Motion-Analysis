"""
Gesture dataset and DataLoader built from data/processed/manifest.csv.

Loads pose_tensor.npz and optflow_features.npz per sample and returns
tensors plus label (class index). Uses config.labels for label<>ID mapping.
Supports filtering by split and variable-length sequences with optional padding in collate_fn.
"""

from pathlib import Path
from typing import Any, Callable, Optional, Union, List, Dict, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from config.labels import LABEL_TO_ID, NUM_CLASSES


# Default columns used from manifest
MANIFEST_COLS = ["sample_id", "dataset", "split", "label", "pose_npz", "optflow_npz"]

# Keys to load from each NPZ and stack into a single tensor (excluding timestamps and valid)
POSEDIM = 9  # track_pos(3) + track_vel(3) + track_acc(3)
OPTFLOWDIM = 6  # avg_speed, max_speed, dominant_angle_deg, direction_concentration, n_pixels, threshold


def _load_pose_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as z:
        return {k: np.asarray(z[k]) for k in z.files}


def _load_optflow_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as z:
        return {k: np.asarray(z[k]) for k in z.files}


def _pose_to_tensor(data: dict[str, np.ndarray]) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract pose features and valid mask. Returns (T, 9+63), (T,) for track+landmarks flat."""
    pos = np.asarray(data["track_pos_xyz"], dtype=np.float32)   # (T, 3)
    vel = np.asarray(data["track_vel_xyz"], dtype=np.float32)   # (T, 3)
    acc = np.asarray(data["track_acc_xyz"], dtype=np.float32)   # (T, 3)
    lm = np.asarray(data["landmarks_xyz"], dtype=np.float32)    # (T, 21, 3)
    valid = np.asarray(data["valid"], dtype=bool)

    track = np.concatenate([pos, vel, acc], axis=1)            # (T, 9)
    landmarks_flat = lm.reshape(lm.shape[0], -1)              # (T, 63)
    pose = np.concatenate([track, landmarks_flat], axis=1)     # (T, 72)
    return torch.from_numpy(pose), torch.from_numpy(valid)


def _optflow_to_tensor(data: dict[str, np.ndarray]) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract optflow features and valid mask. Returns (T, 6), (T,)."""
    avg_speed = np.asarray(data["avg_speed"], dtype=np.float32)
    max_speed = np.asarray(data["max_speed"], dtype=np.float32)
    dominant_angle = np.asarray(data["dominant_angle_deg"], dtype=np.float32)
    concentration = np.asarray(data["direction_concentration"], dtype=np.float32)
    n_pixels = np.asarray(data["n_pixels"], dtype=np.float32)
    threshold = np.asarray(data["threshold"], dtype=np.float32)
    valid = np.asarray(data["valid"], dtype=bool)

    opt = np.stack([avg_speed, max_speed, dominant_angle, concentration, n_pixels, threshold], axis=1)
    return torch.from_numpy(opt), torch.from_numpy(valid)


class GestureDataset(Dataset[dict[str, Any]]):
    """
    Dataset of gesture samples from manifest.csv.

    Each item is a dict with:
        - "pose": (T, 72) float tensor (track 9 + landmarks 63)
        - "optflow": (T, 6) float tensor
        - "valid_pose": (T,) bool
        - "valid_optflow": (T,) bool
        - "label": int class index (0-based)
        - "sample_id": str
        - "length": int (T)
    """

    def __init__(
        self,
        manifest_path: str | Path,
        root_dir: str | Path,
        split: Optional[Union[str, list[str]]] = None,
        label_to_id: Optional[dict[str, int]] = None,
        require_valid_paths: bool = True,
    ):
        """
        Args:
            manifest_path: Path to data/processed/manifest.csv.
            root_dir: Root directory for resolving relative paths in the manifest.
            split: If set, only include rows with this split. Can be a single split
                ("train", "val", "test") or a list of splits (e.g. ["train", "val"]).
            label_to_id: Map label string -> 0-based class ID. Defaults to config.labels.LABEL_TO_ID.
            require_valid_paths: If True, drop rows with missing/empty pose_npz or optflow_npz.
        """
        self.root = Path(root_dir)
        self.manifest_path = Path(manifest_path)
        self.label_to_id = label_to_id if label_to_id is not None else LABEL_TO_ID
        self.require_valid_paths = require_valid_paths

        df = pd.read_csv(self.manifest_path)
        if MANIFEST_COLS[0] not in df.columns:
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

        # Keep only rows whose label is in the label mapping (config.labels by default)
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

        pose_data = _load_pose_npz(pose_path)
        optflow_data = _load_optflow_npz(optflow_path)

        pose_t, valid_pose = _pose_to_tensor(pose_data)
        optflow_t, valid_optflow = _optflow_to_tensor(optflow_data)

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

    Returns dict with:
        - "pose": (B, T_max, 72)
        - "optflow": (B, T_max, 6)
        - "valid_pose": (B, T_max) bool
        - "valid_optflow": (B, T_max) bool
        - "label": (B,) long
        - "length": (B,) long
        - "sample_id": list of str
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


def create_dataloaders(
    manifest_path: str | Path,
    root_dir: str | Path,
    batch_size: int = 32,
    num_workers: int = 0,
    label_to_id: Optional[dict[str, int]] = None,
    max_len: Optional[int] = None,
    pad_value: float = 0.0,
    split_mode: str = "train_val_test",
    pin_memory: bool = True,
):
    """
    Create DataLoaders from manifest.

    Args:
        manifest_path: Path to data/processed/manifest.csv.
        root_dir: Root directory for resolving relative paths in the manifest.
        batch_size: Batch size for all loaders.
        num_workers: Number of worker processes (0 = main process only).
        label_to_id: Map label string -> 0-based class ID. Defaults to config.labels.LABEL_TO_ID.
        max_len: If set, cap sequence length (truncate) when collating.
        pad_value: Value used for padding sequences.
        split_mode: One of:
            - "train_val_test": three loaders — train, val, test (each may be None).
            - "train_test": two loaders — train+val combined, test.
        pin_memory: If True, pin memory for faster CPU->GPU transfer (CUDA only; set False for MPS/CPU).

    Returns:
        - If split_mode == "train_val_test": (train_loader, val_loader, test_loader).
        - If split_mode == "train_test": (train_loader, test_loader).
    """
    from torch.utils.data import DataLoader

    collate = lambda b: collate_gesture_batch(b, pad_value=pad_value, max_len=max_len)

    def _loader(splits: Union[str, list[str]], shuffle: bool) -> Optional[DataLoader]:
        ds = GestureDataset(
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


##########################################
# This is for ts-gcn model
##########################################
def tsgcn_collate_fn(
    batch: List[Dict[str, Any]], 
    max_len: Optional[int] = None,
) -> Dict[str, torch.Tensor]:

    if not batch:
        return {}

    lengths = torch.tensor([s["length"] for s in batch], dtype=torch.long)
    T_max = int(lengths.max().item()) if max_len is None else min(int(lengths.max().item()), max_len)
    B = len(batch)

    # STGCN format (B, C, T, V)
    skeleton = torch.zeros(B, 3, T_max, 21, dtype=torch.float32)
    motion = torch.zeros(B, 3, T_max, 21, dtype=torch.float32)

    track = torch.zeros(B, T_max, 9, dtype=torch.float32)
    optflow = torch.zeros(B, T_max, 6, dtype=torch.float32)

    valid_mask = torch.zeros(B, T_max, dtype=torch.bool)
    labels = torch.zeros(B, dtype=torch.long)
    sample_ids: List[str] = []

    for i, sample in enumerate(batch):

        T = min(sample["length"], T_max)

        pose = sample["pose"][:T]  # (T,72)

        track_features = pose[:, :9]
        landmarks_flat = pose[:, 9:]

        landmarks_3d = landmarks_flat.view(T, 21, 3)  # (T,21,3)

        track[i, :T] = track_features
        optflow[i, :T] = sample["optflow"][:T]

        valid_mask[i, :T] = sample["valid_pose"][:T]

        # (T,21,3) → (3,T,21)
        skeleton[i, :, :T, :] = landmarks_3d.permute(2,0,1)

        if T > 1:
            motion[i, :, 1:T, :] = (
                skeleton[i, :, 1:T, :] -
                skeleton[i, :, :T-1, :]
            )

        labels[i] = sample["label"]
        sample_ids.append(sample["sample_id"])

    # # sanitize non-finite values first (NaN/Inf can appear from preprocessing)
    skeleton = torch.nan_to_num(skeleton, nan=0.0, posinf=0.0, neginf=0.0)
    motion = torch.nan_to_num(motion, nan=0.0, posinf=0.0, neginf=0.0)
    track = torch.nan_to_num(track, nan=0.0, posinf=0.0, neginf=0.0)
    optflow = torch.nan_to_num(optflow, nan=0.0, posinf=0.0, neginf=0.0)

    # apply mask safely (avoid NaN propagation from x * 0)
    mask = valid_mask[:, None, :, None]
    skeleton = torch.where(mask, skeleton, torch.zeros_like(skeleton))
    motion = torch.where(mask, motion, torch.zeros_like(motion))

    return {
        "skeleton": skeleton,   # (B,3,T,21)
        "motion": motion,       # (B,3,T,21)
        "track": track,
        "optflow": optflow,
        "label": labels,
        "length": lengths,
        "valid_mask": valid_mask,
        "sample_id": sample_ids,
    }

def create_tsgcn_dataloaders(
    manifest_path: Union[str, Path],
    root_dir: Union[str, Path],
    batch_size: int = 32,
    num_workers: int = 0,
    label_to_id: Optional[Dict[str, int]] = None,
    max_len: Optional[int] = None,
    split_mode: str = "train_val_test",
    pin_memory: bool = True,
) -> Union[
    Tuple[Optional[DataLoader], Optional[DataLoader], Optional[DataLoader]],
    Tuple[Optional[DataLoader], Optional[DataLoader]]
]:
    """
    Create DataLoaders for TSGCN from manifest.
    
    Args:
        manifest_path: Path to data/processed/manifest.csv
        root_dir: Root directory for resolving relative paths
        batch_size: Batch size for all loaders
        num_workers: Number of worker processes (0 = main process only)
        label_to_id: Map label string -> 0-based class ID. Defaults to config.labels.LABEL_TO_ID
        max_len: If set, cap sequence length (truncate) when collating
        pad_value: Value used for padding sequences
        split_mode: One of:
            - "train_val_test": three loaders — train, val, test (each may be None)
            - "train_test": two loaders — train+val combined, test
        pin_memory: If True, pin memory for faster CPU->GPU transfer
        include_motion: If True, include motion (velocity) features in output
    
    Returns:
        - If split_mode == "train_val_test": (train_loader, val_loader, test_loader)
        - If split_mode == "train_test": (train_loader, test_loader)
    """
    
    # Create collate function with fixed parameters
    from functools import partial

    collate_fn = partial(
        tsgcn_collate_fn,
        max_len=max_len,
    )
        
    def _create_loader(splits: Union[str, List[str]], shuffle: bool) -> Optional[DataLoader]:
        dataset = GestureDataset(
            manifest_path=manifest_path,
            root_dir=root_dir,
            split=splits,
            label_to_id=label_to_id,
            require_valid_paths=True,
        )
        
        if len(dataset) == 0:
            print(f"Warning: No samples found for split: {splits}")
            return None
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            drop_last=False,  # Keep incomplete batches
        )
    
    # Create loaders based on split_mode
    if split_mode == "train_val_test":
        train_loader = _create_loader("train", shuffle=True)
        val_loader = _create_loader("val", shuffle=False)
        test_loader = _create_loader("test", shuffle=False)
        return train_loader, val_loader, test_loader
    
    elif split_mode == "train_test":
        train_loader = _create_loader(["train", "val"], shuffle=True)
        test_loader = _create_loader("test", shuffle=False)
        return train_loader, test_loader
    
    else:
        raise ValueError(
            f'split_mode must be "train_val_test" or "train_test", got {split_mode!r}'
        )