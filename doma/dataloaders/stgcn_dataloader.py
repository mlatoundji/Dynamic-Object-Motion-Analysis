"""
ST-GCN dataloader: builds batches with skeleton, motion, track, optflow from manifest + pose/optflow NPZ.
Used by stgcn and stgcn_opt models. Batch format: skeleton (B,3,T,n), motion (B,3,T,n), track (B,T,9), optflow (B,T,6), label (B,), lengths (B,).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .gesture_features import SampleRow, build_label_map, read_manifest_rows
from .dataloader import _split_rows


def _load_stgcn_sample(
    row: SampleRow,
    *,
    use_landmarks: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Load one sample for ST-GCN. Returns (skeleton, motion, track, optflow, length).
    skeleton: (3, T, n), motion: (3, T, n), track: (T, 9), optflow: (T, 6).
    n = 21 if landmarks else 1.
    Replaces NaN/Inf in arrays so training does not produce NaN loss.
    """
    pose = np.load(row.pose_npz, allow_pickle=True)
    opt = np.load(row.optflow_npz, allow_pickle=True)

    def _sanitize(a: np.ndarray) -> np.ndarray:
        a = np.asarray(a, dtype=np.float32)
        bad = ~np.isfinite(a)
        if np.any(bad):
            a = np.where(bad, 0.0, a)
        return a

    pos = _sanitize(pose["track_pos_xyz"])  # (T, 3)
    vel = _sanitize(pose["track_vel_xyz"])
    acc = _sanitize(pose["track_acc_xyz"])
    T_pose = int(pos.shape[0])

    track = np.concatenate([pos, vel, acc], axis=1).astype(np.float32)  # (T, 9)

    avg = _sanitize(opt["avg_speed"]).reshape(-1, 1)
    mx = _sanitize(opt["max_speed"]).reshape(-1, 1)
    ang = np.deg2rad(_sanitize(opt["dominant_angle_deg"]).reshape(-1, 1))
    sin = np.sin(ang).astype(np.float32)
    cos = np.cos(ang).astype(np.float32)
    conc = _sanitize(opt["direction_concentration"]).reshape(-1, 1)
    thr = _sanitize(opt["threshold"]).reshape(-1, 1)
    optflow = np.concatenate([avg, mx, sin, cos, conc, thr], axis=1).astype(np.float32)  # (T_opt, 6)
    T_opt = int(optflow.shape[0])
    T = min(T_pose, T_opt)
    track = track[:T]
    optflow = optflow[:T]

    if use_landmarks and "landmarks_xyz" in pose.files:
        lm = _sanitize(pose["landmarks_xyz"])  # (T, 21, 3)
        lm = lm[:T]
        n_kp = 21
        skeleton = np.transpose(lm, (2, 0, 1))  # (3, T, 21)
        motion = np.zeros_like(skeleton)
        if T > 1:
            motion[:, 1:] = skeleton[:, 1:] - skeleton[:, :-1]
    else:
        n_kp = 1
        skeleton = pos[:T].T.reshape(3, T, 1)
        motion = vel[:T].T.reshape(3, T, 1)

    return skeleton, motion, track, optflow, int(T)


class _STGCNDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        rows: list[SampleRow],
        *,
        label_to_idx: dict[str, int],
        use_landmarks: bool = True,
    ) -> None:
        self._rows = [r for r in rows if r.label in label_to_idx]
        self._label_to_idx = dict(label_to_idx)
        self._use_landmarks = use_landmarks

    def __len__(self) -> int:
        return len(self._rows)

    @property
    def num_classes(self) -> int:
        return len(self._label_to_idx)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        r = self._rows[int(idx)]
        skeleton, motion, track, optflow, length = _load_stgcn_sample(
            r, use_landmarks=self._use_landmarks
        )
        y = int(self._label_to_idx[r.label])
        return {
            "skeleton": skeleton,
            "motion": motion,
            "track": track,
            "optflow": optflow,
            "length": length,
            "label": y,
            "sample_id": r.sample_id,
        }


def _collate_stgcn(
    batch: list[dict[str, Any]],
    max_len: Optional[int] = None,
) -> dict[str, Any]:
    if not batch:
        return {}
    lengths = torch.tensor([b["length"] for b in batch], dtype=torch.long)
    T_max = int(lengths.max().item())
    if max_len is not None:
        T_max = min(T_max, max_len)
    B = len(batch)
    n = batch[0]["skeleton"].shape[2]
    skeleton = torch.zeros((B, 3, T_max, n), dtype=torch.float32)
    motion = torch.zeros((B, 3, T_max, n), dtype=torch.float32)
    track = torch.zeros((B, T_max, 9), dtype=torch.float32)
    optflow = torch.zeros((B, T_max, 6), dtype=torch.float32)
    labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    for i, b in enumerate(batch):
        t = min(int(b["length"]), T_max)
        if t > 0:
            skeleton[i, :, :t, :] = torch.from_numpy(b["skeleton"][:, :t, :])
            motion[i, :, :t, :] = torch.from_numpy(b["motion"][:, :t, :])
            track[i, :t, :] = torch.from_numpy(b["track"][:t])
            optflow[i, :t, :] = torch.from_numpy(b["optflow"][:t])
    return {
        "skeleton": skeleton,
        "motion": motion,
        "track": track,
        "optflow": optflow,
        "label": labels,
        "lengths": lengths,
    }


def build_dataloaders_stgcn(
    manifest_path: str | Path,
    root_dir: str | Path,
    batch_size: int = 32,
    num_workers: int = 0,
    max_len: Optional[int] = None,
    split_mode: str = "train_val_test",
    use_landmarks: bool = True,
    pin_memory: bool = False,
    generator: Optional[torch.Generator] = None,
):
    """
    Build train/val/test DataLoaders for ST-GCN / ST-GCN-Opt.
    Batch: skeleton (B,3,T,n), motion (B,3,T,n), track (B,T,9), optflow (B,T,6), label (B,), lengths (B,).
    """
    manifest_path = Path(manifest_path)
    root_dir = Path(root_dir)
    rows = read_manifest_rows(manifest_path, repo_root=root_dir)
    if not rows:
        raise RuntimeError(f"No valid rows from manifest: {manifest_path}")
    label_to_idx = build_label_map(rows)
    train_rows, val_rows, test_rows = _split_rows(rows)
    if not train_rows:
        raise RuntimeError("No training rows in manifest.")

    def _loader(rows_split: list[SampleRow], shuffle: bool):
        if not rows_split:
            return None
        ds = _STGCNDataset(
            rows_split,
            label_to_idx=label_to_idx,
            use_landmarks=use_landmarks,
        )
        if len(ds) == 0:
            return None
        collate = lambda b: _collate_stgcn(b, max_len=max_len)
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
    raise ValueError(f'split_mode must be "train_val_test" or "train_test", got {split_mode!r}')
