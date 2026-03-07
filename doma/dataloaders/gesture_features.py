"""
Unified gesture feature extraction from pose/optflow NPZ files.
Shared by doma.dataloaders.dataloader (and CNN-LSTM runner / live classifier).
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np


Split = Literal["train", "val", "test"]


@dataclass(frozen=True)
class SampleRow:
    sample_id: str
    split: Split
    label: str
    pose_npz: str
    optflow_npz: str


@dataclass(frozen=True)
class FeatureConfig:
    use_landmarks: bool = True
    include_optflow: bool = True
    include_pose: bool = True
    angle_as_sincos: bool = True


@dataclass(frozen=True)
class NormStats:
    mean: np.ndarray
    std: np.ndarray

    def to_npz(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            mean=self.mean.astype(np.float32),
            std=self.std.astype(np.float32),
        )

    @staticmethod
    def from_npz(path: Path) -> NormStats:
        d = np.load(path, allow_pickle=True)
        return NormStats(
            mean=np.asarray(d["mean"], dtype=np.float32),
            std=np.asarray(d["std"], dtype=np.float32),
        )


def _resolve_path(p: str, *, repo_root: Path) -> Path:
    pp = Path(p)
    if pp.is_absolute():
        return pp
    return repo_root / pp


def read_manifest_rows(
    manifest_csv: Path | str, *, repo_root: Path | None = None
) -> list[SampleRow]:
    root = Path(repo_root or Path.cwd())
    manifest_csv = Path(manifest_csv)
    rows: list[SampleRow] = []
    with manifest_csv.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            sample_id = (row.get("sample_id") or "").strip()
            split = (row.get("split") or "").strip()
            label = (row.get("label") or "").strip()
            pose_npz = (row.get("pose_npz") or "").strip()
            opt_npz = (row.get("optflow_npz") or "").strip()
            if not sample_id or split not in {"train", "val", "test"} or not label:
                continue
            if not pose_npz or not opt_npz:
                continue
            pose_abs = _resolve_path(pose_npz, repo_root=root)
            opt_abs = _resolve_path(opt_npz, repo_root=root)
            if not pose_abs.exists() or not opt_abs.exists():
                continue
            rows.append(
                SampleRow(
                    sample_id=sample_id,
                    split=split,  # type: ignore[arg-type]
                    label=label,
                    pose_npz=str(pose_abs.as_posix()),
                    optflow_npz=str(opt_abs.as_posix()),
                )
            )
    return rows


def build_label_map(rows: list[SampleRow]) -> dict[str, int]:
    labels = sorted({r.label for r in rows})
    return {lab: i for i, lab in enumerate(labels)}


def pose_features(
    pose_npz: Path | str, *, use_landmarks: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """Extract pose features (pos, vel, acc, optional landmarks). Returns (x, valid)."""
    d = np.load(pose_npz, allow_pickle=True)
    pos = np.asarray(d["track_pos_xyz"], dtype=np.float32)
    vel = np.asarray(d["track_vel_xyz"], dtype=np.float32)
    acc = np.asarray(d["track_acc_xyz"], dtype=np.float32)
    T = int(pos.shape[0])
    valid = np.asarray(d.get("valid", np.ones((T,), dtype=bool)), dtype=bool)
    feats = [pos, vel, acc]
    if use_landmarks and "landmarks_xyz" in d.files:
        lm = np.asarray(d["landmarks_xyz"], dtype=np.float32)
        feats.append(lm.reshape(T, -1))
    x = np.concatenate(feats, axis=1)
    return x, valid


def optflow_features(
    opt_npz: Path | str, *, angle_as_sincos: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """Extract optflow features (avg, max, angle sin/cos, concentration, n_pixels, threshold). Returns (x, valid)."""
    d = np.load(opt_npz, allow_pickle=True)
    avg = np.asarray(d["avg_speed"], dtype=np.float32).reshape(-1, 1)
    mx = np.asarray(d["max_speed"], dtype=np.float32).reshape(-1, 1)
    ang_deg = np.asarray(d["dominant_angle_deg"], dtype=np.float32).reshape(-1)
    conc = np.asarray(d["direction_concentration"], dtype=np.float32).reshape(-1, 1)
    npx = np.asarray(d["n_pixels"], dtype=np.float32).reshape(-1, 1)
    thr = np.asarray(d["threshold"], dtype=np.float32).reshape(-1, 1)
    valid = np.asarray(
        d.get("valid", np.ones((avg.shape[0],), dtype=bool)), dtype=bool
    )
    if angle_as_sincos:
        ang = np.deg2rad(ang_deg)
        sin = np.sin(ang).astype(np.float32).reshape(-1, 1)
        cos = np.cos(ang).astype(np.float32).reshape(-1, 1)
        x = np.concatenate([avg, mx, sin, cos, conc, npx, thr], axis=1)
    else:
        x = np.concatenate(
            [avg, mx, ang_deg.reshape(-1, 1), conc, npx, thr],
            axis=1,
        )
    return x, valid


def load_sample_features(
    row: SampleRow,
    *,
    feat_cfg: FeatureConfig,
    norm: NormStats | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build unified feature vector for one sample. Returns (x, valid).
    x shape (T, F) after aligning pose and optflow; only valid frames should be used.
    """
    pose_x, pose_valid = None, None
    if feat_cfg.include_pose:
        pose_x, pose_valid = pose_features(
            Path(row.pose_npz), use_landmarks=feat_cfg.use_landmarks
        )

    flow_x, flow_valid = None, None
    if feat_cfg.include_optflow:
        flow_x, flow_valid = optflow_features(
            Path(row.optflow_npz), angle_as_sincos=feat_cfg.angle_as_sincos
        )

    if pose_x is None and flow_x is None:
        raise ValueError("At least one of include_pose/include_optflow must be True")

    if pose_x is None:
        x, valid = flow_x, flow_valid  # type: ignore[assignment]
    elif flow_x is None:
        x, valid = pose_x, pose_valid  # type: ignore[assignment]
    else:
        T = min(int(pose_x.shape[0]), int(flow_x.shape[0]))
        x = np.concatenate([pose_x[:T], flow_x[:T]], axis=1)
        valid = pose_valid[:T] & flow_valid[:T]  # type: ignore[operator]

    x = np.asarray(x, dtype=np.float32)
    valid = np.asarray(valid, dtype=bool)
    finite = np.isfinite(x).all(axis=1)
    valid &= finite

    if norm is not None:
        if norm.mean.shape[0] != x.shape[1] or norm.std.shape[0] != x.shape[1]:
            raise ValueError("NormStats dimension mismatch")
        std = np.where(norm.std > 1e-8, norm.std, 1.0).astype(np.float32)
        x = (x - norm.mean.astype(np.float32).reshape(1, -1)) / std.reshape(1, -1)

    return x, valid


def compute_norm_stats(
    rows: list[SampleRow],
    *,
    feat_cfg: FeatureConfig,
    label_to_idx: dict[str, int],
    max_samples: int = 0,
) -> NormStats:
    """Compute mean/std over training samples for normalization."""
    n_used = 0
    sum_x = sum_x2 = None
    count = 0
    for r in rows:
        if max_samples and n_used >= max_samples:
            break
        if r.label not in label_to_idx:
            continue
        x, valid = load_sample_features(r, feat_cfg=feat_cfg, norm=None)
        if int(np.count_nonzero(valid)) == 0:
            continue
        xv = x[valid]
        xv = xv[np.isfinite(xv).all(axis=1)]
        if xv.size == 0:
            continue
        if sum_x is None:
            sum_x = np.zeros((xv.shape[1],), dtype=np.float64)
            sum_x2 = np.zeros((xv.shape[1],), dtype=np.float64)
        sum_x += np.sum(xv, axis=0, dtype=np.float64)
        sum_x2 += np.sum(xv * xv, axis=0, dtype=np.float64)
        count += int(xv.shape[0])
        n_used += 1
    if sum_x is None or sum_x2 is None or count <= 1:
        raise RuntimeError("Unable to compute normalization stats (not enough valid data)")
    mean = (sum_x / float(count)).astype(np.float32)
    var = (sum_x2 / float(count)) - (mean.astype(np.float64) ** 2)
    var = np.maximum(var, 1e-8)
    std = np.sqrt(var).astype(np.float32)
    return NormStats(mean=mean, std=std)
