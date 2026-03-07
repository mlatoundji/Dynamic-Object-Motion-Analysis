from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

try:  # torch is an optional dependency
    import torch
    from torch.utils.data import Dataset
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    Dataset = object  # type: ignore[misc,assignment]


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
    mean: np.ndarray  # (F,)
    std: np.ndarray  # (F,)

    def to_npz(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            mean=self.mean.astype(np.float32),
            std=self.std.astype(np.float32),
        )

    @staticmethod
    def from_npz(path: Path) -> "NormStats":
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
    manifest_csv: Path, *, repo_root: Path | None = None
) -> list[SampleRow]:
    root = repo_root or Path.cwd()
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
            # Store paths relative to repo root for portability.
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


def save_label_map(path: Path, label_to_idx: dict[str, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "label_to_idx": dict(label_to_idx),
        "idx_to_label": {str(v): k for k, v in label_to_idx.items()},
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def load_label_map(path: Path) -> dict[str, int]:
    d = json.loads(path.read_text(encoding="utf-8"))
    m = d.get("label_to_idx")
    if not isinstance(m, dict):
        raise ValueError("Invalid label map json: missing label_to_idx")
    out: dict[str, int] = {}
    for k, v in m.items():
        if not isinstance(k, str) or not isinstance(v, int):
            continue
        out[k] = v
    if not out:
        raise ValueError("Invalid label map json: empty")
    return out


def _pose_features(
    pose_npz: Path, *, use_landmarks: bool
) -> tuple[np.ndarray, np.ndarray]:
    d = np.load(pose_npz, allow_pickle=True)
    pos = np.asarray(d["track_pos_xyz"], dtype=np.float32)
    vel = np.asarray(d["track_vel_xyz"], dtype=np.float32)
    acc = np.asarray(d["track_acc_xyz"], dtype=np.float32)
    T = int(pos.shape[0])
    valid = np.asarray(d.get("valid", np.ones((T,), dtype=bool)), dtype=bool)
    feats = [pos, vel, acc]
    if use_landmarks and "landmarks_xyz" in d.files:
        lm = np.asarray(d["landmarks_xyz"], dtype=np.float32)  # (T,L,3)
        feats.append(lm.reshape(T, -1))
    x = np.concatenate(feats, axis=1)  # (T,Fp)
    return x, valid


def _optflow_features(
    opt_npz: Path, *, angle_as_sincos: bool
) -> tuple[np.ndarray, np.ndarray]:
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
    Returns (x, valid_mask) where:
    - x: (T,F) float32
    - valid_mask: (T,) bool (joint pose+flow validity if both enabled)
    """
    pose_x = None
    pose_valid = None
    if feat_cfg.include_pose:
        pose_x, pose_valid = _pose_features(Path(row.pose_npz), use_landmarks=feat_cfg.use_landmarks)

    flow_x = None
    flow_valid = None
    if feat_cfg.include_optflow:
        flow_x, flow_valid = _optflow_features(Path(row.optflow_npz), angle_as_sincos=feat_cfg.angle_as_sincos)

    if pose_x is None and flow_x is None:
        raise ValueError("At least one of include_pose/include_optflow must be True")

    if pose_x is None:
        x = flow_x  # type: ignore[assignment]
        valid = flow_valid  # type: ignore[assignment]
    elif flow_x is None:
        x = pose_x
        valid = pose_valid  # type: ignore[assignment]
    else:
        # Align lengths defensively.
        T = min(int(pose_x.shape[0]), int(flow_x.shape[0]))
        x = np.concatenate([pose_x[:T], flow_x[:T]], axis=1)
        valid = pose_valid[:T] & flow_valid[:T]

    x = np.asarray(x, dtype=np.float32)
    valid = np.asarray(valid, dtype=bool)

    # Safety: ensure we never mark a timestep valid if any feature is non-finite
    # (e.g., derivatives can produce NaNs even when positions are valid).
    finite = np.isfinite(x).all(axis=1)
    valid &= finite

    # Standardize if provided.
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
    """
    Compute mean/std over all valid timesteps in the given rows.
    """
    n_used = 0
    sum_x = None
    sum_x2 = None
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
        # Extra safety (should already be finite if valid, but keep it robust)
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


class GestureDataset(Dataset):  # type: ignore[misc]
    def __init__(
        self,
        rows: list[SampleRow],
        *,
        label_to_idx: dict[str, int],
        feat_cfg: FeatureConfig,
        norm: NormStats | None = None,
    ) -> None:
        if torch is None:  # pragma: no cover
            raise RuntimeError("PyTorch is required for GestureDataset (install with extra 'train' or 'raft').")
        self._rows = [r for r in rows if r.label in label_to_idx]
        self._label_to_idx = dict(label_to_idx)
        self._feat_cfg = feat_cfg
        self._norm = norm

    def __len__(self) -> int:
        return int(len(self._rows))

    def __getitem__(self, idx: int) -> dict[str, Any]:
        r = self._rows[int(idx)]
        x, valid = load_sample_features(r, feat_cfg=self._feat_cfg, norm=self._norm)
        # Drop invalid timesteps entirely to allow pack_padded_sequence.
        if int(np.count_nonzero(valid)) > 0:
            x = x[valid]
        else:
            x = x[:0]
        y = int(self._label_to_idx[r.label])
        return {"x": x.astype(np.float32), "y": y, "sample_id": r.sample_id, "label": r.label}


def collate_padded(batch: list[dict[str, Any]]) -> dict[str, Any]:
    if torch is None:  # pragma: no cover
        raise RuntimeError("PyTorch required")
    xs = [np.asarray(b["x"], dtype=np.float32) for b in batch]
    ys = torch.tensor([int(b["y"]) for b in batch], dtype=torch.long)
    sample_ids = [str(b["sample_id"]) for b in batch]
    labels = [str(b["label"]) for b in batch]

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
        "label": labels,
    }

