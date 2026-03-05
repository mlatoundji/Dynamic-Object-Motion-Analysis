from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TypedDict

import numpy as np


DatasetName = Literal["ipn_hand", "jester", "ms_asl", "wlasl", "synthetic"]
SplitName = Literal["train", "val", "test"]


class ArtifactPaths(TypedDict, total=False):
    pose_npz: str
    optflow_npz: str
    quality_json: str
    rgb_mp4: str


@dataclass(frozen=True)
class SampleIndex:
    sample_id: str
    dataset: DatasetName
    split: SplitName
    label: str
    source_uri: str
    video_path: str | None = None
    fps: float | None = None
    num_frames: int | None = None
    text: str | None = None  # gloss / translation if available
    # Optional segment information (used by IPN Hand).
    # Convention: 1-indexed frame numbers, inclusive (matches IPN annotations).
    frame_start: int | None = None
    frame_end: int | None = None
    parent_video: str | None = None
    source_annotation: str | None = None


@dataclass(frozen=True)
class PoseTensor:
    """
    Stored to NPZ via to_npz().

    track_* fields implement the spec-friendly shape:
    [t, pos(3), vel(3), acc(3)] per frame.
    landmarks_xyz is optional (T,L,3) for richer models.
    """

    timestamps_ms: np.ndarray  # (T,)
    track_pos_xyz: np.ndarray  # (T,3)
    track_vel_xyz: np.ndarray  # (T,3)
    track_acc_xyz: np.ndarray  # (T,3)
    landmarks_xyz: np.ndarray | None = None  # (T,L,3) in same coord system
    valid: np.ndarray | None = None  # (T,) boolean
    meta: dict[str, Any] | None = None

    def to_npz(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays: dict[str, Any] = {
            "timestamps_ms": self.timestamps_ms.astype(np.float32),
            "track_pos_xyz": self.track_pos_xyz.astype(np.float32),
            "track_vel_xyz": self.track_vel_xyz.astype(np.float32),
            "track_acc_xyz": self.track_acc_xyz.astype(np.float32),
        }
        if self.landmarks_xyz is not None:
            arrays["landmarks_xyz"] = self.landmarks_xyz.astype(np.float32)
        if self.valid is not None:
            arrays["valid"] = self.valid.astype(bool)
        if self.meta is not None:
            arrays["meta_json"] = np.array(
                [_json_dumps(self.meta)], dtype=object
            )
        np.savez_compressed(path, **arrays)


@dataclass(frozen=True)
class OptFlowFeatures:
    timestamps_ms: np.ndarray  # (T,)
    avg_speed: np.ndarray  # (T,)
    max_speed: np.ndarray  # (T,)
    dominant_angle_deg: np.ndarray  # (T,)
    direction_concentration: np.ndarray  # (T,)
    n_pixels: np.ndarray  # (T,)
    threshold: np.ndarray  # (T,)
    valid: np.ndarray  # (T,)
    meta: dict[str, Any] | None = None

    def to_npz(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays: dict[str, Any] = {
            "timestamps_ms": self.timestamps_ms.astype(np.float32),
            "avg_speed": self.avg_speed.astype(np.float32),
            "max_speed": self.max_speed.astype(np.float32),
            "dominant_angle_deg": self.dominant_angle_deg.astype(np.float32),
            "direction_concentration": self.direction_concentration.astype(np.float32),
            "n_pixels": self.n_pixels.astype(np.int32),
            "threshold": self.threshold.astype(np.float32),
            "valid": self.valid.astype(bool),
        }
        if self.meta is not None:
            arrays["meta_json"] = np.array(
                [_json_dumps(self.meta)], dtype=object
            )
        np.savez_compressed(path, **arrays)


def _json_dumps(obj: Any) -> str:
    import json

    return json.dumps(obj, ensure_ascii=False, sort_keys=True)
