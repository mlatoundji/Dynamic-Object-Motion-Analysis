from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .config import DatasetsConfig
from .manifest import sample_to_row, write_manifest_csv
from .optflow import OptFlowExtractConfig, extract_optflow_features_stream
from .pose import PoseExtractConfig, build_pose_tensor, extract_pose_stream
from .schema import OptFlowFeatures, PoseTensor, SampleIndex
from .video import iter_frames_dir, iter_video_frames, iter_video_frames_range


@dataclass(frozen=True)
class BuildOptions:
    overwrite: bool = False
    subset_limit: int = 0
    max_frames: int = 0


def build_dataset(
    cfg: DatasetsConfig,
    *,
    samples: Iterable[SampleIndex],
    out_dir: Path,
    opts: BuildOptions,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, Any]] = []

    n = 0
    for sample in samples:
        n += 1
        if opts.subset_limit and n > opts.subset_limit:
            break

        artifacts = process_sample(
            cfg,
            sample,
            out_dir=out_dir,
            overwrite=opts.overwrite,
            max_frames=opts.max_frames,
        )
        manifest_rows.append(sample_to_row(sample, artifacts=artifacts))

    manifest_path = out_dir / "manifest.csv"
    write_manifest_csv(manifest_path, manifest_rows)
    return manifest_path


def process_sample(
    cfg: DatasetsConfig,
    sample: SampleIndex,
    *,
    out_dir: Path,
    overwrite: bool,
    max_frames: int = 0,
) -> dict[str, str]:
    if sample.video_path is None:
        raise ValueError(f"sample has no video_path: {sample.sample_id}")

    vid_path = Path(sample.video_path)
    if not vid_path.exists():
        raise FileNotFoundError(f"Missing video_path: {vid_path}")

    sample_dir = out_dir / sample.dataset / sample.split / sample.sample_id
    sample_dir.mkdir(parents=True, exist_ok=True)

    pose_path = sample_dir / "pose_tensor.npz"
    flow_path = sample_dir / "optflow_features.npz"
    quality_path = sample_dir / "quality.json"

    need_pose = cfg.outputs.get("save_pose_npz", True) and (
        overwrite or not pose_path.exists()
    )
    need_flow = cfg.outputs.get("save_optflow_npz", True) and (
        overwrite or not flow_path.exists()
    )
    need_quality = cfg.outputs.get("save_quality_json", True) and (
        overwrite or not quality_path.exists()
    )

    if not (need_pose or need_flow or need_quality):
        return _artifact_dict(pose_path, flow_path, quality_path)

    def _frame_iter():
        count = 0
        start_1 = sample.frame_start
        end_1 = sample.frame_end
        if max_frames and start_1 is not None:
            # Cap the segment end by max_frames (inclusive indices).
            end_cap = int(start_1) + int(max_frames) - 1
            end_1 = min(int(end_1) if end_1 is not None else end_cap, end_cap)

        if vid_path.is_dir():
            fps = float(sample.fps or 30.0)
            it = iter_frames_dir(vid_path, fps=fps)
        else:
            if start_1 is not None and end_1 is not None:
                it = iter_video_frames_range(
                    vid_path,
                    start_1=int(start_1),
                    end_1=int(end_1),
                )
            else:
                it = iter_video_frames(vid_path)
        for fr in it:
            yield fr
            count += 1
            if max_frames and count >= max_frames:
                break

    pose_meta: dict[str, Any] | None = None
    flow_meta: dict[str, Any] | None = None
    pose_valid: np.ndarray | None = None
    flow_valid: np.ndarray | None = None

    if need_pose:
        mp = cfg.mediapipe or {}
        pose_cfg = PoseExtractConfig(
            dt_ms=cfg.dt_ms,
            backend=str(mp.get("backend", "hands")),  # type: ignore[arg-type]
            max_num_hands=int(mp.get("max_num_hands", 1)),
            min_detection_confidence=float(
                mp.get("min_detection_confidence", 0.5)
            ),
            min_tracking_confidence=float(
                mp.get("min_tracking_confidence", 0.5)
            ),
        )
        raw_pose = extract_pose_stream(_frame_iter(), cfg=pose_cfg)
        t_reg, pos, vel, acc, lms, valid, meta = build_pose_tensor(
            raw_pose, dt_ms=cfg.dt_ms
        )
        pose_valid = valid
        pose_meta = meta
        PoseTensor(
            timestamps_ms=t_reg,
            track_pos_xyz=pos,
            track_vel_xyz=vel,
            track_acc_xyz=acc,
            landmarks_xyz=lms,
            valid=valid,
            meta=meta,
        ).to_npz(pose_path)

    if need_flow:
        mp = cfg.mediapipe or {}
        flow_cfg = OptFlowExtractConfig(
            dt_ms=cfg.dt_ms,
            roi_size=cfg.roi_size,
            max_num_hands=int(mp.get("max_num_hands", 1)),
            min_detection_confidence=float(
                mp.get("min_detection_confidence", 0.5)
            ),
            min_tracking_confidence=float(
                mp.get("min_tracking_confidence", 0.5)
            ),
        )
        t_reg, feats, valid, meta = extract_optflow_features_stream(
            _frame_iter(), cfg=flow_cfg
        )
        flow_valid = valid
        flow_meta = meta
        OptFlowFeatures(
            timestamps_ms=t_reg,
            avg_speed=feats["avg_speed"],
            max_speed=feats["max_speed"],
            dominant_angle_deg=feats["dominant_angle_deg"],
            direction_concentration=feats["direction_concentration"],
            n_pixels=feats["n_pixels"],
            threshold=feats["threshold"],
            valid=valid,
            meta=meta,
        ).to_npz(flow_path)

    if need_quality:
        q = compute_quality(
            sample,
            pose_valid=pose_valid,
            flow_valid=flow_valid,
            pose_meta=pose_meta,
            flow_meta=flow_meta,
        )
        quality_path.write_text(json.dumps(q, ensure_ascii=False, indent=2), encoding="utf-8")

    return _artifact_dict(pose_path, flow_path, quality_path)


def compute_quality(
    sample: SampleIndex,
    *,
    pose_valid: np.ndarray | None,
    flow_valid: np.ndarray | None,
    pose_meta: dict[str, Any] | None,
    flow_meta: dict[str, Any] | None,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "sample_id": sample.sample_id,
        "dataset": sample.dataset,
        "split": sample.split,
        "label": sample.label,
    }
    if sample.parent_video is not None:
        out["parent_video"] = sample.parent_video
    if sample.source_annotation is not None:
        out["source_annotation"] = sample.source_annotation
    if sample.frame_start is not None:
        out["frame_start"] = int(sample.frame_start)
    if sample.frame_end is not None:
        out["frame_end"] = int(sample.frame_end)
    if pose_valid is not None:
        out["pose_valid_ratio"] = float(np.mean(pose_valid)) if pose_valid.size else 0.0
        out["pose_num_frames"] = int(pose_valid.size)
    if flow_valid is not None:
        out["optflow_valid_ratio"] = float(np.mean(flow_valid)) if flow_valid.size else 0.0
        out["optflow_num_frames"] = int(flow_valid.size)
    if pose_meta is not None:
        out["pose_meta"] = pose_meta
    if flow_meta is not None:
        out["optflow_meta"] = flow_meta
    return out


def _artifact_dict(pose_path: Path, flow_path: Path, quality_path: Path) -> dict[str, str]:
    d: dict[str, str] = {}
    if pose_path.exists():
        d["pose_npz"] = str(pose_path.as_posix())
    if flow_path.exists():
        d["optflow_npz"] = str(flow_path.as_posix())
    if quality_path.exists():
        d["quality_json"] = str(quality_path.as_posix())
    return d


def run_cmd(cmd: list[str]) -> None:
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(
            f"Command failed ({r.returncode}): {' '.join(cmd)}\n{r.stderr}"
        )
