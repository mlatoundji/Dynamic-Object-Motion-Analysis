from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import cv2
import numpy as np

from ..detectors import MediaPipeHandsDetector, clip_bbox
from ..flow import farneback
from ..motion import compute_motion_stats
from .timeseries import ResampleResult, resample_linear
from .video import resize_keep_aspect


@dataclass(frozen=True)
class OptFlowExtractConfig:
    dt_ms: float
    roi_size: tuple[int, int] = (224, 224)  # (H,W)
    threshold_method: Literal["fixed", "otsu", "mad"] = "otsu"
    fixed_threshold: float = 2.0
    subtract_bg: bool = True
    max_num_hands: int = 1
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5


def extract_optflow_features(
    frames_bgr: list[np.ndarray],
    timestamps_ms: np.ndarray,
    *,
    cfg: OptFlowExtractConfig,
) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray, dict[str, Any]]:
    """
    Returns (t_ms_regular, features_dict, valid, meta).
    """
    det = MediaPipeHandsDetector(
        max_num_hands=cfg.max_num_hands,
        min_detection_confidence=cfg.min_detection_confidence,
        min_tracking_confidence=cfg.min_tracking_confidence,
    )
    t_ms = np.asarray(timestamps_ms, dtype=np.float64)
    T = int(len(frames_bgr))

    avg = np.full((T,), np.nan, dtype=np.float64)
    mx = np.full((T,), np.nan, dtype=np.float64)
    ang_deg = np.full((T,), np.nan, dtype=np.float64)
    conc = np.full((T,), np.nan, dtype=np.float64)
    npx = np.full((T,), 0, dtype=np.int32)
    thr = np.full((T,), np.nan, dtype=np.float64)
    valid = np.zeros((T,), dtype=bool)

    prev_roi = None
    prev_mask = None

    for i, frame in enumerate(frames_bgr):
        bbox, mask = det.detect(frame)
        if bbox is None:
            prev_roi = None
            prev_mask = None
            continue

        bbox = clip_bbox(bbox, width=frame.shape[1], height=frame.shape[0])
        roi = frame[bbox.y : bbox.y + bbox.h, bbox.x : bbox.x + bbox.w]
        roi = resize_keep_aspect(roi, size_hw=cfg.roi_size)
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        roi_mask = None
        if mask is not None:
            m = mask[bbox.y : bbox.y + bbox.h, bbox.x : bbox.x + bbox.w].astype(np.uint8) * 255
            roi_mask = resize_keep_aspect(m, size_hw=cfg.roi_size) > 0

        if prev_roi is None:
            prev_roi = roi_gray
            prev_mask = roi_mask
            continue

        flow = farneback(prev_roi, roi_gray)
        if roi_mask is not None:
            flow = flow.copy()
            flow[~roi_mask] = 0.0

        stats, motion_mask = compute_motion_stats(
            flow,
            threshold_method=cfg.threshold_method,
            fixed_threshold=cfg.fixed_threshold,
            subtract_bg=cfg.subtract_bg,
        )

        avg[i] = float(stats.avg_speed)
        mx[i] = float(stats.max_speed)
        ang_deg[i] = float(stats.dominant_angle_deg)
        conc[i] = float(stats.direction_concentration)
        npx[i] = int(stats.n_pixels)
        thr[i] = float(stats.threshold)
        valid[i] = bool(stats.avg_speed > 0 and int(np.count_nonzero(motion_mask)) > 0)

        prev_roi = roi_gray
        prev_mask = roi_mask

    # Resample to regular grid: scalars via linear interpolation; angle via sin/cos to avoid wrap issues.
    r_avg: ResampleResult = resample_linear(t_ms, avg, dt_ms=cfg.dt_ms, axis_time=0)
    r_mx: ResampleResult = resample_linear(t_ms, mx, dt_ms=cfg.dt_ms, axis_time=0)
    r_conc: ResampleResult = resample_linear(t_ms, conc, dt_ms=cfg.dt_ms, axis_time=0)
    r_thr: ResampleResult = resample_linear(t_ms, thr, dt_ms=cfg.dt_ms, axis_time=0)

    ang_rad = np.deg2rad(ang_deg)
    r_sin: ResampleResult = resample_linear(t_ms, np.sin(ang_rad), dt_ms=cfg.dt_ms, axis_time=0)
    r_cos: ResampleResult = resample_linear(t_ms, np.cos(ang_rad), dt_ms=cfg.dt_ms, axis_time=0)
    ang_out = (np.rad2deg(np.arctan2(r_sin.y, r_cos.y)) + 360.0) % 360.0

    # For integers/bools, derive validity from scalar validity and then nearest-time sample for n_pixels.
    valid_out = r_avg.valid & r_mx.valid & r_conc.valid & r_thr.valid & r_sin.valid & r_cos.valid
    r_npx: ResampleResult = resample_linear(t_ms, npx.astype(np.float64), dt_ms=cfg.dt_ms, axis_time=0)
    npx_out = np.where(valid_out & r_npx.valid, np.round(r_npx.y).astype(np.int32), 0)

    feats: dict[str, np.ndarray] = {
        "avg_speed": r_avg.y.astype(np.float32),
        "max_speed": r_mx.y.astype(np.float32),
        "dominant_angle_deg": ang_out.astype(np.float32),
        "direction_concentration": r_conc.y.astype(np.float32),
        "n_pixels": npx_out,
        "threshold": r_thr.y.astype(np.float32),
    }

    meta: dict[str, Any] = {
        "roi_size_hw": [int(cfg.roi_size[0]), int(cfg.roi_size[1])],
        "threshold_method": str(cfg.threshold_method),
        "fixed_threshold": float(cfg.fixed_threshold),
        "subtract_bg": bool(cfg.subtract_bg),
        "resample_dt_ms": float(cfg.dt_ms),
    }

    return r_avg.t_ms, feats, valid_out.astype(bool), meta

