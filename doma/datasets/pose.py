from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from ..detectors import BBox, MediaPipeHandsDetector, clip_bbox
from .timeseries import ResampleResult, finite_diff, resample_linear


PoseBackend = Literal["hands", "holistic"]


@dataclass(frozen=True)
class PoseExtractConfig:
    dt_ms: float
    backend: PoseBackend = "hands"
    max_num_hands: int = 1
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    rotation_normalize: bool = True
    spatial_origin: Literal["first_wrist", "image_center", "mid_shoulders"] = "first_wrist"


@dataclass(frozen=True)
class PoseExtractResult:
    t_ms: np.ndarray  # (T,)
    bbox_xywh: np.ndarray  # (T,4) int; -1 if missing
    track_xyz: np.ndarray  # (T,3) float; NaN if missing
    landmarks_xyz: np.ndarray  # (T,L,3) float; NaN if missing
    valid: np.ndarray  # (T,)
    meta: dict[str, Any]


def _rotation_matrix_z(theta: float) -> np.ndarray:
    c = float(np.cos(theta))
    s = float(np.sin(theta))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def _normalize_rotation(track_xyz: np.ndarray, landmarks_xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Align the palm direction (wrist -> middle_mcp) to +x in the xy-plane.
    """
    # MediaPipe Hands landmark indices: 0=wrist, 9=middle_mcp
    wrist = landmarks_xyz[:, 0, :]
    middle = landmarks_xyz[:, 9, :]
    v = middle - wrist
    # angle in xy plane
    ang = np.arctan2(v[:, 1], v[:, 0])
    out_track = track_xyz.copy()
    out_lm = landmarks_xyz.copy()
    for i in range(track_xyz.shape[0]):
        if not np.isfinite(ang[i]):
            continue
        r = _rotation_matrix_z(-float(ang[i]))
        if np.all(np.isfinite(out_track[i])):
            out_track[i] = (r @ out_track[i].reshape(3, 1)).reshape(3)
        lm = out_lm[i]
        ok = np.isfinite(lm).all(axis=1)
        if int(np.count_nonzero(ok)) > 0:
            out_lm[i, ok] = (r @ lm[ok].T).T
    return out_track, out_lm


def extract_pose_hands(
    frames_bgr: list[np.ndarray],
    timestamps_ms: np.ndarray,
    *,
    cfg: PoseExtractConfig,
) -> PoseExtractResult:
    det = MediaPipeHandsDetector(
        max_num_hands=cfg.max_num_hands,
        min_detection_confidence=cfg.min_detection_confidence,
        min_tracking_confidence=cfg.min_tracking_confidence,
    )

    t_ms = np.asarray(timestamps_ms, dtype=np.float64)
    T = int(len(frames_bgr))
    L = 21

    bboxes = np.full((T, 4), -1, dtype=np.int32)
    track = np.full((T, 3), np.nan, dtype=np.float64)
    lms = np.full((T, L, 3), np.nan, dtype=np.float64)
    valid = np.zeros((T,), dtype=bool)

    origin = None

    for i, frame in enumerate(frames_bgr):
        bbox, _mask, lm_xyz = det.detect_with_landmarks(frame)
        if bbox is None or lm_xyz is None:
            continue

        bbox = clip_bbox(bbox, width=frame.shape[1], height=frame.shape[0])
        bboxes[i] = np.array([bbox.x, bbox.y, bbox.w, bbox.h], dtype=np.int32)

        pts = lm_xyz.astype(np.float64)  # normalized
        lms[i] = pts
        track[i] = pts[0]  # wrist as track point
        valid[i] = True

        if cfg.spatial_origin == "first_wrist" and origin is None and np.all(np.isfinite(track[i])):
            origin = track[i].copy()

    # Spatial normalization (constant origin per clip)
    if cfg.spatial_origin == "first_wrist":
        if origin is None:
            origin = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        track = track - origin.reshape(1, 3)
        lms = lms - origin.reshape(1, 1, 3)
    elif cfg.spatial_origin == "image_center":
        # normalized coords => (0.5,0.5) is center
        origin = np.array([0.5, 0.5, 0.0], dtype=np.float64)
        track = track - origin.reshape(1, 3)
        lms = lms - origin.reshape(1, 1, 3)

    if cfg.rotation_normalize:
        track, lms = _normalize_rotation(track, lms)

    meta: dict[str, Any] = {
        "backend": "mediapipe_hands",
        "track_point": "wrist",
        "coord_system": "mediapipe_normalized",
        "origin": (origin.tolist() if origin is not None else None),
        "rotation_normalize": bool(cfg.rotation_normalize),
        "spatial_origin": str(cfg.spatial_origin),
    }
    return PoseExtractResult(t_ms=t_ms, bbox_xywh=bboxes, track_xyz=track, landmarks_xyz=lms, valid=valid, meta=meta)


def extract_pose_holistic(
    frames_bgr: list[np.ndarray],
    timestamps_ms: np.ndarray,
    *,
    cfg: PoseExtractConfig,
) -> PoseExtractResult:
    """
    MediaPipe Holistic (pose + left/right hands).
    """
    try:
        import mediapipe as mp
    except Exception as e:  # pragma: no cover
        raise RuntimeError("MediaPipe requires extras: poetry install -E hand") from e

    hol = mp.solutions.holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        refine_face_landmarks=False,
        min_detection_confidence=float(cfg.min_detection_confidence),
        min_tracking_confidence=float(cfg.min_tracking_confidence),
    )

    t_ms = np.asarray(timestamps_ms, dtype=np.float64)
    T = int(len(frames_bgr))
    L_pose, L_hand = 33, 21
    L = L_pose + 2 * L_hand

    bboxes = np.full((T, 4), -1, dtype=np.int32)
    track = np.full((T, 3), np.nan, dtype=np.float64)
    lms = np.full((T, L, 3), np.nan, dtype=np.float64)
    valid = np.zeros((T,), dtype=bool)

    origin = None

    for i, frame in enumerate(frames_bgr):
        rgb = frame[:, :, ::-1]  # BGR->RGB
        res = hol.process(rgb)
        if res.pose_landmarks is None:
            continue

        pose = res.pose_landmarks.landmark
        pose_xyz = np.array([[float(p.x), float(p.y), float(p.z)] for p in pose], dtype=np.float64)

        lh_xyz = np.full((L_hand, 3), np.nan, dtype=np.float64)
        rh_xyz = np.full((L_hand, 3), np.nan, dtype=np.float64)
        if res.left_hand_landmarks is not None:
            lh = res.left_hand_landmarks.landmark
            lh_xyz = np.array([[float(p.x), float(p.y), float(p.z)] for p in lh], dtype=np.float64)
        if res.right_hand_landmarks is not None:
            rh = res.right_hand_landmarks.landmark
            rh_xyz = np.array([[float(p.x), float(p.y), float(p.z)] for p in rh], dtype=np.float64)

        lms[i] = np.concatenate([pose_xyz, lh_xyz, rh_xyz], axis=0)

        # Choose a stable body origin (spec §5.2 suggests chest/neck reference).
        # We approximate it by mid-shoulders in normalized coordinates.
        ls = pose_xyz[11]
        rs = pose_xyz[12]
        if np.all(np.isfinite(ls)) and np.all(np.isfinite(rs)):
            o = 0.5 * (ls + rs)
        else:
            o = np.array([0.5, 0.5, 0.0], dtype=np.float64)
        if origin is None:
            origin = o.copy()

        # Track right-wrist (pose idx 16) relative to origin.
        rw = pose_xyz[16]
        if np.all(np.isfinite(rw)):
            track[i] = rw - origin
            valid[i] = True

    if origin is None:
        origin = np.array([0.0, 0.0, 0.0], dtype=np.float64)

    # Spatial normalization: subtract constant origin from all landmarks.
    lms = lms - origin.reshape(1, 1, 3)

    if cfg.rotation_normalize:
        # Align shoulder axis to +x (after origin subtraction).
        ls = lms[:, 11, :]
        rs = lms[:, 12, :]
        v = rs - ls
        ang = np.arctan2(v[:, 1], v[:, 0])
        out_track = track.copy()
        out_lm = lms.copy()
        for i in range(T):
            if not np.isfinite(ang[i]):
                continue
            r = _rotation_matrix_z(-float(ang[i]))
            if np.all(np.isfinite(out_track[i])):
                out_track[i] = (r @ out_track[i].reshape(3, 1)).reshape(3)
            lm = out_lm[i]
            ok = np.isfinite(lm).all(axis=1)
            if int(np.count_nonzero(ok)) > 0:
                out_lm[i, ok] = (r @ lm[ok].T).T
        track, lms = out_track, out_lm

    meta: dict[str, Any] = {
        "backend": "mediapipe_holistic",
        "track_point": "pose_right_wrist",
        "coord_system": "mediapipe_normalized",
        "origin": origin.tolist(),
        "rotation_normalize": bool(cfg.rotation_normalize),
        "spatial_origin": "mid_shoulders",
    }
    return PoseExtractResult(t_ms=t_ms, bbox_xywh=bboxes, track_xyz=track, landmarks_xyz=lms, valid=valid, meta=meta)


def extract_pose(
    frames_bgr: list[np.ndarray],
    timestamps_ms: np.ndarray,
    *,
    cfg: PoseExtractConfig,
) -> PoseExtractResult:
    if cfg.backend == "hands":
        return extract_pose_hands(frames_bgr, timestamps_ms, cfg=cfg)
    if cfg.backend == "holistic":
        return extract_pose_holistic(frames_bgr, timestamps_ms, cfg=cfg)
    raise ValueError(f"Unknown backend: {cfg.backend}")


def build_pose_tensor(
    pose: PoseExtractResult,
    *,
    dt_ms: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """
    Returns (t_ms, track_pos, track_vel, track_acc, landmarks_xyz, valid)
    all resampled to a regular grid.
    """
    # Resample track & landmarks to regular dt
    r_track: ResampleResult = resample_linear(pose.t_ms, pose.track_xyz, dt_ms=dt_ms, axis_time=0)
    r_lm: ResampleResult = resample_linear(pose.t_ms, pose.landmarks_xyz, dt_ms=dt_ms, axis_time=0)

    # Derivatives
    vel = finite_diff(r_track.t_ms, r_track.y, axis_time=0)
    acc = finite_diff(r_track.t_ms, vel, axis_time=0)

    valid = r_track.valid & r_lm.valid
    meta = dict(pose.meta)
    meta["resample_dt_ms"] = float(dt_ms)
    return r_track.t_ms, r_track.y, vel, acc, r_lm.y, valid, meta

