from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np


@dataclass(frozen=True)
class MotionStats:
    avg_speed: float
    max_speed: float
    dominant_angle_deg: float
    direction_concentration: float  # in [0,1]
    n_pixels: int
    threshold: float


def magnitude_angle(flow: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    fx = flow[..., 0].astype(np.float32)
    fy = flow[..., 1].astype(np.float32)
    mag = np.hypot(fx, fy)
    ang = np.arctan2(fy, fx)  # radians, [-pi, pi]
    return mag, ang


def _otsu_threshold(mag: np.ndarray) -> float:
    mag = mag.astype(np.float32)
    if mag.size == 0:
        return 0.0
    # Normalize to 8-bit for Otsu.
    mmax = float(np.max(mag))
    if mmax <= 1e-6:
        return 0.0
    mag_u8 = np.clip((mag / mmax) * 255.0, 0, 255).astype(np.uint8)
    # OpenCV returns (threshold_value, thresholded_image).
    # With Otsu, we need the threshold value.
    thr_u8, _ = cv2.threshold(
        mag_u8, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
    )
    return (float(thr_u8) / 255.0) * mmax


def _mad_threshold(mag: np.ndarray, k: float = 3.0) -> float:
    mag = mag.astype(np.float32)
    if mag.size == 0:
        return 0.0
    med = float(np.median(mag))
    mad = float(np.median(np.abs(mag - med)))
    # 1.4826 * MAD approximates sigma for Gaussian noise.
    return med + k * 1.4826 * mad


def choose_threshold(
    mag: np.ndarray,
    method: Literal["fixed", "otsu", "mad"] = "otsu",
    fixed_value: float = 2.0,
) -> float:
    if method == "fixed":
        return float(fixed_value)
    if method == "otsu":
        return _otsu_threshold(mag)
    if method == "mad":
        return _mad_threshold(mag)
    raise ValueError(f"Unknown threshold method: {method}")


def largest_connected_component(mask: np.ndarray) -> np.ndarray:
    mask_u8 = (mask.astype(np.uint8) * 255)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask_u8, connectivity=8
    )
    if n <= 1:
        return mask.astype(bool)
    # Skip label 0 (background)
    areas = stats[1:, cv2.CC_STAT_AREA]
    best = int(np.argmax(areas)) + 1
    return labels == best


def refine_motion_mask(mask: np.ndarray) -> np.ndarray:
    mask_u8 = (mask.astype(np.uint8) * 255)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel, iterations=1)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel, iterations=1)
    return (mask_u8 > 0)


def subtract_global_motion(
    flow: np.ndarray, bg_mask: np.ndarray | None
) -> tuple[np.ndarray, np.ndarray]:
    fx = flow[..., 0].astype(np.float32)
    fy = flow[..., 1].astype(np.float32)

    if bg_mask is None or int(np.count_nonzero(bg_mask)) < 50:
        u0 = float(np.median(fx))
        v0 = float(np.median(fy))
    else:
        u0 = float(np.median(fx[bg_mask]))
        v0 = float(np.median(fy[bg_mask]))

    corrected = flow.copy().astype(np.float32)
    corrected[..., 0] -= u0
    corrected[..., 1] -= v0
    return corrected, np.array([u0, v0], dtype=np.float32)


def circular_mean_angle_rad(
    ang: np.ndarray, weights: np.ndarray | None = None
) -> tuple[float, float]:
    """
    Returns (mean_angle_rad, concentration_R).
    R in [0,1] measures directional concentration.
    """
    if ang.size == 0:
        return 0.0, 0.0
    if weights is None:
        w = np.ones_like(ang, dtype=np.float32)
    else:
        w = weights.astype(np.float32)
    c = float(np.sum(w * np.cos(ang)))
    s = float(np.sum(w * np.sin(ang)))
    denom = float(np.sum(w))
    if denom <= 1e-6:
        return 0.0, 0.0
    mean = float(np.arctan2(s, c))
    r = float(np.hypot(c, s) / denom)
    return mean, r


def compute_motion_stats(
    flow: np.ndarray,
    threshold_method: Literal["fixed", "otsu", "mad"] = "otsu",
    fixed_threshold: float = 2.0,
    min_pixels: int = 50,
    subtract_bg: bool = True,
) -> tuple[MotionStats, np.ndarray]:
    """
    Returns (stats, motion_mask) for the given flow field
    (typically already cropped to ROI/mask).
    """
    mag, ang = magnitude_angle(flow)
    thr = choose_threshold(
        mag, method=threshold_method, fixed_value=fixed_threshold
    )
    motion = mag > thr
    motion = refine_motion_mask(motion)
    if int(np.count_nonzero(motion)) > 0:
        motion = largest_connected_component(motion)

    if subtract_bg:
        bg_mask = ~motion
        flow2, _ = subtract_global_motion(flow, bg_mask=bg_mask)
        mag, ang = magnitude_angle(flow2)
        motion = mag > thr
        motion = refine_motion_mask(motion)
        if int(np.count_nonzero(motion)) > 0:
            motion = largest_connected_component(motion)
        flow = flow2

    n = int(np.count_nonzero(motion))
    if n < min_pixels:
        return (
            MotionStats(
                avg_speed=0.0,
                max_speed=0.0,
                dominant_angle_deg=0.0,
                direction_concentration=0.0,
                n_pixels=n,
                threshold=float(thr),
            ),
            motion,
        )

    speeds = mag[motion]
    mean_ang, r = circular_mean_angle_rad(ang[motion], weights=speeds)
    dominant_deg = (np.degrees(mean_ang) + 360.0) % 360.0

    return (
        MotionStats(
            avg_speed=float(np.mean(speeds)),
            max_speed=float(np.max(speeds)),
            dominant_angle_deg=float(dominant_deg),
            direction_concentration=float(r),
            n_pixels=n,
            threshold=float(thr),
        ),
        motion,
    )


def angle_to_cardinal(angle_deg: float) -> str:
    # OpenCV/cartesian convention: 0° = +x (right).
    # In image coordinates, y points down.
    dirs = [
        "Est",
        "Sud-Est",
        "Sud",
        "Sud-Ouest",
        "Ouest",
        "Nord-Ouest",
        "Nord",
        "Nord-Est",
    ]
    idx = int(((float(angle_deg) + 22.5) / 45.0)) % 8
    return dirs[idx]
