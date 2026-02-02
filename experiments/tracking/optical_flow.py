"""Optical flow computation and visualization for real-time overlay."""

from __future__ import annotations

import cv2
import numpy as np


def compute_flow(
    prev_gray: np.ndarray,
    curr_gray: np.ndarray,
    method: str = "farneback",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute dense optical flow between two grayscale frames.
    Returns flow_x, flow_y (same shape as input).
    """
    if method == "farneback":
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray,
            curr_gray,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )
    else:
        # Lucas-Kanade sparse would need points; use Farneback as default
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0,
        )
    flow_x = flow[:, :, 0]
    flow_y = flow[:, :, 1]
    return flow_x, flow_y


def flow_to_rgb(flow_x: np.ndarray, flow_y: np.ndarray) -> np.ndarray:
    """Convert flow (fx, fy) to HSV then BGR for visualization (motion direction = hue)."""
    h, w = flow_x.shape
    magnitude, angle = cv2.cartToPolar(flow_x, flow_y, angleInDegrees=True)
    # Normalize magnitude for visibility
    mag_max = np.percentile(magnitude, 98)
    if mag_max > 1e-6:
        magnitude = np.clip(magnitude / mag_max, 0, 1)
    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    hsv[..., 0] = (angle / 2).astype(np.uint8)  # hue = direction
    hsv[..., 1] = 255
    hsv[..., 2] = (magnitude * 255).astype(np.uint8)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr


def overlay_flow_on_frame(
    frame: np.ndarray,
    flow_x: np.ndarray,
    flow_y: np.ndarray,
    alpha: float = 0.6,
    stride: int = 12,
    scale: float = 3.0,
    hand_boxes: list[tuple[int, int, int, int]] | None = None,
    box_padding: int = 10,
) -> np.ndarray:
    """
    Overlay optical flow on the camera frame.
    If hand_boxes is provided, show flow only inside those boxes (with box_padding);
    otherwise overlay on the full frame.
    """
    h, w = frame.shape[:2]
    flow_rgb = flow_to_rgb(flow_x, flow_y)
    if flow_rgb.shape[:2] != (h, w):
        flow_rgb = cv2.resize(flow_rgb, (w, h))
    blended = cv2.addWeighted(frame, 1 - alpha, flow_rgb, alpha, 0)

    if hand_boxes:
        # Mask: flow only inside hand boxes (with padding)
        mask = np.zeros((h, w), dtype=np.uint8)
        for (x1, y1, x2, y2) in hand_boxes:
            x1 = max(0, x1 - box_padding)
            y1 = max(0, y1 - box_padding)
            x2 = min(w, x2 + box_padding)
            y2 = min(h, y2 + box_padding)
            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
        out = frame.copy()
        out[mask > 0] = blended[mask > 0]
        # Arrows only inside hand boxes
        for y in range(stride // 2, h, stride):
            for x in range(stride // 2, w, stride):
                if mask[y, x] == 0:
                    continue
                fx = flow_x[y, x]
                fy = flow_y[y, x]
                mag = np.hypot(fx, fy)
                if mag < 0.5:
                    continue
                x2 = int(x + fx * scale)
                y2 = int(y + fy * scale)
                x2 = max(0, min(w - 1, x2))
                y2 = max(0, min(h - 1, y2))
                cv2.arrowedLine(out, (x, y), (x2, y2), (0, 255, 255), 1, tipLength=0.3)
        return out
    # Full-frame overlay (original behavior)
    out = blended
    for y in range(stride // 2, h, stride):
        for x in range(stride // 2, w, stride):
            fx = flow_x[y, x]
            fy = flow_y[y, x]
            mag = np.hypot(fx, fy)
            if mag < 0.5:
                continue
            x2 = int(x + fx * scale)
            y2 = int(y + fy * scale)
            x2 = max(0, min(w - 1, x2))
            y2 = max(0, min(h - 1, y2))
            cv2.arrowedLine(out, (x, y), (x2, y2), (0, 255, 255), 1, tipLength=0.3)
    return out
