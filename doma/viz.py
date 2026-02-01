from __future__ import annotations

import cv2
import numpy as np

from .detectors import BBox


def draw_bbox(frame: np.ndarray, bbox: BBox, color=(255, 0, 0), thickness: int = 2) -> None:
    cv2.rectangle(frame, (bbox.x, bbox.y), (bbox.x + bbox.w, bbox.y + bbox.h), color, thickness)


def draw_flow_arrows(
    frame: np.ndarray,
    flow: np.ndarray,
    origin_xy: tuple[int, int] = (0, 0),
    step: int = 10,
    scale: float = 2.0,
    mask: np.ndarray | None = None,
) -> None:
    ox, oy = origin_xy
    h, w = flow.shape[:2]
    for r in range(0, h, step):
        for c in range(0, w, step):
            if mask is not None and not bool(mask[r, c]):
                continue
            fx, fy = flow[r, c]
            start = (ox + c, oy + r)
            end = (int(ox + c + fx * scale), int(oy + r + fy * scale))
            cv2.arrowedLine(frame, start, end, (0, 255, 0), 1, tipLength=0.3)


