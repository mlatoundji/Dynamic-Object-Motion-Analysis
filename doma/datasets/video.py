from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


@dataclass(frozen=True)
class VideoFrame:
    idx: int
    t_ms: float
    bgr: np.ndarray


def iter_video_frames(path: Path) -> Iterator[VideoFrame]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps <= 1e-6:
        fps = 30.0

    idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t_ms = (idx / fps) * 1000.0
            yield VideoFrame(idx=idx, t_ms=float(t_ms), bgr=frame)
            idx += 1
    finally:
        cap.release()


def iter_frames_dir(dir_path: Path, *, fps: float = 30.0, pattern: str = "*.jpg") -> Iterator[VideoFrame]:
    paths = sorted(dir_path.glob(pattern))
    if not paths:
        # Try png
        paths = sorted(dir_path.glob("*.png"))
    if not paths:
        raise RuntimeError(f"No frames found in {dir_path}")
    fps = float(fps) if fps and fps > 0 else 30.0

    for idx, p in enumerate(paths):
        img = cv2.imread(str(p))
        if img is None:
            continue
        t_ms = (idx / fps) * 1000.0
        yield VideoFrame(idx=idx, t_ms=float(t_ms), bgr=img)


def resize_keep_aspect(img: np.ndarray, size_hw: tuple[int, int]) -> np.ndarray:
    th, tw = int(size_hw[0]), int(size_hw[1])
    h, w = img.shape[:2]
    if h <= 0 or w <= 0:
        return np.zeros((th, tw) + img.shape[2:], img.dtype)
    scale = min(tw / w, th / h)
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)

    if img.ndim == 2:
        canvas = np.zeros((th, tw), dtype=img.dtype)
    else:
        canvas = np.zeros((th, tw, img.shape[2]), dtype=img.dtype)
    y0 = (th - nh) // 2
    x0 = (tw - nw) // 2
    canvas[y0 : y0 + nh, x0 : x0 + nw] = resized
    return canvas

