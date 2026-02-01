from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np


FarnebackParams = dict[str, float | int]


DEFAULT_FARNEBACK_PARAMS: FarnebackParams = dict(
    pyr_scale=0.5,
    levels=3,
    winsize=15,
    iterations=3,
    poly_n=5,
    poly_sigma=1.2,
    flags=0,
)


@dataclass(frozen=True)
class LKParams:
    winSize: tuple[int, int] = (21, 21)
    maxLevel: int = 3
    criteria: tuple[int, int, float] = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)


def farneback(prev_gray: np.ndarray, gray: np.ndarray, params: FarnebackParams | None = None) -> np.ndarray:
    p = DEFAULT_FARNEBACK_PARAMS if params is None else params
    return cv2.calcOpticalFlowFarneback(prev_gray, gray, None, **p)


def lucas_kanade_sparse(prev_gray: np.ndarray, gray: np.ndarray, params: LKParams | None = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (points_xy, flow_uv) for tracked features.
    points_xy: (N,2), flow_uv: (N,2)
    """
    p = LKParams() if params is None else params
    pts0 = cv2.goodFeaturesToTrack(prev_gray, maxCorners=300, qualityLevel=0.01, minDistance=7, blockSize=7)
    if pts0 is None:
        return np.zeros((0, 2), np.float32), np.zeros((0, 2), np.float32)
    pts1, st, _err = cv2.calcOpticalFlowPyrLK(prev_gray, gray, pts0, None, **p.__dict__)
    st = st.reshape(-1).astype(bool)
    p0 = pts0.reshape(-1, 2)[st]
    p1 = pts1.reshape(-1, 2)[st]
    return p0, (p1 - p0)


def raft_dense(prev_bgr: np.ndarray, bgr: np.ndarray) -> np.ndarray:
    """
    RAFT via torchvision (optional dependency). Returns dense flow (H,W,2) in pixels.
    """
    try:
        import torch
        from torchvision.models.optical_flow import raft_small, Raft_Small_Weights
    except Exception as e:  # pragma: no cover
        raise RuntimeError("RAFT requires extras: poetry install -E raft") from e

    weights = Raft_Small_Weights.DEFAULT
    model = raft_small(weights=weights, progress=False).eval()

    # Torchvision RAFT expects tensors in [0,1], shape (N,3,H,W)
    preprocess = weights.transforms()

    img1 = cv2.cvtColor(prev_bgr, cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    t1 = torch.from_numpy(img1).permute(2, 0, 1).float() / 255.0
    t2 = torch.from_numpy(img2).permute(2, 0, 1).float() / 255.0
    t1, t2 = preprocess(t1, t2)

    with torch.no_grad():
        # returns list of flows at different iterations; last is best
        flows = model(t1.unsqueeze(0), t2.unsqueeze(0))
        flow = flows[-1][0]  # (2,H,W)

    flow_np = flow.permute(1, 2, 0).cpu().numpy().astype(np.float32)
    return flow_np


FlowMethod = Literal["farneback", "lk", "raft"]


