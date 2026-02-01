from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np


@dataclass(frozen=True)
class BBox:
    x: int
    y: int
    w: int
    h: int
    score: float = 1.0


Detector = Literal["manual", "mediapipe", "yolo"]


def clip_bbox(b: BBox, width: int, height: int) -> BBox:
    x = max(0, min(b.x, width - 1))
    y = max(0, min(b.y, height - 1))
    w = max(1, min(b.w, width - x))
    h = max(1, min(b.h, height - y))
    return BBox(x=x, y=y, w=w, h=h, score=b.score)


class ManualROI:
    def __init__(self) -> None:
        self._drawing = False
        self._ix = -1
        self._iy = -1
        self._rect: BBox | None = None

    @property
    def bbox(self) -> BBox | None:
        return self._rect

    def reset(self) -> None:
        self._rect = None

    def mouse_cb(self, event, x, y, flags, param) -> None:  # noqa: ANN001
        if event == cv2.EVENT_LBUTTONDOWN:
            self._drawing = True
            self._ix, self._iy = x, y
            self._rect = None
        elif event == cv2.EVENT_MOUSEMOVE and self._drawing:
            self._rect = BBox(x=min(self._ix, x), y=min(self._iy, y), w=abs(x - self._ix), h=abs(y - self._iy))
        elif event == cv2.EVENT_LBUTTONUP:
            self._drawing = False
            rect = BBox(x=min(self._ix, x), y=min(self._iy, y), w=abs(x - self._ix), h=abs(y - self._iy))
            self._rect = rect if (rect.w >= 10 and rect.h >= 10) else None


class MediaPipeHandsDetector:
    def __init__(self, max_num_hands: int = 1) -> None:
        try:
            import mediapipe as mp
        except Exception as e:  # pragma: no cover
            raise RuntimeError("MediaPipe requires extras: poetry install -E hand") from e

        self._mp = mp
        self._hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def detect(self, frame_bgr: np.ndarray) -> tuple[BBox | None, np.ndarray | None]:
        """
        Returns (bbox, mask) where mask is a boolean ROI mask (convex hull of landmarks).
        """
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        res = self._hands.process(rgb)
        if not res.multi_hand_landmarks:
            return None, None

        lm = res.multi_hand_landmarks[0].landmark
        pts = np.array([(int(p.x * w), int(p.y * h)) for p in lm], dtype=np.int32)
        x, y, bw, bh = cv2.boundingRect(pts)
        bbox = clip_bbox(BBox(x=x, y=y, w=bw, h=bh, score=1.0), width=w, height=h)

        hull = cv2.convexHull(pts)
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(mask, hull, 255)
        return bbox, (mask > 0)


class YOLODetector:
    def __init__(self, model: str = "yolov8n.pt", conf: float = 0.25) -> None:
        try:
            from ultralytics import YOLO
        except Exception as e:  # pragma: no cover
            raise RuntimeError("YOLO requires extras: poetry install -E yolo") from e

        self._yolo = YOLO(model)
        self._conf = float(conf)

    def detect(self, frame_bgr: np.ndarray) -> BBox | None:
        """
        NOTE: This is a generic detector; you must use a model that can detect hands,
        or adjust class filtering based on your model's label set.
        """
        h, w = frame_bgr.shape[:2]
        res = self._yolo.predict(frame_bgr, conf=self._conf, verbose=False)
        if not res or res[0].boxes is None or len(res[0].boxes) == 0:
            return None

        boxes = res[0].boxes
        # Take best-scoring box.
        i = int(np.argmax(boxes.conf.cpu().numpy()))
        xyxy = boxes.xyxy[i].cpu().numpy()
        score = float(boxes.conf[i].cpu().numpy())
        x1, y1, x2, y2 = [int(v) for v in xyxy]
        return clip_bbox(BBox(x=x1, y=y1, w=max(1, x2 - x1), h=max(1, y2 - y1), score=score), width=w, height=h)


