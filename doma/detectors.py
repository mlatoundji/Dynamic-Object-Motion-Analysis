from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
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
            self._rect = BBox(
                x=min(self._ix, x),
                y=min(self._iy, y),
                w=abs(x - self._ix),
                h=abs(y - self._iy),
            )
        elif event == cv2.EVENT_LBUTTONUP:
            self._drawing = False
            rect = BBox(
                x=min(self._ix, x),
                y=min(self._iy, y),
                w=abs(x - self._ix),
                h=abs(y - self._iy),
            )
            self._rect = rect if (rect.w >= 10 and rect.h >= 10) else None


class MediaPipeHandsDetector:
    def __init__(
        self,
        max_num_hands: int = 1,
        *,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        model_complexity: int = 1,
        model_path: str | None = None,
        auto_download_model: bool = True,
    ) -> None:
        try:
            import mediapipe as mp
        except Exception as e:  # pragma: no cover
            raise RuntimeError("MediaPipe requires extras: uv sync --extra hand") from e

        self._mp = mp
        self._mode: Literal["solutions", "tasks"]
        self._hands = None
        self._landmarker = None

        if hasattr(mp, "solutions"):
            self._mode = "solutions"
            self._hands = mp.solutions.hands.Hands(
                static_image_mode=False,
                max_num_hands=max_num_hands,
                model_complexity=int(model_complexity),
                min_detection_confidence=float(min_detection_confidence),
                min_tracking_confidence=float(min_tracking_confidence),
            )
        else:
            # Newer MediaPipe builds may ship Tasks-only (no `mp.solutions`).
            self._mode = "tasks"
            mp_root = Path(__file__).resolve().parents[1]
            models_dir = mp_root / ".mediapipe_models"
            models_dir.mkdir(parents=True, exist_ok=True)

            if model_path is None:
                model_path = str((models_dir / "hand_landmarker.task").resolve())
            model_file = Path(model_path)

            if (not model_file.exists()) and auto_download_model:
                model_url = (
                    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
                    "hand_landmarker/float16/1/hand_landmarker.task"
                )
                _download_file(
                    url=model_url,
                    dst=model_file,
                )

            if not model_file.exists():
                raise RuntimeError(
                    "MediaPipe Tasks requires a HandLandmarker model file. "
                    f"Download it to `{model_file}` or pass `model_path=...`."
                )

            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision

            base_options = python.BaseOptions(model_asset_path=str(model_file))
            options = vision.HandLandmarkerOptions(
                base_options=base_options,
                num_hands=int(max_num_hands),
                min_hand_detection_confidence=float(min_detection_confidence),
                min_tracking_confidence=float(min_tracking_confidence),
            )
            self._landmarker = vision.HandLandmarker.create_from_options(options)

    def detect(
        self, frame_bgr: np.ndarray
    ) -> tuple[BBox | None, np.ndarray | None]:
        """
        Returns (bbox, mask) where mask is a boolean ROI mask (convex hull of landmarks).
        """
        bbox, mask, _lm_xyz = self.detect_with_landmarks(frame_bgr)
        return bbox, mask

    def detect_with_landmarks(
        self, frame_bgr: np.ndarray
    ) -> tuple[BBox | None, np.ndarray | None, np.ndarray | None]:
        """
        Returns (bbox, mask, landmarks_xyz) where:
        - mask is a boolean ROI mask (convex hull of landmarks) in full-frame coordinates
        - landmarks_xyz is (21,3) in MediaPipe normalized coords (x,y in [0,1], z relative)
        """
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        if self._mode == "solutions":
            assert self._hands is not None
            res = self._hands.process(rgb)
            if not res.multi_hand_landmarks:
                return None, None, None
            lm = res.multi_hand_landmarks[0].landmark
            lm_xyz = np.array(
                [[float(p.x), float(p.y), float(p.z)] for p in lm],
                dtype=np.float32,
            )
        else:
            assert self._landmarker is not None
            mp = self._mp
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            det = self._landmarker.detect(image)
            if not det.hand_landmarks:
                return None, None, None
            lm = det.hand_landmarks[0]
            lm_xyz = np.array(
                [[float(p.x), float(p.y), float(p.z)] for p in lm],
                dtype=np.float32,
            )

        pts = np.array(
            [(int(p[0] * w), int(p[1] * h)) for p in lm_xyz],
            dtype=np.int32,
        )
        x, y, bw, bh = cv2.boundingRect(pts)
        bbox = clip_bbox(
            BBox(x=x, y=y, w=bw, h=bh, score=1.0),
            width=w,
            height=h,
        )

        hull = cv2.convexHull(pts)
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(mask, hull, 255)
        return bbox, (mask > 0), lm_xyz


def _download_file(url: str, dst: Path) -> None:
    import urllib.request

    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    urllib.request.urlretrieve(url, tmp)  # noqa: S310
    tmp.replace(dst)


class YOLODetector:
    def __init__(self, model: str = "yolov8n.pt", conf: float = 0.25) -> None:
        try:
            from ultralytics import YOLO
        except Exception as e:  # pragma: no cover
            raise RuntimeError("YOLO requires extras: uv sync --extra yolo") from e

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
        return clip_bbox(
            BBox(
                x=x1,
                y=y1,
                w=max(1, x2 - x1),
                h=max(1, y2 - y1),
                score=score,
            ),
            width=w,
            height=h,
        )
