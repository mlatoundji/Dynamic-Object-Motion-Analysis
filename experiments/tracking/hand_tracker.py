"""Hand and finger detection using MediaPipe Tasks (Hand Landmarker) and optional YOLO."""

from __future__ import annotations

import os
import sys
import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional

# MediaPipe: prefer Tasks API (0.10+); fallback to legacy solutions.hands (0.9.x) if Tasks fails
MEDIAPIPE_AVAILABLE = False
HandLandmarker = None
HandLandmarkerOptions = None
BaseOptions = None
Image = None
ImageFormat = None
_MPHandsLegacy = None  # legacy solutions.hands.Hands for mediapipe 0.9.x

try:
    from mediapipe.tasks.python import BaseOptions as _BaseOptions
    from mediapipe.tasks.python.vision import HandLandmarker as _HandLandmarker
    from mediapipe.tasks.python.vision import HandLandmarkerOptions as _HandLandmarkerOptions
    from mediapipe.tasks.python.vision.core.image import Image as _Image
    from mediapipe.tasks.python.vision.core.image import ImageFormat as _ImageFormat
    BaseOptions = _BaseOptions
    HandLandmarker = _HandLandmarker
    HandLandmarkerOptions = _HandLandmarkerOptions
    Image = _Image
    ImageFormat = _ImageFormat
    MEDIAPIPE_AVAILABLE = HandLandmarker is not None and Image is not None
except (ImportError, AttributeError):
    pass

# Legacy API (mediapipe 0.9.x) – use when Tasks API is unavailable or fails (e.g. GPU on macOS)
if _MPHandsLegacy is None:
    try:
        import mediapipe as _mp
        if hasattr(_mp, "solutions") and hasattr(_mp.solutions, "hands"):
            _MPHandsLegacy = _mp.solutions.hands.Hands
    except (ImportError, AttributeError):
        pass

# YOLO for optional hand object detection
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

# CUDA / Apple MPS device selection for YOLO (prefer CUDA, then MPS, else default/CPU)
def _get_yolo_device() -> Optional[str]:
    """Return 'cuda', 'mps', or None. Prefer CUDA, then Apple MPS, else default."""
    if not YOLO_AVAILABLE:
        return None
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) is not None:
            if torch.backends.mps.is_built() and torch.backends.mps.is_available():
                return "mps"
    except Exception:
        pass
    return None


def _get_hand_landmarker_model_path() -> Optional[str]:
    """Return path to hand_landmarker.task, downloading if needed."""
    cache_dir = os.environ.get(
        "MEDIAPIPE_CACHE",
        os.path.join(os.path.dirname(__file__), ".mediapipe_models"),
    )
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, "hand_landmarker.task")
    if os.path.isfile(path):
        return path
    url = (
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
        "hand_landmarker/float16/1/hand_landmarker.task"
    )
    try:
        import urllib.request
        print("Downloading hand_landmarker.task (one-time)...", file=sys.stderr)
        urllib.request.urlretrieve(url, path)
        return path
    except Exception as e:
        print(f"Could not download hand model: {e}", file=sys.stderr)
        return None


def _is_valid_yolo_model_path(path: Optional[str]) -> bool:
    """Return True only if path looks like a YOLO weights file (.pt) that exists."""
    if not path or not isinstance(path, str):
        return False
    path = path.strip()
    if not path.lower().endswith(".pt"):
        return False
    return os.path.isfile(path)


def _load_yolo_hand_model():
    """Load a hand-capable YOLO model (Hugging Face or default)."""
    if not YOLO_AVAILABLE:
        return None
    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(
            repo_id="keremberke/yolov8m-hand-detection",
            filename="best.pt",
        )
        if _is_valid_yolo_model_path(path):
            return YOLO(path)
    except Exception:
        pass
    try:
        return YOLO("yolov8n.pt")
    except Exception:
        return None


@dataclass
class HandLandmarks:
    """21 landmarks per hand (MediaPipe format)."""
    landmarks: np.ndarray  # (21, 3) x, y, z normalized [0, 1]
    handedness: str
    bbox: tuple[float, float, float, float]  # x_center, y_center, w, h normalized


@dataclass
class HandDetection:
    """Single hand: bbox in pixels + optional landmarks."""
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    landmarks: Optional[HandLandmarks] = None


class HandTracker:
    """Two-hand and finger tracking via MediaPipe Hand Landmarker (Tasks API or legacy)."""

    def __init__(
        self,
        yolo_model_path: Optional[str] = None,
        yolo_conf: float = 0.5,
        max_hands: int = 2,
    ):
        self.yolo_conf = yolo_conf
        self.max_hands = max_hands
        self._yolo = None
        self._yolo_hand_classes: set[str] = set()
        self._yolo_device: Optional[str] = _get_yolo_device()
        self._hand_landmarker = None
        self._mp_hands_legacy = None

        # YOLO (optional, hands-only) – only load from a valid .pt file path
        if YOLO_AVAILABLE:
            if _is_valid_yolo_model_path(yolo_model_path):
                try:
                    self._yolo = YOLO(yolo_model_path)
                except Exception:
                    self._yolo = _load_yolo_hand_model()
            else:
                self._yolo = _load_yolo_hand_model()
            if self._yolo is not None:
                names = getattr(self._yolo, "names", None) or getattr(self._yolo.model, "names", None) or {}
                if isinstance(names, dict):
                    name_list = list(names.values())
                else:
                    name_list = list(names) if names else []
                self._yolo_hand_classes = {
                    str(n).lower() for n in name_list
                    if n and "hand" in str(n).lower()
                }
                if not self._yolo_hand_classes:
                    self._yolo = None
                elif self._yolo_device:
                    print(f"YOLO using {self._yolo_device.upper()}.", file=sys.stderr)

        # MediaPipe Hand Landmarker (Tasks API)
        if MEDIAPIPE_AVAILABLE and HandLandmarker is not None:
            model_path = _get_hand_landmarker_model_path()
            if model_path:
                try:
                    base_opts = BaseOptions(
                        model_asset_path=model_path,
                        delegate=BaseOptions.Delegate.CPU,
                    )
                    options = HandLandmarkerOptions(
                        base_options=base_opts,
                        num_hands=max_hands,
                        min_hand_detection_confidence=0.4,
                        min_hand_presence_confidence=0.4,
                        min_tracking_confidence=0.4,
                    )
                    self._hand_landmarker = HandLandmarker.create_from_options(options)
                except Exception as e:
                    print(f"MediaPipe Hand Landmarker (Tasks API) failed: {e}", file=sys.stderr)
            if self._hand_landmarker is None and _MPHandsLegacy is not None:
                try:
                    self._mp_hands_legacy = _MPHandsLegacy(
                        static_image_mode=False,
                        max_num_hands=max_hands,
                        min_detection_confidence=0.5,
                        min_tracking_confidence=0.4,
                    )
                    print("Using MediaPipe legacy hands (solutions.hands).", file=sys.stderr)
                except Exception as e:
                    print(f"MediaPipe legacy hands failed: {e}", file=sys.stderr)

    def _yolo_detect(self, frame: np.ndarray) -> list[tuple[int, int, int, int, float]]:
        """Return list of (x1, y1, x2, y2, conf) for hand classes only."""
        if self._yolo is None or not self._yolo_hand_classes:
            return []
        kwargs = {"conf": self.yolo_conf, "verbose": False}
        if self._yolo_device is not None:
            kwargs["device"] = self._yolo_device
        results = self._yolo(frame, **kwargs)[0]
        boxes = []
        for box in results.boxes:
            xyxy = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            name = (results.names.get(cls_id) or "").lower()
            if name in self._yolo_hand_classes:
                x1, y1, x2, y2 = map(int, xyxy)
                boxes.append((x1, y1, x2, y2, conf))
        return boxes

    def _mediapipe_detect(self, frame_rgb: np.ndarray) -> list[HandLandmarks]:
        """Detect hands using MediaPipe (Tasks API or legacy solutions.hands)."""
        rgb = np.ascontiguousarray(frame_rgb)
        if rgb.dtype != np.uint8:
            rgb = rgb.astype(np.uint8)

        # Legacy API (mediapipe 0.9.x)
        if self._mp_hands_legacy is not None:
            out = self._mp_hands_legacy.process(rgb)
            hands = []
            if out.multi_hand_landmarks and out.multi_handedness:
                for lm, handedness in zip(out.multi_hand_landmarks, out.multi_handedness):
                    arr = np.array([[p.x, p.y, p.z] for p in lm.landmark], dtype=np.float64)
                    label = handedness.classification[0].label
                    xs, ys = arr[:, 0], arr[:, 1]
                    x_c, y_c = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
                    w = max(xs.max() - xs.min(), 0.01)
                    h = max(ys.max() - ys.min(), 0.01)
                    hands.append(HandLandmarks(landmarks=arr, handedness=label, bbox=(x_c, y_c, w, h)))
            return hands[: self.max_hands]

        # Tasks API (mediapipe 0.10+)
        if self._hand_landmarker is None:
            return []
        try:
            mp_image = Image(ImageFormat.SRGB, rgb)
            result = self._hand_landmarker.detect(mp_image)
        except Exception:
            return []
        hands = []
        if not result.hand_landmarks:
            return []
        for i, hand_lms in enumerate(result.hand_landmarks):
            arr = np.array([
                [getattr(p, "x", 0) or 0, getattr(p, "y", 0) or 0, getattr(p, "z", 0) or 0]
                for p in hand_lms
            ], dtype=np.float64)
            if arr.shape[0] < 21:
                continue
            label = "Left"
            if result.handedness and i < len(result.handedness) and result.handedness[i]:
                cat = result.handedness[i][0]
                label = getattr(cat, "category_name", None) or "Left"
            xs, ys = arr[:, 0], arr[:, 1]
            x_c = (xs.min() + xs.max()) / 2
            y_c = (ys.min() + ys.max()) / 2
            w = max(xs.max() - xs.min(), 0.01)
            h = max(ys.max() - ys.min(), 0.01)
            hands.append(HandLandmarks(landmarks=arr, handedness=label, bbox=(x_c, y_c, w, h)))
        return hands[: self.max_hands]

    def process(self, frame: np.ndarray) -> list[HandDetection]:
        """Run hand detection; return list of HandDetection (bbox pixels + landmarks)."""
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        yolo_boxes = self._yolo_detect(frame)
        mp_hands = self._mediapipe_detect(rgb)
        detections = []
        for hand in mp_hands:
            x_c, y_c, bw, bh = hand.bbox
            x1 = int((x_c - bw / 2) * w)
            y1 = int((y_c - bh / 2) * h)
            x2 = int((x_c + bw / 2) * w)
            y2 = int((y_c + bh / 2) * h)
            x1, x2 = max(0, x1), min(w, x2)
            y1, y2 = max(0, y1), min(h, y2)
            detections.append(
                HandDetection(bbox=(x1, y1, x2, y2), confidence=0.9, landmarks=hand)
            )
        if yolo_boxes and len(detections) < 2:
            for (x1, y1, x2, y2, conf) in yolo_boxes:
                if not detections or not any(
                    self._iou((x1, y1, x2, y2), d.bbox) > 0.5 for d in detections
                ):
                    detections.append(
                        HandDetection(bbox=(x1, y1, x2, y2), confidence=conf, landmarks=None)
                    )
        return detections[: self.max_hands]

    @staticmethod
    def _iou(box1: tuple[int, ...], box2: tuple[int, ...]) -> float:
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)
        inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        a1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        a2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        return inter / (a1 + a2 - inter + 1e-6)

    def draw(
        self,
        frame: np.ndarray,
        detections: list[HandDetection],
        draw_landmarks: bool = True,
        draw_boxes: bool = True,
    ) -> np.ndarray:
        """Draw bounding boxes and hand skeleton on frame. Left=cyan, Right=magenta (BGR)."""
        out = frame.copy()
        h, w = out.shape[:2]
        color_left = (255, 255, 0)   # BGR cyan
        color_right = (255, 0, 255)  # BGR magenta
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
            (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16),
            (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
        ]
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            handedness = det.landmarks.handedness if det.landmarks else None
            is_left = handedness and handedness.strip().lower() == "left"
            box_color = color_left if is_left else color_right
            label = "L" if is_left else ("R" if handedness else "?")
            if draw_boxes:
                cv2.rectangle(out, (x1, y1), (x2, y2), box_color, 2)
                cv2.putText(
                    out, f"{label} {det.confidence:.2f}", (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 1,
                )
            if draw_landmarks and det.landmarks is not None:
                lm = det.landmarks.landmarks
                for idx in range(min(len(lm), 21)):
                    px, py = lm[idx, 0], lm[idx, 1]
                    ix, iy = int(px * w), int(py * h)
                    cv2.circle(out, (ix, iy), 4, box_color, -1)
                for i, j in connections:
                    if i < len(lm) and j < len(lm):
                        p1 = (int(lm[i, 0] * w), int(lm[i, 1] * h))
                        p2 = (int(lm[j, 0] * w), int(lm[j, 1] * h))
                        cv2.line(out, p1, p2, box_color, 2)
        return out
