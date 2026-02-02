#!/usr/bin/env python3
"""
Real-time two-hand and finger gesture tracking with optional optical flow overlay.
Records and draws movement trajectory of each hand keypoint; optical flow (when on) is shown only in hand boxes.
"""

from __future__ import annotations

import argparse
import collections
import sys

import cv2
import numpy as np

from hand_tracker import HandTracker
from hand_tracker import HandDetection
from optical_flow import compute_flow, overlay_flow_on_frame

# Trajectory: by handedness — index 0 = Left, 1 = Right; each is 21 landmarks × deque of (x, y)
TRAJECTORY_MAX_LEN = 60  # frames to keep per keypoint
LEFT_IDX = 0
RIGHT_IDX = 1
COLOR_LEFT = (255, 255, 0)   # BGR cyan
COLOR_RIGHT = (255, 0, 255)  # BGR magenta


def _handedness_to_idx(handedness: str | None) -> int | None:
    """Return LEFT_IDX (0) for Left, RIGHT_IDX (1) for Right, else None."""
    if not handedness:
        return None
    h = handedness.strip().lower()
    if h == "left":
        return LEFT_IDX
    if h == "right":
        return RIGHT_IDX
    return None


def _init_trajectories() -> list[list[collections.deque]]:
    return [
        [collections.deque(maxlen=TRAJECTORY_MAX_LEN) for _ in range(21)]
        for _ in range(2)
    ]


def _sample_flow(
    flow_x: np.ndarray,
    flow_y: np.ndarray,
    px: float,
    py: float,
    h: int,
    w: int,
    radius: int = 1,
) -> tuple[float, float]:
    """
    Sample flow at (px, py). If radius > 0, use median in (2*radius+1)^2 neighborhood.
    Uses subpixel interpolation when radius is 0 for accuracy. Returns (fx, fy).
    """
    if radius > 0:
        ix = max(0, min(w - 1, int(round(px))))
        iy = max(0, min(h - 1, int(round(py))))
        iy0 = max(0, iy - radius)
        iy1 = min(h, iy + radius + 1)
        ix0 = max(0, ix - radius)
        ix1 = min(w, ix + radius + 1)
        patch_x = flow_x[iy0:iy1, ix0:ix1]
        patch_y = flow_y[iy0:iy1, ix0:ix1]
        return float(np.median(patch_x)), float(np.median(patch_y))
    # Subpixel sampling via bilinear interpolation
    px_c = max(0.0, min(w - 1.001, px))
    py_c = max(0.0, min(h - 1.001, py))
    x0, y0 = int(px_c), int(py_c)
    x1, y1 = min(w - 1, x0 + 1), min(h - 1, y0 + 1)
    tx, ty = px_c - x0, py_c - y0
    fx = (1 - tx) * (1 - ty) * flow_x[y0, x0] + tx * (1 - ty) * flow_x[y0, x1]
    fx += (1 - tx) * ty * flow_x[y1, x0] + tx * ty * flow_x[y1, x1]
    fy = (1 - tx) * (1 - ty) * flow_y[y0, x0] + tx * (1 - ty) * flow_y[y0, x1]
    fy += (1 - tx) * ty * flow_y[y1, x0] + tx * ty * flow_y[y1, x1]
    return float(fx), float(fy)


# Max distance (pixels) from detection beyond which we re-anchor to detection (avoids flow drift when alpha=0)
MAX_DRIFT_PIXELS = 40

# Simulated slow detection: run detector every N frames; flow interpolates in between (key 'I')
DETECTION_INTERVAL = 5


def _update_trajectories(
    trajectories: list[list[collections.deque]],
    detections: list[HandDetection],
    h: int,
    w: int,
    flow_x: np.ndarray | None = None,
    flow_y: np.ndarray | None = None,
    flow_smooth_alpha: float = 0.9,
    max_drift_pixels: float = MAX_DRIFT_PIXELS,
) -> None:
    """
    Update trajectory buffers from detections.
    When flow is enabled: new point = blend(detection, flow_predicted).
    flow_smooth_alpha = weight on detection (1 = detection only, 0 = flow only).
    If the chosen point is more than max_drift_pixels from detection, we re-anchor
    to detection so flow-only (alpha=0) does not drift off the hand.
    """
    use_flow = (
        flow_x is not None
        and flow_y is not None
        and flow_x.shape == (h, w)
        and flow_y.shape == (h, w)
    )
    drift_thresh_sq = max_drift_pixels * max_drift_pixels
    seen = {LEFT_IDX: False, RIGHT_IDX: False}
    for det in detections:
        if det.landmarks is None:
            continue
        idx = _handedness_to_idx(det.landmarks.handedness)
        if idx is None:
            continue
        seen[idx] = True
        lm = det.landmarks.landmarks
        for j in range(min(21, len(lm))):
            px, py = lm[j, 0], lm[j, 1]
            x_det = int(px * w)
            y_det = int(py * h)
            x_det = max(0, min(w - 1, x_det))
            y_det = max(0, min(h - 1, y_det))
            if use_flow and len(trajectories[idx][j]) > 0:
                prev_x, prev_y = trajectories[idx][j][-1]
                fx, fy = _sample_flow(flow_x, flow_y, prev_x, prev_y, h, w, radius=1)
                x_pred = prev_x + fx
                y_pred = prev_y + fy
                x_smooth = flow_smooth_alpha * x_det + (1.0 - flow_smooth_alpha) * x_pred
                y_smooth = flow_smooth_alpha * y_det + (1.0 - flow_smooth_alpha) * y_pred
                x_smooth = max(0, min(w - 1, int(round(x_smooth))))
                y_smooth = max(0, min(h - 1, int(round(y_smooth))))
                # Re-anchor: if flow drifted too far from detection, snap to detection
                dx = x_smooth - x_det
                dy = y_smooth - y_det
                if dx * dx + dy * dy > drift_thresh_sq:
                    x_smooth, y_smooth = x_det, y_det
                trajectories[idx][j].append((x_smooth, y_smooth))
            else:
                trajectories[idx][j].append((x_det, y_det))
    for idx in (LEFT_IDX, RIGHT_IDX):
        if not seen[idx]:
            for j in range(21):
                trajectories[idx][j].clear()


def _update_trajectories_flow_interp(
    trajectories: list[list[collections.deque]],
    detections: list[HandDetection],
    h: int,
    w: int,
    flow_x: np.ndarray,
    flow_y: np.ndarray,
) -> None:
    """
    Flow => real-time, detection => correction.
    When detections are provided (slow detector ran this frame): append detected positions (correct).
    When detections are empty or for hands not in detections: append prev + flow(prev) (interpolate).
    """
    seen: set[int] = set()
    for det in detections:
        if det.landmarks is None:
            continue
        idx = _handedness_to_idx(det.landmarks.handedness)
        if idx is None:
            continue
        seen.add(idx)
        lm = det.landmarks.landmarks
        for j in range(min(21, len(lm))):
            px, py = lm[j, 0], lm[j, 1]
            x_det = int(px * w)
            y_det = int(py * h)
            x_det = max(0, min(w - 1, x_det))
            y_det = max(0, min(h - 1, y_det))
            trajectories[idx][j].append((x_det, y_det))
    # Interpolate with flow for hands not updated by detection (or when no detections)
    for idx in (LEFT_IDX, RIGHT_IDX):
        if idx in seen:
            continue
        for j in range(21):
            q = trajectories[idx][j]
            if len(q) == 0:
                continue
            prev_x, prev_y = q[-1]
            fx, fy = _sample_flow(flow_x, flow_y, prev_x, prev_y, h, w, radius=1)
            x_new = max(0, min(w - 1, int(round(prev_x + fx))))
            y_new = max(0, min(h - 1, int(round(prev_y + fy))))
            trajectories[idx][j].append((x_new, y_new))
    # When we had detections, clear trajectory for hands we didn't see (re-init on next detection)
    if seen:
        for idx in (LEFT_IDX, RIGHT_IDX):
            if idx not in seen:
                for j in range(21):
                    trajectories[idx][j].clear()


def _draw_trajectories(
    frame: np.ndarray,
    trajectories: list[list[collections.deque]],
    thickness: int = 2,
    color_override: tuple[int, int, int] | None = None,
) -> np.ndarray:
    """Draw trajectory polylines. If color_override is set, use it for both hands; else use L=cyan R=magenta."""
    out = frame.copy()
    if color_override is not None:
        colors = (color_override, color_override)
    else:
        colors = (COLOR_LEFT, COLOR_RIGHT)
    for hand_idx in (LEFT_IDX, RIGHT_IDX):
        color = colors[hand_idx]
        for j in range(21):
            q = trajectories[hand_idx][j]
            if len(q) < 2:
                continue
            pts = np.array(list(q), dtype=np.int32)
            cv2.polylines(out, [pts], False, color, thickness)
    return out


def run(
    camera_id: int = 0,
    width: int = 1280,
    height: int = 720,
    yolo_model: str | None = None,
) -> None:
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print("Error: Could not open camera.", file=sys.stderr)
        sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    tracker = HandTracker(yolo_model_path=yolo_model, max_hands=2)
    if tracker._hand_landmarker is None and tracker._mp_hands_legacy is None:
        print("Warning: MediaPipe hands not available — hand count will always be 0.", file=sys.stderr)
        print("  On macOS, if Tasks API failed with GPU error, try: pip install 'mediapipe>=0.9,<0.10'", file=sys.stderr)
    else:
        print("MediaPipe hands: OK (hand + finger tracking active).")
    optical_flow_enabled = False
    prev_gray: np.ndarray | None = None
    trajectories_raw = _init_trajectories()   # detection-only (non-flow)
    trajectories_flow = _init_trajectories()  # flow-smoothed (when flow on)
    window_name = "Hand Tracking"

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, actual_w, actual_h)

    # Simulated slow detection: run detector every DETECTION_INTERVAL frames; flow interpolates in between
    slow_detection_mode = False
    detection_frame_counter = 0

    print("Controls:")
    print("  O - Toggle optical flow overlay (inside hand boxes only)")
    print("  I - Toggle interpolation mode (slow detection + flow interpolation)")
    print("  C - Clear all trajectories")
    print("  Q - Quit")

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        # No flip: raw camera view (use cv2.flip(frame, 1) for mirror view)
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Hand and finger tracking (every frame in normal mode; every DETECTION_INTERVAL in slow-detection mode)
        if slow_detection_mode:
            if detection_frame_counter % DETECTION_INTERVAL == 0:
                detections = tracker.process(frame)
            else:
                detections = []
            detection_frame_counter += 1
        else:
            detections = tracker.process(frame)

        # Flow: compute when optical flow overlay is on, or when in slow-detection mode (needed for interpolation)
        flow_x, flow_y = None, None
        if (optical_flow_enabled or slow_detection_mode) and prev_gray is not None and prev_gray.shape == gray.shape:
            flow_x, flow_y = compute_flow(prev_gray, gray)

        if slow_detection_mode:
            # Optical flow => real-time; object detection => correction (only when we ran the detector)
            if flow_x is not None and flow_y is not None:
                _update_trajectories_flow_interp(trajectories_flow, detections, h, w, flow_x, flow_y)
            else:
                # First frame or no flow yet: use detection-only updates
                _update_trajectories(trajectories_flow, detections, h, w)
        elif optical_flow_enabled and flow_x is not None and flow_y is not None:
            _update_trajectories(trajectories_raw, detections, h, w, flow_x=None, flow_y=None)
            _update_trajectories(
                trajectories_flow, detections, h, w,
                flow_x=flow_x, flow_y=flow_y,
                flow_smooth_alpha=0.92,
            )
        else:
            _update_trajectories(trajectories_raw, detections, h, w)

        display = tracker.draw(frame, detections, draw_landmarks=True, draw_boxes=True)
        if slow_detection_mode:
            # Flow-interpolated trajectory (real-time); detection corrects every N frames
            display = _draw_trajectories(display, trajectories_flow, thickness=2)
        elif optical_flow_enabled:
            # Draw both: yellow = detection-only, green = flow-smoothed (compare intuitively)
            display = _draw_trajectories(
                display, trajectories_raw, thickness=2,
                color_override=(0, 255, 255),
            )  # BGR yellow
            display = _draw_trajectories(
                display, trajectories_flow, thickness=2,
                color_override=(0, 255, 0),
            )  # BGR green
        else:
            display = _draw_trajectories(display, trajectories_raw, thickness=2)

        # Optical flow overlay when enabled: only inside hand boxes
        if optical_flow_enabled and flow_x is not None and flow_y is not None:
            hand_boxes = [d.bbox for d in detections]
            display = overlay_flow_on_frame(
                display, flow_x, flow_y,
                alpha=0.5, stride=16, scale=4.0,
                hand_boxes=hand_boxes if hand_boxes else None,
                box_padding=10,
            )

        prev_gray = gray.copy()

        # Status text
        if slow_detection_mode:
            mode = f"Interp mode (det every {DETECTION_INTERVAL}) - flow=real-time, det=correction"
        else:
            mode = "Tracking + Optical flow" if optical_flow_enabled else "Tracking only"
        cv2.putText(
            display, mode, (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2,
        )
        cv2.putText(
            display, "O: flow  I: interp  C: clear  Q: quit", (10, h - 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1,
        )
        if slow_detection_mode:
            cv2.putText(
                display, f"Hands (when det ran): {len(detections)}  L=cyan R=magenta (flow-interpolated)", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1,
            )
        elif optical_flow_enabled:
            cv2.putText(
                display, f"Hands: {len(detections)}  Yellow=detection  Green=flow-smoothed", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1,
            )
        else:
            cv2.putText(
                display, f"Hands: {len(detections)}  L=cyan R=magenta", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1,
            )

        cv2.imshow(window_name, display)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == ord("Q") or key == 27:
            break
        if key == ord("o") or key == ord("O"):
            optical_flow_enabled = not optical_flow_enabled
        if key == ord("i") or key == ord("I"):
            slow_detection_mode = not slow_detection_mode
            if slow_detection_mode:
                # Clear flow trajectories so next detection re-initializes
                trajectories_flow[:] = _init_trajectories()
                detection_frame_counter = 0
        if key == ord("c") or key == ord("C"):
            trajectories_raw[:] = _init_trajectories()
            trajectories_flow[:] = _init_trajectories()

    cap.release()
    cv2.destroyAllWindows()


def main() -> None:
    p = argparse.ArgumentParser(description="Real-time hand and finger tracking with optional optical flow.")
    p.add_argument("--camera", type=int, default=0, help="Camera device id")
    p.add_argument("--width", type=int, default=1280, help="Capture width")
    p.add_argument("--height", type=int, default=720, help="Capture height")
    p.add_argument("--yolo", type=str, default=None, help="Path to YOLO hand detection .pt model (optional)")
    args = p.parse_args()
    run(camera_id=args.camera, width=args.width, height=args.height, yolo_model=args.yolo)


if __name__ == "__main__":
    main()
