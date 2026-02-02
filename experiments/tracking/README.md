# Real-time two-hand and finger gesture tracking

Prototype for real-time hand and finger gesture tracking using a PC camera. It uses **object detection (YOLO)** and **optical flow**, with a minimal UI to compare tracking with and without optical flow.

## Features

- **Hand detection**: YOLO-style object detection for hands (optional hand-specific YOLO model; MediaPipe used for robust hand + finger detection).
- **Finger tracking**: 21 landmarks per hand via MediaPipe (wrist, fingers, joints).
- **Left vs right hand**: Each hand is labeled **L** (Left) or **R** (Right) from MediaPipe handedness. **Left** = cyan box and trajectory; **Right** = magenta box and trajectory.
- **Keypoint trajectories**: Movement of each of the 21 hand landmarks is recorded and drawn as trails (cyan for left hand, magenta for right) over the last ~60 frames. **When optical flow is enabled**, trajectory points are **smoothed using optical flow**: each new point is a blend of the detected landmark position and the position predicted by applying flow at the previous point (reduces jitter).
- **Optical flow**: When enabled (**O**), dense Farneback optical flow is shown **only inside hand bounding boxes** (not the full frame). Toggle **O** to compare tracking with and without flow.

## Requirements

- Python 3.11–3.12 (3.14 not yet supported by some dependencies)
- Webcam

## Setup

```bash
uv sync
# or: pip install -e .
```

## Usage

```bash
uv run python main.py
# or: python main.py
```

**Controls**

- **O** – Toggle optical flow overlay (compare with/without flow).
- **Q** or **Esc** – Quit.

**Optional**

- `--camera 0` – Camera device id.
- `--width 1280 --height 720` – Capture resolution.
- `--yolo path/to/hand_model.pt` – Custom YOLO hand detection model (e.g. [yolov8n.pt](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8n.pt)).

## Architecture

- **hand_tracker.py**: Hand detection (YOLO when a hand model is available) and finger landmarks (MediaPipe). Merges detections and draws boxes + 21-point hand skeleton.
- **optical_flow.py**: Computes dense optical flow (Farneback) and visualizes it (colormap + motion arrows) over the frame.
- **main.py**: Camera capture, hand/finger tracking, optional optical flow overlay, and minimal UI (status text, key controls).

## Notes

- **CUDA / MPS**: YOLO inference uses **CUDA** when available (NVIDIA GPU), otherwise **MPS** on Apple Silicon (M1/M2/M3), else CPU.
- Hand detection uses **MediaPipe**: the app tries the Tasks API (Hand Landmarker) first, then falls back to the legacy **solutions.hands** API if the Tasks API fails (e.g. GPU/OpenGL errors on macOS).
- **If you always see 0 hands**: On macOS, MediaPipe 0.10 may fail to init the Hand Landmarker (GPU service). Install the legacy version: `pip install 'mediapipe>=0.9,<0.10'` so the app can use `solutions.hands` instead.
- If no YOLO hand model is provided, hand boxes and finger landmarks come from MediaPipe only.
- For YOLO hand-only detection you can train or download a hand YOLOv8 model and pass it with `--yolo`.
