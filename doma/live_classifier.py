from __future__ import annotations

import argparse
import json
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from .detectors import MediaPipeHandsDetector, clip_bbox
from .datasets.timeseries import finite_diff, resample_linear
from .datasets.video import resize_keep_aspect
from .flow import farneback
from .motion import compute_motion_stats

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]

from .train.data import FeatureConfig, NormStats
from .train.model import CNNLSTMClassifier, ModelConfig


def _load_bundle(run_dir: Path) -> tuple[Path, NormStats, dict[str, int], float]:
    run_dir = run_dir.resolve()
    ckpt = run_dir / "checkpoints" / "best.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt}")

    norm = NormStats.from_npz(run_dir / "norm.npz")
    label_map_path = run_dir / "label_map.json"
    label_map = json.loads(label_map_path.read_text(encoding="utf-8")).get("label_to_idx", {})
    if not isinstance(label_map, dict) or not label_map:
        raise ValueError("Invalid label_map.json")
    label_to_idx = {str(k): int(v) for k, v in label_map.items()}

    dt_ms = 33.333
    tc = run_dir / "train_config.json"
    if tc.exists():
        try:
            dt_ms = float(json.loads(tc.read_text(encoding="utf-8")).get("dt_ms", dt_ms))
        except Exception:
            pass
    return ckpt, norm, label_to_idx, float(dt_ms)


def _rotation_matrix_z(theta: float) -> np.ndarray:
    c = float(np.cos(theta))
    s = float(np.sin(theta))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def _normalize_pose_frame(landmarks_xyz: np.ndarray, *, origin: np.ndarray | None) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """
    landmarks_xyz: (21,3) float64 in MediaPipe normalized coords.
    Returns (track_xyz, landmarks_xyz_norm, origin_out).
    """
    pts = landmarks_xyz.astype(np.float64)
    wrist = pts[0].copy()
    if origin is None and np.all(np.isfinite(wrist)):
        origin = wrist.copy()
    if origin is None:
        origin = np.array([0.0, 0.0, 0.0], dtype=np.float64)

    pts = pts - origin.reshape(1, 3)
    track = pts[0].copy()

    # Rotation normalize: align wrist->middle_mcp to +x.
    middle = pts[9].copy()
    v = middle - track
    ang = float(np.arctan2(v[1], v[0])) if np.all(np.isfinite(v)) else float("nan")
    if np.isfinite(ang):
        r = _rotation_matrix_z(-ang)
        pts = (r @ pts.T).T
        track = (r @ track.reshape(3, 1)).reshape(3)

    return track, pts, origin


class LiveBuffer:
    def __init__(self, *, max_ms: float) -> None:
        self.max_ms = float(max_ms)
        self.t_ms: deque[float] = deque()
        self.track_xyz: deque[np.ndarray] = deque()
        self.lm_xyz: deque[np.ndarray] = deque()
        self.pose_valid: deque[bool] = deque()

        self.avg_speed: deque[float] = deque()
        self.max_speed: deque[float] = deque()
        self.ang_deg: deque[float] = deque()
        self.conc: deque[float] = deque()
        self.npx: deque[float] = deque()
        self.thr: deque[float] = deque()
        self.flow_valid: deque[bool] = deque()

    def push(
        self,
        *,
        t_ms: float,
        track_xyz: np.ndarray,
        lm_xyz: np.ndarray,
        pose_valid: bool,
        flow_feats: dict[str, float],
        flow_valid: bool,
    ) -> None:
        self.t_ms.append(float(t_ms))
        self.track_xyz.append(np.asarray(track_xyz, dtype=np.float64))
        self.lm_xyz.append(np.asarray(lm_xyz, dtype=np.float64))
        self.pose_valid.append(bool(pose_valid))

        self.avg_speed.append(float(flow_feats.get("avg_speed", np.nan)))
        self.max_speed.append(float(flow_feats.get("max_speed", np.nan)))
        self.ang_deg.append(float(flow_feats.get("dominant_angle_deg", np.nan)))
        self.conc.append(float(flow_feats.get("direction_concentration", np.nan)))
        self.npx.append(float(flow_feats.get("n_pixels", 0.0)))
        self.thr.append(float(flow_feats.get("threshold", np.nan)))
        self.flow_valid.append(bool(flow_valid))

        self._trim()

    def _trim(self) -> None:
        while len(self.t_ms) >= 2:
            span = float(self.t_ms[-1] - self.t_ms[0])
            if span <= self.max_ms:
                break
            self.t_ms.popleft()
            self.track_xyz.popleft()
            self.lm_xyz.popleft()
            self.pose_valid.popleft()
            self.avg_speed.popleft()
            self.max_speed.popleft()
            self.ang_deg.popleft()
            self.conc.popleft()
            self.npx.popleft()
            self.thr.popleft()
            self.flow_valid.popleft()

    def build_sequence(
        self, *, dt_ms: float, feat_cfg: FeatureConfig
    ) -> tuple[np.ndarray, int]:
        """
        Returns (x, length) for a single sample, standardized is handled outside.
        """
        if len(self.t_ms) < 2:
            return np.zeros((0, 1), dtype=np.float32), 0

        t = np.asarray(self.t_ms, dtype=np.float64)
        track = np.stack(list(self.track_xyz), axis=0)  # (T,3)
        lm = np.stack(list(self.lm_xyz), axis=0)  # (T,21,3)

        # Pose resample + vel/acc (matches dataset build logic).
        r_track = resample_linear(t, track, dt_ms=float(dt_ms), axis_time=0)
        r_lm = resample_linear(t, lm, dt_ms=float(dt_ms), axis_time=0)
        vel = finite_diff(r_track.t_ms, r_track.y, axis_time=0)
        acc = finite_diff(r_track.t_ms, vel, axis_time=0)
        pose_valid = r_track.valid & r_lm.valid

        feats = []
        if feat_cfg.include_pose:
            feats.append(r_track.y.astype(np.float32))
            feats.append(vel.astype(np.float32))
            feats.append(acc.astype(np.float32))
            if feat_cfg.use_landmarks:
                feats.append(r_lm.y.astype(np.float32).reshape(r_lm.y.shape[0], -1))

        # Optflow resample
        if feat_cfg.include_optflow:
            avg = resample_linear(
                t,
                np.asarray(self.avg_speed, dtype=np.float64),
                dt_ms=float(dt_ms),
                axis_time=0,
            )
            mx = resample_linear(
                t,
                np.asarray(self.max_speed, dtype=np.float64),
                dt_ms=float(dt_ms),
                axis_time=0,
            )
            conc = resample_linear(
                t,
                np.asarray(self.conc, dtype=np.float64),
                dt_ms=float(dt_ms),
                axis_time=0,
            )
            thr = resample_linear(
                t,
                np.asarray(self.thr, dtype=np.float64),
                dt_ms=float(dt_ms),
                axis_time=0,
            )

            ang = np.deg2rad(np.asarray(self.ang_deg, dtype=np.float64))
            s = resample_linear(t, np.sin(ang), dt_ms=float(dt_ms), axis_time=0)
            c = resample_linear(t, np.cos(ang), dt_ms=float(dt_ms), axis_time=0)
            ang_out = (np.rad2deg(np.arctan2(s.y, c.y)) + 360.0) % 360.0

            npx = resample_linear(
                t,
                np.asarray(self.npx, dtype=np.float64),
                dt_ms=float(dt_ms),
                axis_time=0,
            )
            npx_out = np.round(npx.y).astype(np.float32)

            flow_valid = avg.valid & mx.valid & conc.valid & thr.valid & s.valid & c.valid

            feats.append(avg.y.astype(np.float32).reshape(-1, 1))
            feats.append(mx.y.astype(np.float32).reshape(-1, 1))
            feats.append(np.sin(np.deg2rad(ang_out)).astype(np.float32).reshape(-1, 1))
            feats.append(np.cos(np.deg2rad(ang_out)).astype(np.float32).reshape(-1, 1))
            feats.append(conc.y.astype(np.float32).reshape(-1, 1))
            feats.append(npx_out.reshape(-1, 1))
            feats.append(thr.y.astype(np.float32).reshape(-1, 1))
        else:
            flow_valid = np.ones_like(pose_valid, dtype=bool)

        x = (
            np.concatenate(feats, axis=1)
            if feats
            else np.zeros((0, 1), dtype=np.float32)
        )
        valid = pose_valid & flow_valid & np.isfinite(x).all(axis=1)
        if int(np.count_nonzero(valid)) == 0:
            return x[:0], 0
        xv = x[valid]
        return xv.astype(np.float32), int(xv.shape[0])


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="PoC: live hand gesture classification (CNN-LSTM)")
    p.add_argument("--run", required=True, help="Training run directory (runs/...)")
    p.add_argument("--source", default="0", help="Camera index or video path")
    p.add_argument("--window-ms", type=float, default=1500.0)
    p.add_argument("--infer-every-ms", type=float, default=200.0)
    p.add_argument("--ema", type=float, default=0.6, help="EMA smoothing for probabilities (0=no smoothing)")
    p.add_argument("--d0x-thr", type=float, default=0.6, help="If P(D0X) >= thr, display D0X as idle")
    p.add_argument("--roi", type=int, default=224, help="ROI size (square) for optflow features")
    p.add_argument("--no-landmarks", action="store_true")
    p.add_argument("--no-optflow", action="store_true")
    p.add_argument("--no-pose", action="store_true")
    args = p.parse_args(argv)

    if torch is None:
        raise SystemExit("PyTorch is required. Install with: poetry install -E train -E hand")

    ckpt_path, norm, label_to_idx, dt_ms = _load_bundle(Path(args.run))
    idx_to_label = {v: k for k, v in label_to_idx.items()}

    ckpt = torch.load(ckpt_path, map_location="cpu")
    mcfg = ModelConfig(**ckpt["model_config"])
    model = CNNLSTMClassifier(mcfg)
    model.load_state_dict(ckpt["model"])
    model.eval()

    feat_cfg = FeatureConfig(
        use_landmarks=not bool(args.no_landmarks),
        include_optflow=not bool(args.no_optflow),
        include_pose=not bool(args.no_pose),
        angle_as_sincos=True,
    )

    # Source
    try:
        cam_idx = int(args.source)
        cap = cv2.VideoCapture(cam_idx)
    except ValueError:
        cap = cv2.VideoCapture(str(args.source))
    if not cap.isOpened():
        raise SystemExit("Cannot open video source")

    win = "DOMA - Live Classifier"
    cv2.namedWindow(win)

    det = MediaPipeHandsDetector(max_num_hands=1)
    roi_size = (int(args.roi), int(args.roi))
    prev_roi_gray = None
    origin = None

    buf = LiveBuffer(max_ms=float(args.window_ms))
    last_infer = 0.0
    ema_probs = None

    t0 = time.time()
    frame_idx = 0
    fps_ema = None

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        disp = frame.copy()

        now = time.time()
        fps = 0.0
        if frame_idx > 0:
            fps = 1.0 / max(1e-6, (now - t0))
        t0 = now
        fps_ema = fps if fps_ema is None else (0.9 * fps_ema + 0.1 * fps)
        frame_idx += 1

        bbox, mask, lm_xyz = det.detect_with_landmarks(frame)
        pose_valid = bbox is not None and lm_xyz is not None

        track = np.full((3,), np.nan, dtype=np.float64)
        lm = np.full((21, 3), np.nan, dtype=np.float64)

        if pose_valid and lm_xyz is not None:
            track, lm, origin = _normalize_pose_frame(lm_xyz, origin=origin)

        flow_feats = {
            "avg_speed": np.nan,
            "max_speed": np.nan,
            "dominant_angle_deg": np.nan,
            "direction_concentration": np.nan,
            "n_pixels": 0.0,
            "threshold": np.nan,
        }
        flow_valid = False

        if bbox is not None:
            bbox = clip_bbox(bbox, width=frame.shape[1], height=frame.shape[0])
            x0, y0, w, h = int(bbox.x), int(bbox.y), int(bbox.w), int(bbox.h)
            cv2.rectangle(disp, (x0, y0), (x0 + w, y0 + h), (0, 255, 255), 2)
            roi = frame[y0:y0 + h, x0:x0 + w]
            roi = resize_keep_aspect(roi, size_hw=roi_size)
            roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

            roi_mask = None
            if mask is not None:
                m = mask[y0:y0 + h, x0:x0 + w].astype(np.uint8) * 255
                roi_mask = resize_keep_aspect(m, size_hw=roi_size) > 0

            if prev_roi_gray is not None:
                flow = farneback(prev_roi_gray, roi_gray)
                if roi_mask is not None:
                    flow = flow.copy()
                    flow[~roi_mask] = 0.0
                stats, motion_mask = compute_motion_stats(
                    flow,
                    threshold_method="otsu",
                    fixed_threshold=2.0,
                    subtract_bg=True,
                )
                flow_feats = {
                    "avg_speed": float(stats.avg_speed),
                    "max_speed": float(stats.max_speed),
                    "dominant_angle_deg": float(stats.dominant_angle_deg),
                    "direction_concentration": float(stats.direction_concentration),
                    "n_pixels": float(stats.n_pixels),
                    "threshold": float(stats.threshold),
                }
                flow_valid = bool(
                    stats.avg_speed > 0
                    and int(np.count_nonzero(motion_mask)) > 0
                )

            prev_roi_gray = roi_gray
        else:
            prev_roi_gray = None

        buf.push(
            t_ms=float(now * 1000.0),
            track_xyz=track,
            lm_xyz=lm,
            pose_valid=bool(pose_valid),
            flow_feats=flow_feats,
            flow_valid=bool(flow_valid),
        )

        # Inference throttling
        if (now - last_infer) * 1000.0 >= float(args.infer_every_ms):
            last_infer = now
            x_seq, length = buf.build_sequence(dt_ms=dt_ms, feat_cfg=feat_cfg)

            if length > 0:
                # Standardize using training stats.
                std = np.where(norm.std > 1e-8, norm.std, 1.0).astype(
                    np.float32
                )
                x_seq = (
                    x_seq - norm.mean.astype(np.float32).reshape(1, -1)
                ) / std.reshape(1, -1)

                xt = torch.from_numpy(x_seq).unsqueeze(0)  # (1,T,F)
                lt = torch.tensor([int(length)], dtype=torch.long)
                with torch.no_grad():
                    logits = model(xt, lt)
                    probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.float32)

                if ema_probs is None or float(args.ema) <= 0:
                    ema_probs = probs
                else:
                    a = float(args.ema)
                    ema_probs = a * ema_probs + (1.0 - a) * probs

        # HUD
        if ema_probs is not None:
            top = int(np.argmax(ema_probs))
            top_label = str(idx_to_label.get(top, str(top)))
            top_p = float(ema_probs[top])

            d0x_idx = label_to_idx.get("D0X")
            if d0x_idx is not None and float(ema_probs[d0x_idx]) >= float(args.d0x_thr):
                top_label = "D0X"
                top_p = float(ema_probs[d0x_idx])

            hud = (
                f"{top_label}  p={top_p:.2f}  "
                f"fps={float(fps_ema or 0.0):.1f}"
            )
            cv2.putText(
                disp,
                hud,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2,
            )

            # top-3
            k = min(3, int(len(ema_probs)))
            topk = np.argsort(-ema_probs)[:k]
            for i, idx in enumerate(topk.tolist()):
                lab = str(idx_to_label.get(int(idx), str(idx)))
                p_i = float(ema_probs[int(idx)])
                cv2.putText(
                    disp,
                    f"{lab}: {p_i:.2f}",
                    (10, 60 + 25 * i),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 0),
                    2,
                )
        else:
            cv2.putText(
                disp,
                "Warming up...",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 255),
                2,
            )

        cv2.imshow(win, disp)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    return 0

if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

