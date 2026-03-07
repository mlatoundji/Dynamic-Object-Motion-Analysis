from __future__ import annotations

import argparse
import csv
import json
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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

from doma.models.cnn_lstm import CNNLSTM, ModelConfig
from doma.models.temporal_transformer import TemporalTransformer, ModelConfig as TTConfig
from doma.dataloaders import FeatureConfig, NormStats
from .datasets.manifest import write_manifest_csv
from .datasets.schema import OptFlowFeatures, PoseTensor


def _now_ms() -> float:
    return float(time.time() * 1000.0)


def _bbox_center_norm(bbox, *, width: int, height: int) -> tuple[float, float]:
    cx = (float(bbox.x) + 0.5 * float(bbox.w)) / max(1.0, float(width))
    cy = (float(bbox.y) + 0.5 * float(bbox.h)) / max(1.0, float(height))
    return cx, cy


def _bbox_iou(a, b) -> float:
    ax1, ay1 = float(a.x), float(a.y)
    ax2, ay2 = float(a.x + a.w), float(a.y + a.h)
    bx1, by1 = float(b.x), float(b.y)
    bx2, by2 = float(b.x + b.w), float(b.y + b.h)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    if denom <= 1e-9:
        return 0.0
    return float(inter / denom)


def _map_bbox_x_for_display(bbox, *, frame_w: int) -> int:
    # Horizontal flip mapping: x' = W - (x + w)
    return int(frame_w - (int(bbox.x) + int(bbox.w)))


class _SessionLogger:
    def __init__(
        self,
        *,
        out_dir: Path,
        run_dir: Path,
        args: argparse.Namespace,
        labels_sorted: list[str],
    ) -> None:
        self.out_dir = out_dir
        self.run_dir = run_dir
        self.args = args
        self.labels_sorted = list(labels_sorted)

        self.session_id = time.strftime("live_%Y%m%d-%H%M%S")
        self.out_dir = (self.out_dir / self.session_id).resolve()
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.csv_path = self.out_dir / f"report_{self.session_id}.csv"
        self.txt_path = self.out_dir / f"report_{self.session_id}.txt"
        self.npz_path = self.out_dir / f"dump_{self.session_id}.npz"

        self._csv_f = self.csv_path.open("w", encoding="utf-8", newline="")
        self._csv = csv.DictWriter(self._csv_f, fieldnames=self._csv_fields())
        self._csv.writeheader()
        self._csv_f.flush()

        # Inference dumps (per inference)
        self.infer_t_ms: list[float] = []
        self.infer_lengths: list[int] = []
        self.infer_x_seq: list[np.ndarray] = []
        self.infer_probs: list[np.ndarray] = []
        self.infer_ema_probs: list[np.ndarray] = []

        self.frames = 0
        self.infers = 0
        self.resets = 0
        self.infer_ms_sum = 0.0
        self.pred_counts: dict[str, int] = {}

        self._write_txt_header()

    def _csv_fields(self) -> list[str]:
        return [
            "t_wall_ms",
            "frame_idx",
            "fps_ema",
            "did_infer",
            "infer_ms",
            "reset_reason",
            "hand_side",
            "pose_valid",
            "flow_valid",
            "bbox_x",
            "bbox_y",
            "bbox_w",
            "bbox_h",
            "bbox_score",
            "bbox_cx_norm",
            "bbox_cy_norm",
            "avg_speed",
            "max_speed",
            "dominant_angle_deg",
            "direction_concentration",
            "n_pixels",
            "threshold",
            "pred_label",
            "pred_idx",
            "pred_p",
            "ema_alpha",
            "topk_labels",
            "topk_ps",
        ]

    def _write_txt_header(self) -> None:
        payload = {
            "session_id": self.session_id,
            "created_wall_ms": _now_ms(),
            "run_dir": str(self.run_dir.as_posix()),
            "session_dir": str(self.out_dir.as_posix()),
            "cmd": "doma-live-classifier",
            "args": vars(self.args),
            "labels_sorted": self.labels_sorted,
        }
        self.txt_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def log_frame(self, row: dict[str, Any]) -> None:
        self.frames += 1
        if str(row.get("reset_reason") or ""):
            self.resets += 1
        self._csv.writerow(row)
        if (self.frames % 30) == 0:
            self._csv_f.flush()

    def log_infer(
        self,
        *,
        t_wall_ms: float,
        length: int,
        x_seq: np.ndarray,
        probs: np.ndarray,
        ema_probs: np.ndarray,
        infer_ms: float,
        pred_label: str,
        reset_reason: str | None,
    ) -> None:
        self.infers += 1
        self.infer_ms_sum += float(infer_ms)
        self.pred_counts[pred_label] = int(self.pred_counts.get(pred_label, 0)) + 1

        if bool(getattr(self.args, "dump_npz", False)):
            self.infer_t_ms.append(float(t_wall_ms))
            self.infer_lengths.append(int(length))
            self.infer_x_seq.append(np.asarray(x_seq, dtype=np.float32))
            self.infer_probs.append(np.asarray(probs, dtype=np.float32))
            self.infer_ema_probs.append(np.asarray(ema_probs, dtype=np.float32))

    def finalize(self) -> None:
        try:
            self._csv_f.flush()
        finally:
            self._csv_f.close()

        if bool(getattr(self.args, "dump_npz", False)) and self.infers > 0:
            x_obj = np.asarray(self.infer_x_seq, dtype=object)
            probs = np.stack(self.infer_probs, axis=0).astype(np.float32)
            ema = np.stack(self.infer_ema_probs, axis=0).astype(np.float32)
            np.savez_compressed(
                self.npz_path,
                infer_t_wall_ms=np.asarray(self.infer_t_ms, dtype=np.float64),
                lengths=np.asarray(self.infer_lengths, dtype=np.int32),
                x_seq_list=x_obj,
                probs_list=probs,
                ema_probs_list=ema,
                labels_sorted=np.asarray(self.labels_sorted, dtype=object),
            )

        # Append summary to txt
        avg_infer_ms = (self.infer_ms_sum / float(self.infers)) if self.infers else 0.0
        summary = {
            "frames": int(self.frames),
            "infers": int(self.infers),
            "avg_infer_ms": float(avg_infer_ms),
            "resets": int(self.resets),
            "pred_counts": dict(sorted(self.pred_counts.items(), key=lambda kv: (-kv[1], kv[0]))),
            "csv_path": str(self.csv_path.as_posix()),
            "npz_path": (str(self.npz_path.as_posix()) if bool(getattr(self.args, "dump_npz", False)) else None),
        }
        with self.txt_path.open("a", encoding="utf-8") as f:
            f.write("\n\n")
            f.write(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


@dataclass(frozen=True)
class _Rect:
    x: int
    y: int
    w: int
    h: int

    def contains(self, px: int, py: int) -> bool:
        return (self.x <= px < (self.x + self.w)) and (self.y <= py < (self.y + self.h))


@dataclass
class _Button:
    label: str  # token (D0X, B0A, G01...)
    text: str  # display text
    rect: _Rect
    hotkey: str


class _AnnoRing:
    """
    Lightweight ring buffer for annotation capture.
    Stores per-frame signals so we can extract [t0,t1] slices.
    """

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

        self.hand_side: deque[str] = deque()

        # Per-frame "held" prediction (last inference)
        self.pred_label: deque[str] = deque()
        self.pred_p: deque[float] = deque()

    def push(
        self,
        *,
        t_ms: float,
        track_xyz: np.ndarray,
        lm_xyz: np.ndarray,
        pose_valid: bool,
        flow_feats: dict[str, float],
        flow_valid: bool,
        hand_side: str,
        pred_label: str,
        pred_p: float,
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

        self.hand_side.append(str(hand_side))
        self.pred_label.append(str(pred_label))
        self.pred_p.append(float(pred_p))

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
            self.hand_side.popleft()
            self.pred_label.popleft()
            self.pred_p.popleft()

    def slice(self, *, t0_ms: float, t1_ms: float) -> dict[str, Any]:
        t = np.asarray(self.t_ms, dtype=np.float64)
        if t.size == 0:
            return {"t_ms": np.zeros((0,), dtype=np.float64)}
        m = (t >= float(t0_ms)) & (t <= float(t1_ms))
        idx = np.where(m)[0]
        if idx.size == 0:
            return {"t_ms": np.zeros((0,), dtype=np.float64)}

        i0 = int(idx[0])
        i1 = int(idx[-1]) + 1

        def _stack_arr(q: deque[np.ndarray], *, shape: tuple[int, ...]) -> np.ndarray:
            arrs = list(q)[i0:i1]
            if not arrs:
                return np.zeros((0,) + shape, dtype=np.float64)
            return np.stack(arrs, axis=0)

        out: dict[str, Any] = {
            "t_ms": t[i0:i1].astype(np.float64),
            "track_xyz": _stack_arr(self.track_xyz, shape=(3,)),
            "lm_xyz": _stack_arr(self.lm_xyz, shape=(21, 3)),
            "pose_valid": np.asarray(list(self.pose_valid)[i0:i1], dtype=bool),
            "avg_speed": np.asarray(list(self.avg_speed)[i0:i1], dtype=np.float64),
            "max_speed": np.asarray(list(self.max_speed)[i0:i1], dtype=np.float64),
            "dominant_angle_deg": np.asarray(list(self.ang_deg)[i0:i1], dtype=np.float64),
            "direction_concentration": np.asarray(list(self.conc)[i0:i1], dtype=np.float64),
            "n_pixels": np.asarray(list(self.npx)[i0:i1], dtype=np.float64),
            "threshold": np.asarray(list(self.thr)[i0:i1], dtype=np.float64),
            "flow_valid": np.asarray(list(self.flow_valid)[i0:i1], dtype=bool),
            "hand_side": list(self.hand_side)[i0:i1],
            "pred_label": list(self.pred_label)[i0:i1],
            "pred_p": np.asarray(list(self.pred_p)[i0:i1], dtype=np.float64),
        }
        return out


class _AnnotationManager:
    def __init__(
        self,
        *,
        labels_sorted: list[str],
        label_desc: dict[str, str],
        session_dir: Path,
        dataset_root: Path,
        dt_ms: float,
        feat_cfg: FeatureConfig,
        mirror_view: bool,
        flip_features: bool,
        pre_ms: float,
        post_ms: float,
    ) -> None:
        self.labels_sorted = list(labels_sorted)
        self.label_desc = dict(label_desc)
        self.session_dir = session_dir
        self.dataset_root = dataset_root
        self.dt_ms = float(dt_ms)
        self.feat_cfg = feat_cfg
        self.mirror_view = bool(mirror_view)
        self.flip_features = bool(flip_features)
        self.pre_ms = float(pre_ms)
        self.post_ms = float(post_ms)

        self.session_start_ms: float | None = None

        # UI / state
        self.enabled = True
        self.armed_label = self.labels_sorted[0] if self.labels_sorted else ""
        self.capturing = False
        self.capture_label = ""
        self.capture_start_ms = 0.0
        self.capture_end_target_ms: float | None = None  # post-buffer target
        self.capture_id = 0
        self.counts: dict[str, int] = {}

        # Buffers: keep enough for pre/post and a full gesture window
        self.ring = _AnnoRing(max_ms=max(6000.0, float(pre_ms + post_ms + 3000.0)))

        # Exports
        self.ann_dir = (self.session_dir / "annotations").resolve()
        self.ann_dir.mkdir(parents=True, exist_ok=True)
        self.captures_jsonl = self.ann_dir / "captures.jsonl"
        self.segments_csv = self.ann_dir / "segments.csv"
        self.confusion_json = self.ann_dir / "confusion_counts.json"

        # Session dir name is already `live_YYYYmmdd-HHMMSS`
        self.dataset_dir = (self.dataset_root / self.session_dir.name).resolve()
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        self.dataset_train_dir = (self.dataset_dir / "train").resolve()
        self.dataset_train_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_csv = self.dataset_dir / "manifest.csv"
        self._manifest_rows: list[dict[str, Any]] = self._load_manifest_rows()

        self._hotkeys = self._build_hotkeys()
        self._buttons: list[_Button] = []
        self._layout_valid_for: tuple[int, int] | None = None

        self._conf_counts: dict[str, dict[str, int]] = self._load_confusion()

        # Per-inference tensors inside the session (for per-capture dumps)
        self._infer_t_ms: deque[float] = deque()
        self._infer_x_seq: deque[np.ndarray] = deque()
        self._infer_probs: deque[np.ndarray] = deque()
        self._infer_ema_probs: deque[np.ndarray] = deque()

    def _load_confusion(self) -> dict[str, dict[str, int]]:
        if not self.confusion_json.exists():
            return {}
        try:
            d = json.loads(self.confusion_json.read_text(encoding="utf-8"))
            if isinstance(d, dict):
                out: dict[str, dict[str, int]] = {}
                for gt, row in d.items():
                    if not isinstance(row, dict):
                        continue
                    out[str(gt)] = {str(k): int(v) for k, v in row.items()}
                return out
        except Exception:
            return {}
        return {}

    def _save_confusion(self) -> None:
        self.confusion_json.write_text(
            json.dumps(self._conf_counts, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def _load_manifest_rows(self) -> list[dict[str, Any]]:
        if not self.manifest_csv.exists():
            return []
        try:
            import csv as _csv

            with self.manifest_csv.open("r", encoding="utf-8", newline="") as f:
                r = _csv.DictReader(f)
                return [dict(row) for row in r]
        except Exception:
            return []

    def _save_manifest(self) -> None:
        write_manifest_csv(self.manifest_csv, self._manifest_rows)

    def _build_hotkeys(self) -> dict[int, str]:
        keys = "1234567890qwertyuiopasdfghjklzxcvbnm"
        out: dict[int, str] = {}
        for i, lab in enumerate(self.labels_sorted):
            if i >= len(keys):
                break
            out[ord(keys[i])] = lab
        return out

    def _label_text(self, label: str) -> str:
        desc = self.label_desc.get(label, "")
        if desc:
            return f"{label} — {desc}"
        return label

    def _ensure_layout(self, *, frame_w: int, frame_h: int) -> None:
        if self._layout_valid_for == (int(frame_w), int(frame_h)) and self._buttons:
            return

        margin = 10
        btn_h = 26
        btn_w = 360
        gap = 6
        cols = 2 if len(self.labels_sorted) > 7 else 1
        col_gap = 10
        rows = int(np.ceil(len(self.labels_sorted) / float(cols))) if cols else 0

        total_w = cols * btn_w + (cols - 1) * col_gap
        x0 = max(margin, int(frame_w - margin - total_w))
        y0 = 60

        buttons: list[_Button] = []
        for idx, lab in enumerate(self.labels_sorted):
            col = idx // rows if rows else 0
            row = idx % rows if rows else idx
            bx = int(x0 + col * (btn_w + col_gap))
            by = int(y0 + row * (btn_h + gap))
            hk = ""
            for k, v in self._hotkeys.items():
                if v == lab:
                    hk = chr(int(k))
                    break
            text = self._label_text(lab)
            if hk:
                text = f"[{hk}] {text}"
            buttons.append(_Button(label=lab, text=text, rect=_Rect(bx, by, btn_w, btn_h), hotkey=hk))

        self._buttons = buttons
        self._layout_valid_for = (int(frame_w), int(frame_h))

    def on_mouse(self, event: int, x: int, y: int, flags: int, param: Any) -> None:  # noqa: ANN401
        if not self.enabled:
            return
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        for b in self._buttons:
            if b.rect.contains(int(x), int(y)):
                self.toggle_capture(b.label, now_ms=_now_ms())
                return

    def on_key(self, key: int, *, now_ms: float) -> None:
        if not self.enabled:
            return
        if key in self._hotkeys:
            self.armed_label = self._hotkeys[int(key)]
            return
        if key == ord(" "):
            if self.armed_label:
                self.toggle_capture(self.armed_label, now_ms=float(now_ms))
            return

    def toggle_capture(self, label: str, *, now_ms: float) -> None:
        if not label:
            return
        if not self.capturing:
            self.capturing = True
            self.capture_label = str(label)
            self.capture_start_ms = float(now_ms) - float(self.pre_ms)
            self.capture_end_target_ms = None
            return

        # Already capturing: only allow stopping the same label
        if str(label) != str(self.capture_label):
            return
        if self.capture_end_target_ms is None:
            self.capture_end_target_ms = float(now_ms) + float(self.post_ms)

    def on_frame(
        self,
        *,
        t_wall_ms: float,
        track_xyz: np.ndarray,
        lm_xyz: np.ndarray,
        pose_valid: bool,
        flow_feats: dict[str, float],
        flow_valid: bool,
        hand_side: str,
        pred_label: str,
        pred_p: float,
        did_infer: bool,
        x_seq: np.ndarray | None,
        probs: np.ndarray | None,
        ema_probs: np.ndarray | None,
    ) -> None:
        if self.session_start_ms is None:
            self.session_start_ms = float(t_wall_ms)

        self.ring.push(
            t_ms=float(t_wall_ms),
            track_xyz=track_xyz,
            lm_xyz=lm_xyz,
            pose_valid=bool(pose_valid),
            flow_feats=flow_feats,
            flow_valid=bool(flow_valid),
            hand_side=str(hand_side),
            pred_label=str(pred_label),
            pred_p=float(pred_p),
        )

        if bool(did_infer) and x_seq is not None and probs is not None and ema_probs is not None:
            self._infer_t_ms.append(float(t_wall_ms))
            self._infer_x_seq.append(np.asarray(x_seq, dtype=np.float32))
            self._infer_probs.append(np.asarray(probs, dtype=np.float32))
            self._infer_ema_probs.append(np.asarray(ema_probs, dtype=np.float32))
            while len(self._infer_t_ms) >= 2 and (self._infer_t_ms[-1] - self._infer_t_ms[0]) > float(self.ring.max_ms):
                self._infer_t_ms.popleft()
                self._infer_x_seq.popleft()
                self._infer_probs.popleft()
                self._infer_ema_probs.popleft()

        if not self.capturing:
            return
        if self.capture_end_target_ms is None:
            return
        if float(t_wall_ms) < float(self.capture_end_target_ms):
            return

        # Finalize capture once post-buffer elapsed.
        t0 = float(self.capture_start_ms)
        t1 = float(self.capture_end_target_ms)
        self._finalize_capture(t0_ms=t0, t1_ms=t1, pred_label=str(pred_label), pred_p=float(pred_p))

        self.capturing = False
        self.capture_label = ""
        self.capture_end_target_ms = None

    def draw(self, disp_bgr: np.ndarray) -> None:
        if not self.enabled:
            return
        h, w = disp_bgr.shape[:2]
        self._ensure_layout(frame_w=int(w), frame_h=int(h))

        for b in self._buttons:
            is_capture = self.capturing and (b.label == self.capture_label)
            is_armed = (b.label == self.armed_label) and not is_capture

            if is_capture:
                color = (0, 0, 255)
            elif is_armed:
                color = (255, 0, 0)
            else:
                color = (50, 50, 50)

            cv2.rectangle(disp_bgr, (b.rect.x, b.rect.y), (b.rect.x + b.rect.w, b.rect.y + b.rect.h), color, 2)
            cv2.putText(
                disp_bgr,
                b.text[:60],
                (b.rect.x + 6, b.rect.y + b.rect.h - 7),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (240, 240, 240),
                1,
            )

        # HUD line
        if self.capturing:
            label = self._label_text(self.capture_label)
            cv2.putText(
                disp_bgr,
                f"CAPTURING: {label}",
                (10, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )
        else:
            armed = self._label_text(self.armed_label) if self.armed_label else ""
            cv2.putText(
                disp_bgr,
                f"ANNOTATIONS: armed={armed} (SPACE start/stop)",
                (10, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (200, 200, 200),
                1,
            )

    def _majority(self, labs: list[str]) -> tuple[str, dict[str, int]]:
        counts: dict[str, int] = {}
        for lab in labs:
            s = str(lab).strip()
            if not s:
                continue
            counts[s] = int(counts.get(s, 0)) + 1
        if not counts:
            return "", {}
        best = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
        return best, counts

    def _finalize_capture(self, *, t0_ms: float, t1_ms: float, pred_label: str, pred_p: float) -> None:
        self.capture_id += 1
        cap_id = f"{self.capture_id:04d}"
        gt = str(self.capture_label)
        self.counts[gt] = int(self.counts.get(gt, 0)) + 1

        seg = self.ring.slice(t0_ms=float(t0_ms), t1_ms=float(t1_ms))
        t = np.asarray(seg.get("t_ms", np.zeros((0,), np.float64)), dtype=np.float64)
        if t.size == 0:
            return

        # Session-relative segments.csv (ms)
        sess0 = float(self.session_start_ms or float(t0_ms))
        seg_start_rel = float(t0_ms - sess0)
        seg_end_rel = float(t1_ms - sess0)
        if not self.segments_csv.exists():
            self.segments_csv.write_text("t_start_ms,t_end_ms,label\n", encoding="utf-8")
        with self.segments_csv.open("a", encoding="utf-8", newline="") as f:
            f.write(f"{seg_start_rel:.3f},{seg_end_rel:.3f},{gt}\n")

        # Prediction majority inside segment (per-frame held prediction)
        infer_t = np.asarray(self._infer_t_ms, dtype=np.float64)
        infer_idx = np.where((infer_t >= float(t0_ms)) & (infer_t <= float(t1_ms)))[0]
        infer_probs = None
        infer_ema_probs = None
        infer_x_obj = None
        infer_t_sel = None
        pred_majority = ""
        pred_counts: dict[str, int] = {}

        if infer_idx.size > 0:
            # Majority on inference instants (more faithful than held per-frame label)
            labs = []
            for i in infer_idx.tolist():
                p = self._infer_ema_probs[int(i)]
                lab = self.labels_sorted[int(np.argmax(p))] if p.size else ""
                labs.append(str(lab))
            pred_majority, pred_counts = self._majority(labs)

            infer_t_sel = infer_t[infer_idx].astype(np.float64)
            infer_x_obj = np.asarray([self._infer_x_seq[int(i)] for i in infer_idx.tolist()], dtype=object)
            infer_probs = np.stack([self._infer_probs[int(i)] for i in infer_idx.tolist()], axis=0).astype(np.float32)
            infer_ema_probs = np.stack([self._infer_ema_probs[int(i)] for i in infer_idx.tolist()], axis=0).astype(np.float32)
        else:
            # Fallback: majority on held per-frame labels
            pred_labs = [str(x) for x in (seg.get("pred_label") or [])]
            pred_majority, pred_counts = self._majority(pred_labs)

        # Update confusion counts
        self._conf_counts.setdefault(gt, {})
        if pred_majority:
            self._conf_counts[gt][pred_majority] = int(self._conf_counts[gt].get(pred_majority, 0)) + 1
        self._save_confusion()

        # Export capture NPZ (session)
        cap_npz = self.ann_dir / f"capture_{cap_id}.npz"
        np.savez_compressed(
            cap_npz,
            t_wall_ms=t.astype(np.float64),
            infer_t_wall_ms=(infer_t_sel if infer_t_sel is not None else np.zeros((0,), dtype=np.float64)),
            x_seq_list=(infer_x_obj if infer_x_obj is not None else np.asarray([], dtype=object)),
            probs_list=(infer_probs if infer_probs is not None else np.zeros((0, 0), dtype=np.float32)),
            ema_probs_list=(infer_ema_probs if infer_ema_probs is not None else np.zeros((0, 0), dtype=np.float32)),
            gt_label=np.array([gt], dtype=object),
            pred_majority=np.array([pred_majority], dtype=object),
            pred_counts=np.array([json.dumps(pred_counts, ensure_ascii=False, sort_keys=True)], dtype=object),
            track_xyz=np.asarray(seg.get("track_xyz", np.zeros((0, 3), np.float64)), dtype=np.float32),
            lm_xyz=np.asarray(seg.get("lm_xyz", np.zeros((0, 21, 3), np.float64)), dtype=np.float32),
            pose_valid=np.asarray(seg.get("pose_valid", np.zeros((0,), bool)), dtype=bool),
            avg_speed=np.asarray(seg.get("avg_speed", np.zeros((0,), np.float64)), dtype=np.float32),
            max_speed=np.asarray(seg.get("max_speed", np.zeros((0,), np.float64)), dtype=np.float32),
            dominant_angle_deg=np.asarray(seg.get("dominant_angle_deg", np.zeros((0,), np.float64)), dtype=np.float32),
            direction_concentration=np.asarray(seg.get("direction_concentration", np.zeros((0,), np.float64)), dtype=np.float32),
            n_pixels=np.asarray(seg.get("n_pixels", np.zeros((0,), np.float64)), dtype=np.float32),
            threshold=np.asarray(seg.get("threshold", np.zeros((0,), np.float64)), dtype=np.float32),
            flow_valid=np.asarray(seg.get("flow_valid", np.zeros((0,), bool)), dtype=bool),
            hand_side=np.asarray(seg.get("hand_side", []), dtype=object),
            meta_json=np.array(
                [
                    json.dumps(
                        {
                            "mirror_view": self.mirror_view,
                            "flip_features": self.flip_features,
                            "dt_ms": self.dt_ms,
                            "pre_ms": self.pre_ms,
                            "post_ms": self.post_ms,
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                ],
                dtype=object,
            ),
        )

        # Append captures.jsonl
        rec = {
            "capture_id": cap_id,
            "gt_label": gt,
            "t_start_ms_rel_session": seg_start_rel,
            "t_end_ms_rel_session": seg_end_rel,
            "pred_majority": pred_majority,
            "pred_counts": pred_counts,
            "pose_valid_ratio": float(np.mean(np.asarray(seg.get("pose_valid"), dtype=bool))) if t.size else 0.0,
            "flow_valid_ratio": float(np.mean(np.asarray(seg.get("flow_valid"), dtype=bool))) if t.size else 0.0,
        }
        with self.captures_jsonl.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False, sort_keys=True))
            f.write("\n")

        # Export training-ready sample
        self._export_training_sample(
            cap_id=cap_id,
            gt_label=gt,
            seg=seg,
            t0_ms=float(t0_ms),
            t1_ms=float(t1_ms),
        )

    def _export_training_sample(
        self,
        *,
        cap_id: str,
        gt_label: str,
        seg: dict[str, Any],
        t0_ms: float,
        t1_ms: float,
    ) -> None:
        sample_id = f"{self.session_dir.name}_{cap_id}"
        sample_dir = (self.dataset_train_dir / sample_id).resolve()
        sample_dir.mkdir(parents=True, exist_ok=True)

        t_wall = np.asarray(seg.get("t_ms", np.zeros((0,), np.float64)), dtype=np.float64)
        t_rel = t_wall - float(t_wall[0]) if t_wall.size else t_wall

        track = np.asarray(seg.get("track_xyz", np.zeros((0, 3), np.float64)), dtype=np.float64)
        lms = np.asarray(seg.get("lm_xyz", np.zeros((0, 21, 3), np.float64)), dtype=np.float64)
        pose_valid_raw = np.asarray(seg.get("pose_valid", np.zeros((0,), bool)), dtype=bool)

        # Pose resample + derivatives
        r_track = resample_linear(t_rel, track, dt_ms=float(self.dt_ms), axis_time=0)
        r_lm = resample_linear(t_rel, lms, dt_ms=float(self.dt_ms), axis_time=0)
        vel = finite_diff(r_track.t_ms, r_track.y, axis_time=0)
        acc = finite_diff(r_track.t_ms, vel, axis_time=0)
        pose_valid = r_track.valid & r_lm.valid

        # Optflow resample (match datasets/optflow.py approach)
        avg = resample_linear(t_rel, np.asarray(seg.get("avg_speed", []), dtype=np.float64), dt_ms=float(self.dt_ms), axis_time=0)
        mx = resample_linear(t_rel, np.asarray(seg.get("max_speed", []), dtype=np.float64), dt_ms=float(self.dt_ms), axis_time=0)
        conc = resample_linear(t_rel, np.asarray(seg.get("direction_concentration", []), dtype=np.float64), dt_ms=float(self.dt_ms), axis_time=0)
        thr = resample_linear(t_rel, np.asarray(seg.get("threshold", []), dtype=np.float64), dt_ms=float(self.dt_ms), axis_time=0)

        ang_deg = np.asarray(seg.get("dominant_angle_deg", []), dtype=np.float64)
        ang_rad = np.deg2rad(ang_deg)
        r_sin = resample_linear(t_rel, np.sin(ang_rad), dt_ms=float(self.dt_ms), axis_time=0)
        r_cos = resample_linear(t_rel, np.cos(ang_rad), dt_ms=float(self.dt_ms), axis_time=0)
        ang_out = (np.rad2deg(np.arctan2(r_sin.y, r_cos.y)) + 360.0) % 360.0

        npx = resample_linear(t_rel, np.asarray(seg.get("n_pixels", []), dtype=np.float64), dt_ms=float(self.dt_ms), axis_time=0)
        flow_valid_interp = avg.valid & mx.valid & conc.valid & thr.valid & r_sin.valid & r_cos.valid & npx.valid

        raw_flow_valid = np.asarray(seg.get("flow_valid", np.zeros((0,), bool)), dtype=np.float64)
        r_flow_valid = resample_linear(t_rel, raw_flow_valid, dt_ms=float(self.dt_ms), axis_time=0)
        flow_valid = flow_valid_interp & r_flow_valid.valid & (r_flow_valid.y >= 0.5)

        # Align lengths defensively
        T = int(min(r_track.y.shape[0], avg.y.shape[0]))
        t_reg = r_track.t_ms[:T]
        pose_valid = pose_valid[:T]
        flow_valid = flow_valid[:T]

        pose_path = sample_dir / "pose_tensor.npz"
        flow_path = sample_dir / "optflow_features.npz"
        quality_path = sample_dir / "quality.json"

        PoseTensor(
            timestamps_ms=t_reg.astype(np.float32),
            track_pos_xyz=r_track.y[:T].astype(np.float32),
            track_vel_xyz=vel[:T].astype(np.float32),
            track_acc_xyz=acc[:T].astype(np.float32),
            landmarks_xyz=r_lm.y[:T].astype(np.float32) if bool(self.feat_cfg.use_landmarks) else None,
            valid=pose_valid.astype(bool),
            meta={
                "source": "live_annotations",
                "session": self.session_dir.name,
                "capture_id": cap_id,
                "gt_label": gt_label,
                "mirror_view": self.mirror_view,
                "flip_features": self.flip_features,
                "resample_dt_ms": float(self.dt_ms),
            },
        ).to_npz(pose_path)

        OptFlowFeatures(
            timestamps_ms=t_reg.astype(np.float32),
            avg_speed=avg.y[:T].astype(np.float32),
            max_speed=mx.y[:T].astype(np.float32),
            dominant_angle_deg=ang_out[:T].astype(np.float32),
            direction_concentration=conc.y[:T].astype(np.float32),
            n_pixels=np.round(npx.y[:T]).astype(np.int32),
            threshold=thr.y[:T].astype(np.float32),
            valid=flow_valid.astype(bool),
            meta={
                "source": "live_annotations",
                "session": self.session_dir.name,
                "capture_id": cap_id,
                "gt_label": gt_label,
                "resample_dt_ms": float(self.dt_ms),
            },
        ).to_npz(flow_path)

        q = {
            "sample_id": sample_id,
            "dataset": "annotated",
            "split": "train",
            "label": gt_label,
            "pose_valid_ratio": float(np.mean(pose_valid)) if pose_valid.size else 0.0,
            "optflow_valid_ratio": float(np.mean(flow_valid)) if flow_valid.size else 0.0,
            "t_start_wall_ms": float(t0_ms),
            "t_end_wall_ms": float(t1_ms),
        }
        quality_path.write_text(json.dumps(q, ensure_ascii=False, indent=2), encoding="utf-8")

        repo_root = _repo_root()
        row = {
            "sample_id": sample_id,
            "dataset": "annotated",
            "split": "train",
            "label": gt_label,
            "source_uri": f"live://{self.session_dir.name}#{cap_id}",
            "pose_npz": str(pose_path.relative_to(repo_root).as_posix()),
            "optflow_npz": str(flow_path.relative_to(repo_root).as_posix()),
            "quality_json": str(quality_path.relative_to(repo_root).as_posix()),
        }
        self._manifest_rows.append(row)
        self._save_manifest()


def _load_bundle(run_dir: Path) -> tuple[Path, NormStats, dict[str, int], float]:
    """Load checkpoint path, norm stats, label map, and dt_ms from a training run directory.
    Supports both layouts: run_dir/best.pt (doma-train) and run_dir/checkpoints/best.pt (legacy).
    """
    run_dir = run_dir.resolve()
    ckpt = run_dir / "best.pt"
    if not ckpt.exists():
        ckpt = run_dir / "best.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"Missing checkpoint: {run_dir / 'best.pt'} or {run_dir / 'best.pt'}")

    norm_path = run_dir / "norm.npz"
    if not norm_path.exists():
        raise FileNotFoundError(f"Missing norm stats: {norm_path} (train with doma-train --model cnn_lstm to produce it)")
    norm = NormStats.from_npz(norm_path)

    label_map_path = run_dir / "label_map.json"
    if not label_map_path.exists():
        raise FileNotFoundError(f"Missing label map: {label_map_path}")
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


def _repo_root() -> Path:
    # .../doma/live_classifier.py -> repo root
    return Path(__file__).resolve().parents[1]


def _load_label_descriptions() -> dict[str, str]:
    """
    Loads IPN label -> semantic description from `docs/labels.md` when available.
    Falls back to a built-in mapping if the file is missing/unparseable.
    """
    fallback = {
        "D0X": "Non-gesture",
        "B0A": "Pointing with one finger",
        "B0B": "Pointing with two fingers",
        "G01": "Click with one finger",
        "G02": "Click with two fingers",
        "G03": "Throw up",
        "G04": "Throw down",
        "G05": "Throw left",
        "G06": "Throw right",
        "G07": "Open twice",
        "G08": "Double click with one finger",
        "G09": "Double click with two fingers",
        "G10": "Zoom in",
        "G11": "Zoom out",
    }

    path = _repo_root() / "docs" / "labels.md"
    if not path.exists():
        return fallback

    out: dict[str, str] = {}
    lines = path.read_text(encoding="utf-8").splitlines()
    for line in lines:
        s = line.strip()
        if not s:
            continue
        if s.lower().startswith("id\tlabel\tgesture"):
            continue
        if s.startswith("id ") or s.lower().startswith("id\t"):
            continue
        parts = [p.strip() for p in s.split("\t")]
        if len(parts) < 3:
            continue
        _id, label, gesture = parts[0], parts[1], parts[2]
        if not label or not gesture:
            continue
        if label.lower().startswith("all "):
            continue
        # Basic validation: IPN labels are short tokens (D0X, B0A, G01...)
        if len(label) > 8:
            continue
        out[label] = gesture

    return out or fallback


def _label_with_desc(label: str, desc: dict[str, str]) -> str:
    d = desc.get(label)
    if not d:
        return label
    return f"{label} — {d}"


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
    p.add_argument("--run", required=True, help="Run directory (models/<run_id> from doma-train); cnn_lstm or temporal_transformer; must contain best.pt, norm.npz, label_map.json")
    p.add_argument("--source", default="0", help="Camera index or video path")
    p.add_argument("--window-ms", type=float, default=1500.0)
    p.add_argument("--infer-every-ms", type=float, default=200.0)
    p.add_argument("--ema", type=float, default=0.6, help="EMA smoothing for probabilities (0=no smoothing)")
    p.add_argument("--d0x-thr", type=float, default=0.6, help="If P(D0X) >= thr, display D0X as idle")
    p.add_argument("--roi", type=int, default=224, help="ROI size (square) for optflow features")
    p.add_argument("--mirror-view", action=argparse.BooleanOptionalAction, default=True, help="Mirror the display (HUD) horizontally")
    p.add_argument("--flip-features", action=argparse.BooleanOptionalAction, default=True, help="Flip frames for feature extraction (match typical webcam mirror view)")
    p.add_argument("--no-landmarks", action="store_true")
    p.add_argument("--no-optflow", action="store_true")
    p.add_argument("--no-pose", action="store_true")
    p.add_argument("--topk", type=int, default=5, help="Top-k predictions to show/log")

    # Logging
    p.add_argument("--log", action=argparse.BooleanOptionalAction, default=True, help="Enable session logging (CSV/TXT + optional NPZ)")
    p.add_argument("--log-dir", default="doma/sessions", help="Root directory to store sessions under (a per-session subdir is created)")
    p.add_argument("--dump-npz", action=argparse.BooleanOptionalAction, default=True, help="When logging, also dump x_seq + probs per inference (NPZ)")

    # Auto-reset (prevents cross-hand contamination of state)
    p.add_argument("--reset-lost-ms", type=float, default=600.0, help="Reset state after bbox is missing for this duration")
    p.add_argument("--reset-iou-thr", type=float, default=0.10, help="Reset if bbox IoU drops below this threshold")
    p.add_argument("--reset-center-jump", type=float, default=0.25, help="Reset if bbox center jumps more than this (normalized)")
    p.add_argument("--reset-side-frames", type=int, default=3, help="Reset if hand side (L/R) changes for N consecutive frames")

    # Live annotations
    p.add_argument("--annotations", action=argparse.BooleanOptionalAction, default=False, help="Enable live gesture annotation mode (buttons + hotkeys)")
    p.add_argument("--annot-pre-ms", type=float, default=300.0, help="Pre-buffer duration (ms) included before capture start")
    p.add_argument("--annot-post-ms", type=float, default=300.0, help="Post-buffer duration (ms) included after capture end")
    p.add_argument("--annot-dataset-root", default="data/annotated", help="Root folder for training-ready exports (gitignored)")
    args = p.parse_args(argv)

    if torch is None:
        raise SystemExit("PyTorch is required. Install with: poetry install -E train -E hand")

    ckpt_path, norm, label_to_idx, dt_ms = _load_bundle(Path(args.run))
    run_dir = Path(args.run).resolve()
    idx_to_label = {v: k for k, v in label_to_idx.items()}
    label_desc = _load_label_descriptions()
    labels_sorted = [lab for lab, _ in sorted(label_to_idx.items(), key=lambda kv: kv[1])]

    ckpt = torch.load(ckpt_path, map_location="cpu")
    # Support both formats: live-classifier (model + model_config) and train.py (model_state_dict + model_config.json)
    raw_cfg = ckpt.get("model_config")
    if raw_cfg is None:
        model_config_path = run_dir / "model_config.json"
        if not model_config_path.exists():
            raise FileNotFoundError(f"Checkpoint has no model_config and {model_config_path} not found")
        raw_cfg = json.loads(model_config_path.read_text(encoding="utf-8"))

    # Detect model type: Temporal Transformer has d_model / nhead; CNN-LSTM does not
    if "d_model" in raw_cfg:
        mcfg = TTConfig(**raw_cfg)
        model = TemporalTransformer(mcfg)
    else:
        mcfg = ModelConfig(**raw_cfg)
        model = CNNLSTM(mcfg)

    state = ckpt.get("model", ckpt.get("model_state_dict"))
    if state is None:
        raise KeyError("Checkpoint must contain 'model' or 'model_state_dict'")
    model.load_state_dict(state, strict=True)
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
    last_probs = None
    last_x_seq = None
    last_topk = None
    last_pred_label = ""
    last_pred_idx = -1
    last_pred_p = float("nan")
    last_infer_ms = float("nan")

    t0 = time.time()
    frame_idx = 0
    fps_ema = None

    # Reset heuristics state
    last_bbox = None
    lost_since_ms: float | None = None
    last_side: str | None = None
    side_streak = 0

    # Optional logging
    logger: _SessionLogger | None = None
    if bool(getattr(args, "log", False)):
        base_dir = Path(str(args.log_dir)).expanduser()
        if not base_dir.is_absolute():
            base_dir = (_repo_root() / base_dir).resolve()
        logger = _SessionLogger(
            out_dir=base_dir,
            run_dir=Path(args.run).resolve(),
            args=args,
            labels_sorted=labels_sorted,
        )

    ann: _AnnotationManager | None = None
    if bool(getattr(args, "annotations", False)):
        if logger is None:
            raise SystemExit("Annotations require logging (do not use --no-log with --annotations).")
        ds_root = Path(str(args.annot_dataset_root)).expanduser()
        if not ds_root.is_absolute():
            ds_root = (_repo_root() / ds_root).resolve()
        ann = _AnnotationManager(
            labels_sorted=labels_sorted,
            label_desc=label_desc,
            session_dir=logger.out_dir,
            dataset_root=ds_root,
            dt_ms=float(dt_ms),
            feat_cfg=feat_cfg,
            mirror_view=bool(args.mirror_view),
            flip_features=bool(args.flip_features),
            pre_ms=float(args.annot_pre_ms),
            post_ms=float(args.annot_post_ms),
        )
        cv2.setMouseCallback(win, ann.on_mouse)

    def _reset_state(reason: str) -> None:
        nonlocal prev_roi_gray, origin, buf, last_infer, ema_probs
        nonlocal last_bbox, lost_since_ms, last_side, side_streak
        prev_roi_gray = None
        origin = None
        buf = LiveBuffer(max_ms=float(args.window_ms))
        last_infer = 0.0
        ema_probs = None
        last_bbox = None
        lost_since_ms = None
        last_side = None
        side_streak = 0

    def _hand_side_from_bbox(bbox, *, frame_w: int) -> str:
        cx = (float(bbox.x) + 0.5 * float(bbox.w)) / max(1.0, float(frame_w))
        return "L" if float(cx) < 0.5 else "R"

    while True:
        ok, frame_raw = cap.read()
        if not ok:
            break
        frame_w = int(frame_raw.shape[1])
        frame_h = int(frame_raw.shape[0])

        frame_feat = cv2.flip(frame_raw, 1) if bool(args.flip_features) else frame_raw
        disp = cv2.flip(frame_raw, 1).copy() if bool(args.mirror_view) else frame_raw.copy()

        now = time.time()
        fps = 0.0
        if frame_idx > 0:
            fps = 1.0 / max(1e-6, (now - t0))
        t0 = now
        fps_ema = fps if fps_ema is None else (0.9 * fps_ema + 0.1 * fps)
        frame_idx += 1

        bbox, mask, lm_xyz = det.detect_with_landmarks(frame_feat)
        pose_valid = bbox is not None and lm_xyz is not None

        reset_reason = ""

        if bbox is None:
            if lost_since_ms is None:
                lost_since_ms = _now_ms()
            elif (_now_ms() - float(lost_since_ms)) >= float(args.reset_lost_ms):
                reset_reason = "lost"
                _reset_state(reset_reason)
        else:
            # We have a bbox: clear lost timer
            lost_since_ms = None

            # Jump / side-change detection
            side = _hand_side_from_bbox(bbox, frame_w=frame_w)
            if last_side is None:
                last_side = side
                side_streak = 0
            elif side != last_side:
                side_streak += 1
                if side_streak >= int(args.reset_side_frames):
                    reset_reason = "side_change"
                    _reset_state(reset_reason)
                    side = _hand_side_from_bbox(bbox, frame_w=frame_w)
                    last_side = side
            else:
                side_streak = 0

            if last_bbox is not None:
                iou = _bbox_iou(last_bbox, bbox)
                c0x, c0y = _bbox_center_norm(last_bbox, width=frame_w, height=frame_h)
                c1x, c1y = _bbox_center_norm(bbox, width=frame_w, height=frame_h)
                jump = float(np.hypot(c1x - c0x, c1y - c0y))
                if (iou < float(args.reset_iou_thr)) or (jump > float(args.reset_center_jump)):
                    reset_reason = "jump"
                    _reset_state(reset_reason)
            last_bbox = bbox

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
            bbox = clip_bbox(bbox, width=frame_feat.shape[1], height=frame_feat.shape[0])
            x0, y0, w, h = int(bbox.x), int(bbox.y), int(bbox.w), int(bbox.h)
            # Draw bbox on display (may have different flip than features)
            x_disp = x0
            if bool(args.mirror_view) ^ bool(args.flip_features):
                x_disp = _map_bbox_x_for_display(bbox, frame_w=frame_w)
            cv2.rectangle(disp, (x_disp, y0), (x_disp + w, y0 + h), (0, 255, 255), 2)

            roi = frame_feat[y0:y0 + h, x0:x0 + w]
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
        did_infer = False
        if (now - last_infer) * 1000.0 >= float(args.infer_every_ms):
            last_infer = now
            x_seq, length = buf.build_sequence(dt_ms=dt_ms, feat_cfg=feat_cfg)

            if length > 0:
                did_infer = True
                # Standardize using training stats.
                std = np.where(norm.std > 1e-8, norm.std, 1.0).astype(
                    np.float32
                )
                x_seq = (
                    x_seq - norm.mean.astype(np.float32).reshape(1, -1)
                ) / std.reshape(1, -1)

                xt = torch.from_numpy(x_seq).unsqueeze(0)  # (1,T,F)
                lt = torch.tensor([int(length)], dtype=torch.long)
                t_inf0 = time.perf_counter()
                with torch.no_grad():
                    logits = model(batch={"x": xt, "lengths": lt})
                    probs = (
                        torch.softmax(logits, dim=1)
                        .squeeze(0)
                        .cpu()
                        .numpy()
                        .astype(np.float32)
                    )
                last_infer_ms = float((time.perf_counter() - t_inf0) * 1000.0)

                if ema_probs is None or float(args.ema) <= 0:
                    ema_probs = probs
                else:
                    a = float(args.ema)
                    ema_probs = a * ema_probs + (1.0 - a) * probs
                last_probs = probs
                last_x_seq = x_seq

        # HUD
        if ema_probs is not None:
            top = int(np.argmax(ema_probs))
            top_label = str(idx_to_label.get(top, str(top)))
            top_p = float(ema_probs[top])

            d0x_idx = label_to_idx.get("D0X")
            if d0x_idx is not None and float(ema_probs[d0x_idx]) >= float(args.d0x_thr):
                top_label = "D0X"
                top_p = float(ema_probs[d0x_idx])

            hud_label = _label_with_desc(top_label, label_desc)
            hud = f"{hud_label}  p={top_p:.2f}  fps={float(fps_ema or 0.0):.1f}"
            cv2.putText(
                disp,
                hud,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2,
            )

            # top-k
            k = min(int(max(1, args.topk)), int(len(ema_probs)))
            topk = np.argsort(-ema_probs)[:k]
            last_topk = topk.tolist()
            last_pred_label = str(top_label)
            last_pred_idx = int(top)
            last_pred_p = float(top_p)
            for i, idx in enumerate(topk.tolist()):
                lab = str(idx_to_label.get(int(idx), str(idx)))
                lab = _label_with_desc(lab, label_desc)
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

        # Logging (per frame)
        if logger is not None:
            if bbox is not None:
                cxn, cyn = _bbox_center_norm(bbox, width=frame_w, height=frame_h)
                side = _hand_side_from_bbox(bbox, frame_w=frame_w)
                bbox_score = float(getattr(bbox, "score", 1.0))
                bx, by, bw, bh = int(bbox.x), int(bbox.y), int(bbox.w), int(bbox.h)
            else:
                cxn, cyn = float("nan"), float("nan")
                side = ""
                bbox_score = float("nan")
                bx = by = bw = bh = -1

            topk_labels = ""
            topk_ps = ""
            if ema_probs is not None and last_topk is not None:
                labs = [str(idx_to_label.get(int(i), str(i))) for i in last_topk]
                ps = [float(ema_probs[int(i)]) for i in last_topk]
                topk_labels = "|".join(labs)
                topk_ps = "|".join([f"{p:.6f}" for p in ps])

            row = {
                "t_wall_ms": float(now * 1000.0),
                "frame_idx": int(frame_idx),
                "fps_ema": float(fps_ema or 0.0),
                "did_infer": int(1 if did_infer else 0),
                "infer_ms": float(last_infer_ms if did_infer else float("nan")),
                "reset_reason": str(reset_reason),
                "hand_side": str(side),
                "pose_valid": int(1 if bool(pose_valid) else 0),
                "flow_valid": int(1 if bool(flow_valid) else 0),
                "bbox_x": int(bx),
                "bbox_y": int(by),
                "bbox_w": int(bw),
                "bbox_h": int(bh),
                "bbox_score": float(bbox_score),
                "bbox_cx_norm": float(cxn),
                "bbox_cy_norm": float(cyn),
                "avg_speed": float(flow_feats.get("avg_speed", np.nan)),
                "max_speed": float(flow_feats.get("max_speed", np.nan)),
                "dominant_angle_deg": float(flow_feats.get("dominant_angle_deg", np.nan)),
                "direction_concentration": float(flow_feats.get("direction_concentration", np.nan)),
                "n_pixels": float(flow_feats.get("n_pixels", 0.0)),
                "threshold": float(flow_feats.get("threshold", np.nan)),
                "pred_label": str(last_pred_label),
                "pred_idx": int(last_pred_idx),
                "pred_p": float(last_pred_p),
                "ema_alpha": float(args.ema),
                "topk_labels": str(topk_labels),
                "topk_ps": str(topk_ps),
            }
            logger.log_frame(row)

            if did_infer and last_probs is not None and ema_probs is not None:
                logger.log_infer(
                    t_wall_ms=float(now * 1000.0),
                    length=int(length),
                    x_seq=x_seq,
                    probs=last_probs,
                    ema_probs=ema_probs,
                    infer_ms=float(last_infer_ms),
                    pred_label=str(last_pred_label),
                    reset_reason=(reset_reason or None),
                )

        # Annotations (per frame)
        if ann is not None:
            side_frame = ""
            if bbox is not None:
                side_frame = _hand_side_from_bbox(bbox, frame_w=frame_w)
            ann.on_frame(
                t_wall_ms=float(now * 1000.0),
                track_xyz=track,
                lm_xyz=lm,
                pose_valid=bool(pose_valid),
                flow_feats=flow_feats,
                flow_valid=bool(flow_valid),
                hand_side=str(side_frame),
                pred_label=str(last_pred_label),
                pred_p=float(last_pred_p),
                did_infer=bool(did_infer),
                x_seq=(last_x_seq if bool(did_infer) else None),
                probs=(last_probs if bool(did_infer) else None),
                ema_probs=(ema_probs if bool(did_infer) else None),
            )
            ann.draw(disp)

        cv2.imshow(win, disp)
        key = cv2.waitKey(1) & 0xFF
        if ann is not None and key not in (255,):
            ann.on_key(int(key), now_ms=float(now * 1000.0))
        if key == ord("q"):
            break
        if key == ord("r"):
            reset_reason = "manual"
            _reset_state(reset_reason)

    cap.release()
    cv2.destroyAllWindows()
    if logger is not None:
        logger.finalize()
    return 0

if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

