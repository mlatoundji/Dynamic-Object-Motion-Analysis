from __future__ import annotations

import argparse
import time
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .detectors import ManualROI, MediaPipeHandsDetector, YOLODetector
from .flow import farneback, raft_dense
from .motion import angle_to_cardinal, compute_motion_stats
from .viz import draw_bbox, draw_flow_arrows


def _open_source(source: str) -> cv2.VideoCapture:
    try:
        i = int(source)
        cap = cv2.VideoCapture(i)
    except ValueError:
        cap = cv2.VideoCapture(source)
    return cap


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Dynamic Object Motion Analysis (hand-guided optical flow)")
    p.add_argument("--source", default="0", help="camera index (e.g. 0) or video path")
    p.add_argument("--flow", choices=["farneback", "raft"], default="farneback")
    p.add_argument("--detector", choices=["manual", "mediapipe", "yolo"], default="manual")
    p.add_argument("--threshold", choices=["fixed", "otsu", "mad"], default="otsu")
    p.add_argument("--fixed-thr", type=float, default=2.0)
    p.add_argument("--step", type=int, default=10, help="vector drawing step")
    p.add_argument("--export", default="", help="CSV export path (optional)")
    args = p.parse_args(argv)

    cap = _open_source(args.source)
    if not cap.isOpened():
        raise SystemExit("Erreur: impossible d'ouvrir la source vidéo.")

    win = "DOMA - Hand Guided Optical Flow"
    cv2.namedWindow(win)

    manual = ManualROI()
    if args.detector == "manual":
        cv2.setMouseCallback(win, manual.mouse_cb)
        detector = None
        yolo = None
    elif args.detector == "mediapipe":
        detector = MediaPipeHandsDetector(max_num_hands=1)
        yolo = None
    else:
        detector = None
        yolo = YOLODetector()

    prev_gray = None
    prev_bgr = None
    prev_roi_gray = None
    prev_time = time.time()

    rows: list[dict] = []

    print("Contrôles :")
    print("  - 'q' : quitter")
    print("  - 'r' : reset ROI (mode manual)")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        disp = frame.copy()

        bbox = None
        mask = None
        if args.detector == "manual":
            bbox = manual.bbox
        elif args.detector == "mediapipe":
            bbox, mask = detector.detect(frame)  # type: ignore[union-attr]
        else:
            bbox = yolo.detect(frame)  # type: ignore[union-attr]

        now = time.time()
        dt = max(1e-6, now - prev_time)
        fps = 1.0 / dt
        prev_time = now

        if bbox is not None and bbox.w > 0 and bbox.h > 0:
            draw_bbox(disp, bbox)
            roi_gray = gray[bbox.y : bbox.y + bbox.h, bbox.x : bbox.x + bbox.w]
            roi_bgr = frame[bbox.y : bbox.y + bbox.h, bbox.x : bbox.x + bbox.w]

            if mask is not None:
                roi_mask = mask[bbox.y : bbox.y + bbox.h, bbox.x : bbox.x + bbox.w]
            else:
                roi_mask = None

            flow = None
            if args.flow == "farneback":
                if prev_roi_gray is not None and prev_roi_gray.shape == roi_gray.shape:
                    flow = farneback(prev_roi_gray, roi_gray)
            else:
                if prev_bgr is not None:
                    flow_full = raft_dense(prev_bgr, frame)
                    flow = flow_full[bbox.y : bbox.y + bbox.h, bbox.x : bbox.x + bbox.w]

            if flow is not None:
                if roi_mask is not None:
                    # Apply mask by zeroing outside.
                    flow = flow.copy()
                    flow[~roi_mask] = 0.0

                stats, motion_mask = compute_motion_stats(
                    flow,
                    threshold_method=args.threshold,
                    fixed_threshold=args.fixed_thr,
                    subtract_bg=True,
                )

                draw_flow_arrows(disp, flow, origin_xy=(bbox.x, bbox.y), step=args.step, mask=motion_mask)

                direction = angle_to_cardinal(stats.dominant_angle_deg) if stats.avg_speed > 0 else "Stable"
                info = (
                    f"FPS:{fps:4.1f} | v:{stats.avg_speed:5.2f}px/f | "
                    f"dir:{direction} ({stats.dominant_angle_deg:3.0f}°) | R:{stats.direction_concentration:.2f}"
                )
                cv2.putText(disp, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                rows.append(
                    {
                        "t": now,
                        "fps": fps,
                        "detector": args.detector,
                        "flow": args.flow,
                        **asdict(stats),
                    }
                )

            prev_roi_gray = roi_gray.copy()
        else:
            prev_roi_gray = None
            cv2.putText(disp, "Selectionnez la main (manual) ou attendez la detection", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        prev_gray = gray
        prev_bgr = frame

        cv2.imshow(win, disp)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("r") and args.detector == "manual":
            manual.reset()
            prev_roi_gray = None

    cap.release()
    cv2.destroyAllWindows()

    if args.export:
        out = Path(args.export)
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f"Export CSV: {out}")

    return 0


