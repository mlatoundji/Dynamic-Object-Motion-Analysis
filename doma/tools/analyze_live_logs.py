from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Segment:
    start_ms: float
    end_ms: float
    label: str


def _read_report_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        return [dict(row) for row in r]


def _to_float(s: str | None) -> float:
    if s is None:
        return float("nan")
    ss = str(s).strip()
    if not ss:
        return float("nan")
    try:
        return float(ss)
    except Exception:
        return float("nan")


def _to_int(s: str | None, default: int = 0) -> int:
    if s is None:
        return int(default)
    ss = str(s).strip()
    if not ss:
        return int(default)
    try:
        return int(float(ss))
    except Exception:
        return int(default)


def _majority_label(labels: list[str]) -> str:
    counts: dict[str, int] = {}
    for lab in labels:
        if not lab:
            continue
        counts[lab] = int(counts.get(lab, 0)) + 1
    if not counts:
        return ""
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]


def _read_segments_csv(path: Path) -> list[Segment]:
    with path.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        out: list[Segment] = []
        for row in r:
            start_ms = _to_float(row.get("t_start_ms"))
            end_ms = _to_float(row.get("t_end_ms"))
            label = str(row.get("label") or "").strip()
            if (
                not np.isfinite(start_ms)
                or not np.isfinite(end_ms)
                or not label
            ):
                continue
            if end_ms <= start_ms:
                continue
            out.append(
                Segment(
                    start_ms=float(start_ms),
                    end_ms=float(end_ms),
                    label=label,
                )
            )
        return out


def _compute_stability(pred_labels: list[str]) -> dict[str, Any]:
    # Stability stats on a sequence of labels (per frame).
    changes = 0
    runs: list[int] = []
    cur = ""
    run_len = 0
    for lab in pred_labels:
        if lab != cur:
            if run_len > 0:
                runs.append(run_len)
            changes += 1 if cur else 0
            cur = lab
            run_len = 1
        else:
            run_len += 1
    if run_len > 0:
        runs.append(run_len)

    return {
        "label_changes": int(changes),
        "num_runs": int(len(runs)),
        "avg_run_len_frames": float(np.mean(runs)) if runs else 0.0,
        "median_run_len_frames": float(np.median(runs)) if runs else 0.0,
        "p90_run_len_frames": float(np.percentile(runs, 90)) if runs else 0.0,
    }


def _first_stable_match_latency(
    *,
    t_ms: np.ndarray,
    pred_label: list[str],
    pred_p: np.ndarray,
    gt_label: str,
    start_ms: float,
    end_ms: float,
    stable_k: int,
    p_thr: float,
) -> float:
    idx = np.where((t_ms >= start_ms) & (t_ms <= end_ms))[0]
    if idx.size == 0:
        return float("nan")
    k = int(max(1, stable_k))
    for j in range(int(idx.size)):
        i0 = int(idx[j])
        i1 = i0 + k
        if i1 > int(len(pred_label)):
            break
        ok = True
        for i in range(i0, i1):
            if pred_label[i] != gt_label:
                ok = False
                break
            if np.isfinite(p_thr) and float(pred_p[i]) < float(p_thr):
                ok = False
                break
        if ok:
            return float(t_ms[i0] - start_ms)
    return float("nan")


def analyze(
    *,
    csv_path: Path,
    segments_path: Path | None,
    stable_k: int,
    p_thr: float,
) -> dict[str, Any]:
    rows = _read_report_csv(csv_path)
    if not rows:
        return {"error": "empty_csv", "csv": str(csv_path.as_posix())}

    t_wall_ms = np.asarray(
        [_to_float(r.get("t_wall_ms")) for r in rows],
        dtype=np.float64,
    )
    did_infer = np.asarray(
        [_to_int(r.get("did_infer"), 0) for r in rows],
        dtype=np.int32,
    )
    infer_ms = np.asarray(
        [_to_float(r.get("infer_ms")) for r in rows],
        dtype=np.float64,
    )
    reset_reason = [str(r.get("reset_reason") or "").strip() for r in rows]
    pred_label = [str(r.get("pred_label") or "").strip() for r in rows]
    pred_p = np.asarray(
        [_to_float(r.get("pred_p")) for r in rows],
        dtype=np.float64,
    )

    t0 = (
        float(np.nanmin(t_wall_ms))
        if np.isfinite(t_wall_ms).any()
        else 0.0
    )
    t_rel_ms = t_wall_ms - float(t0)
    duration_s = (
        float(np.nanmax(t_rel_ms) / 1000.0)
        if np.isfinite(t_rel_ms).any()
        else 0.0
    )

    resets = int(sum(1 for s in reset_reason if s))

    infer_mask = did_infer.astype(bool) & np.isfinite(infer_ms)
    infer_ms_avg = (
        float(np.mean(infer_ms[infer_mask]))
        if int(np.count_nonzero(infer_mask))
        else 0.0
    )

    # Distribution on inference instants only (more meaningful)
    pred_counts: dict[str, int] = {}
    for i, lab in enumerate(pred_label):
        if not bool(did_infer[i]):
            continue
        if not lab:
            continue
        pred_counts[lab] = int(pred_counts.get(lab, 0)) + 1

    # Stability computed on per-frame labels (last inference held)
    stability = _compute_stability(pred_labels=pred_label)

    out: dict[str, Any] = {
        "csv": str(csv_path.as_posix()),
        "frames": int(len(rows)),
        "duration_s": float(duration_s),
        "fps_avg": float((len(rows) / duration_s) if duration_s > 1e-6 else 0.0),
        "resets": int(resets),
        "infer": {
            "count": int(np.count_nonzero(did_infer)),
            "avg_infer_ms": float(infer_ms_avg),
        },
        "pred_counts_on_infer": dict(
            sorted(pred_counts.items(), key=lambda kv: (-kv[1], kv[0]))
        ),
        "stability": stability,
    }

    if segments_path is None:
        return out

    segs = _read_segments_csv(segments_path)
    if not segs:
        out["segments"] = {
            "error": "no_segments_parsed",
            "segments_csv": str(segments_path.as_posix()),
        }
        return out

    labels_gt = sorted({s.label for s in segs})
    labels_pred = sorted({lab for lab in pred_label if lab})
    label_to_i = {lab: i for i, lab in enumerate(labels_gt)}
    pred_to_j = {lab: j for j, lab in enumerate(labels_pred)}
    cm = np.zeros((len(labels_gt), len(labels_pred)), dtype=np.int32)

    latencies: list[dict[str, Any]] = []
    seg_rows: list[dict[str, Any]] = []
    for s in segs:
        idx = np.where(
            (t_rel_ms >= float(s.start_ms)) & (t_rel_ms <= float(s.end_ms))
        )[0]
        labs = [pred_label[int(i)] for i in idx.tolist() if pred_label[int(i)]]
        maj = _majority_label(labs)
        seg_rows.append(
            {
                "label": s.label,
                "pred_majority": maj,
                "t_start_ms": float(s.start_ms),
                "t_end_ms": float(s.end_ms),
            }
        )
        if s.label in label_to_i and maj in pred_to_j:
            cm[label_to_i[s.label], pred_to_j[maj]] += 1

        lat_ms = _first_stable_match_latency(
            t_ms=t_rel_ms,
            pred_label=pred_label,
            pred_p=pred_p,
            gt_label=s.label,
            start_ms=float(s.start_ms),
            end_ms=float(s.end_ms),
            stable_k=int(stable_k),
            p_thr=float(p_thr),
        )
        latencies.append({"label": s.label, "latency_ms": float(lat_ms)})

    out["segments"] = {
        "segments_csv": str(segments_path.as_posix()),
        "stable_k": int(stable_k),
        "p_thr": float(p_thr),
        "labels_gt": labels_gt,
        "labels_pred": labels_pred,
        "cm_majority": cm.tolist(),
        "segment_rows": seg_rows,
        "latency_ms_by_segment": latencies,
        "latency_ms_summary": {
            "mean": (
                float(np.nanmean([d["latency_ms"] for d in latencies]))
                if latencies
                else float("nan")
            ),
            "median": (
                float(np.nanmedian([d["latency_ms"] for d in latencies]))
                if latencies
                else float("nan")
            ),
            "p90": (
                float(np.nanpercentile([d["latency_ms"] for d in latencies], 90))
                if latencies
                else float("nan")
            ),
        },
    }
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Analyze DOMA live classifier CSV logs")
    p.add_argument("--csv", required=True, help="Path to report_*.csv")
    p.add_argument(
        "--segments",
        default="",
        help=(
            "Optional segments CSV with columns: t_start_ms,t_end_ms,label "
            "(time is relative to first frame)"
        ),
    )
    p.add_argument(
        "--stable-k",
        type=int,
        default=3,
        help="Consecutive frames required for a stable match (latency metric)",
    )
    p.add_argument(
        "--p-thr",
        type=float,
        default=0.0,
        help="Min pred_p to count as match for latency (0 disables)",
    )
    p.add_argument("--out", default="", help="Optional output JSON path (otherwise prints)")
    args = p.parse_args(argv)

    csv_path = Path(str(args.csv)).expanduser().resolve()
    seg_path = (
        Path(str(args.segments)).expanduser().resolve()
        if str(args.segments).strip()
        else None
    )

    report = analyze(
        csv_path=csv_path,
        segments_path=seg_path,
        stable_k=int(args.stable_k),
        p_thr=float(args.p_thr),
    )

    payload = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    if str(args.out).strip():
        out_path = Path(str(args.out)).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(payload, encoding="utf-8")
    else:
        print(payload)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
