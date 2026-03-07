from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def _load_confusion(path: Path) -> list[tuple[str, str, int]]:
    d = json.loads(path.read_text(encoding="utf-8"))
    rows: list[tuple[str, str, int]] = []
    if not isinstance(d, dict):
        return rows
    for gt, pred_row in d.items():
        if not isinstance(pred_row, dict):
            continue
        for pred, n in pred_row.items():
            try:
                rows.append((str(gt), str(pred), int(n)))
            except Exception:
                continue
    return rows


def _load_jsonl_ratios(path: Path) -> tuple[int, list[float], list[float]]:
    n = 0
    pose: list[float] = []
    flow: list[float] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                d = json.loads(s)
            except Exception:
                continue
            n += 1
            pose.append(float(d.get("pose_valid_ratio", 0.0)))
            flow.append(float(d.get("flow_valid_ratio", 0.0)))
    return n, pose, flow


def _load_manifest_label_counts(path: Path) -> tuple[int, dict[str, int]]:
    counts: dict[str, int] = {}
    n = 0
    with path.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            n += 1
            lab = str(row.get("label") or "").strip()
            if not lab:
                continue
            counts[lab] = int(counts.get(lab, 0)) + 1
    return n, counts


def main() -> int:
    p = argparse.ArgumentParser(description="Analyze DOMA live annotation exports")
    p.add_argument("--session", required=True, help="Path to session dir (doma/sessions/live_...)")
    p.add_argument(
        "--dataset",
        default="",
        help="Optional path to training-ready dataset root (data/annotated/live_...)",
    )
    args = p.parse_args()

    session_dir = Path(args.session).expanduser().resolve()
    ann_dir = session_dir / "annotations"
    conf_path = ann_dir / "confusion_counts.json"
    cap_path = ann_dir / "captures.jsonl"
    seg_path = ann_dir / "segments.csv"

    if not conf_path.exists():
        raise SystemExit(f"Missing {conf_path}")
    if not cap_path.exists():
        raise SystemExit(f"Missing {cap_path}")

    conf_rows = _load_confusion(conf_path)
    total = sum(n for _gt, _pred, n in conf_rows)
    correct = sum(n for gt, pred, n in conf_rows if gt == pred)
    acc = (float(correct) / float(total)) if total else float("nan")

    per_gt: dict[str, dict[str, object]] = {}
    for gt, pred, n in conf_rows:
        st = per_gt.setdefault(gt, {"total": 0, "correct": 0, "preds": {}})
        st["total"] = int(st["total"]) + int(n)
        if gt == pred:
            st["correct"] = int(st["correct"]) + int(n)
        preds = st["preds"]
        assert isinstance(preds, dict)
        preds[pred] = int(preds.get(pred, 0)) + int(n)

    # sort by recall
    per_gt_out = []
    for gt, st in per_gt.items():
        tot = int(st["total"])
        cor = int(st["correct"])
        recall = (float(cor) / float(tot)) if tot else 0.0
        preds = st["preds"]
        assert isinstance(preds, dict)
        top_preds = sorted(preds.items(), key=lambda kv: (-int(kv[1]), str(kv[0])))[:5]
        per_gt_out.append(
            {
                "gt": gt,
                "n": tot,
                "recall": recall,
                "top_preds": top_preds,
            }
        )
    per_gt_out.sort(key=lambda d: (float(d["recall"]), -int(d["n"]), str(d["gt"])))

    cap_n, pose_rat, flow_rat = _load_jsonl_ratios(cap_path)

    out = {
        "session": str(session_dir.as_posix()),
        "files": {
            "captures_jsonl": str(cap_path.as_posix()),
            "segments_csv": str(seg_path.as_posix()) if seg_path.exists() else None,
            "confusion_counts_json": str(conf_path.as_posix()),
        },
        "captures_jsonl_lines": int(cap_n),
        "confusion_majority": {
            "n": int(total),
            "correct": int(correct),
            "accuracy": float(acc),
        },
        "validity": {
            "pose_valid_mean": float(np.mean(pose_rat)) if pose_rat else float("nan"),
            "flow_valid_mean": float(np.mean(flow_rat)) if flow_rat else float("nan"),
            "pose_valid_min": float(np.min(pose_rat)) if pose_rat else float("nan"),
            "flow_valid_min": float(np.min(flow_rat)) if flow_rat else float("nan"),
        },
        "per_gt_recall_sorted_low_to_high": per_gt_out,
    }

    if args.dataset:
        ds = Path(args.dataset).expanduser().resolve()
        man = ds / "manifest.csv"
        if man.exists():
            n_rows, lab_counts = _load_manifest_label_counts(man)
            out["dataset"] = {
                "root": str(ds.as_posix()),
                "manifest_csv": str(man.as_posix()),
                "manifest_rows": int(n_rows),
                "label_counts": dict(
                    sorted(lab_counts.items(), key=lambda kv: (-kv[1], kv[0]))
                ),
            }
        else:
            out["dataset"] = {"root": str(ds.as_posix()), "manifest_csv": None}

    print(json.dumps(out, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

