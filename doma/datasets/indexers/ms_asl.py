from __future__ import annotations

import json
from pathlib import Path

from ..schema import SampleIndex


def index_ms_asl(raw_root: Path, cfg: dict, *, subset_limit: int = 0) -> list[SampleIndex]:
    """
    Expects official MS-ASL split JSON files.
    Fields vary, but commonly include:
      - 'video_id' or 'id'
      - 'label' (int) and/or 'text'/'gloss'
      - 'url' (YouTube)

    This indexer only emits rows; downloading is handled in the CLI/builder step.
    """
    base = raw_root / str(cfg.get("raw_dir", "ms_asl"))
    ann = (cfg.get("annotations") or {}) if isinstance(cfg.get("annotations"), dict) else {}
    split_files = {
        "train": ann.get("train_json"),
        "val": ann.get("val_json"),
        "test": ann.get("test_json"),
    }
    out: list[SampleIndex] = []

    for split, rel in split_files.items():
        if not rel:
            continue
        p = base / str(rel)
        if not p.exists():
            continue
        items = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(items, dict) and "annotations" in items:
            items = items["annotations"]
        if not isinstance(items, list):
            continue
        for it in items:
            if subset_limit and len(out) >= subset_limit:
                return out
            vid = str(it.get("video_id") or it.get("id") or "")
            url = str(it.get("url") or "")
            label = str(it.get("text") or it.get("gloss") or it.get("label") or "unknown")
            if not vid or not url:
                continue
            # downloaded videos go to base/videos/<video_id>.mp4
            mp4 = base / "videos" / f"{vid}.mp4"
            out.append(
                SampleIndex(
                    sample_id=f"msasl_{split}_{vid}",
                    dataset="ms_asl",
                    split=split,  # type: ignore[arg-type]
                    label=label,
                    text=str(it.get("text") or it.get("gloss") or ""),
                    source_uri=url,
                    video_path=str(mp4),
                )
            )
    return out

