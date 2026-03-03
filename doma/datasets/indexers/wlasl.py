from __future__ import annotations

import json
from pathlib import Path

from ..schema import SampleIndex


def index_wlasl(raw_root: Path, cfg: dict, *, subset_limit: int = 0) -> list[SampleIndex]:
    """
    Expects a WLASL JSON (e.g., WLASL2000.json).
    Common structure:
      [
        { "gloss": "...", "instances": [ { "video_id": "...", "url": "...", ...}, ... ] },
        ...
      ]
    """
    base = raw_root / str(cfg.get("raw_dir", "wlasl"))
    ann = (cfg.get("annotations") or {}) if isinstance(cfg.get("annotations"), dict) else {}
    rel = ann.get("json")
    if not rel:
        return []
    p = base / str(rel)
    if not p.exists():
        return []
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        return []

    out: list[SampleIndex] = []
    # WLASL JSON may include split info per instance; if absent, default train.
    for entry in data:
        gloss = str(entry.get("gloss") or entry.get("label") or "unknown")
        inst = entry.get("instances") or []
        if not isinstance(inst, list):
            continue
        for it in inst:
            if subset_limit and len(out) >= subset_limit:
                return out
            vid = str(it.get("video_id") or it.get("id") or "")
            url = str(it.get("url") or "")
            split = str(it.get("split") or "train")
            if not vid or not url:
                continue
            mp4 = base / "videos" / f"{vid}.mp4"
            out.append(
                SampleIndex(
                    sample_id=f"wlasl_{split}_{vid}",
                    dataset="wlasl",
                    split=split,  # type: ignore[arg-type]
                    label=gloss,
                    text=gloss,
                    source_uri=url,
                    video_path=str(mp4),
                )
            )
    return out

