from __future__ import annotations

import csv
from pathlib import Path

from ..schema import SampleIndex


def index_jester(raw_root: Path, cfg: dict, *, subset_limit: int = 0) -> list[SampleIndex]:
    """
    Supports common Jester/20BN-Jester formats:
    - labels file: labels.csv or jester-v1-labels.csv (one label per line)
    - split files: train.csv / validation.csv / test.csv
      Lines can be either:
        - "<video_id>;<label>"
        - "<video_id> <label>"
        - "<video_id>\\t<label>"
    Frame layout supported:
      - raw_root/jester/20bn-jester-v1/<video_id>/*.jpg  (converted later)
    Video layout supported:
      - raw_root/jester/videos/<video_id>.mp4
    """
    base = raw_root / str(cfg.get("raw_dir", "jester"))
    split_map = {"train": "train.csv", "val": "validation.csv", "test": "test.csv"}

    out: list[SampleIndex] = []
    for split, fname in split_map.items():
        p = base / fname
        if not p.exists():
            continue
        with p.open("r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f):
                if subset_limit and len(out) >= subset_limit:
                    return out
                line = line.strip()
                if not line:
                    continue
                if ";" in line:
                    vid, label = line.split(";", 1)
                elif "\t" in line:
                    vid, label = line.split("\t", 1)
                else:
                    parts = line.split()
                    if len(parts) < 2:
                        continue
                    vid, label = parts[0], " ".join(parts[1:])

                vid = vid.strip()
                label = label.strip()
                mp4 = base / "videos" / f"{vid}.mp4"
                frames_dir = base / "20bn-jester-v1" / vid

                video_path = None
                source_uri = ""
                if mp4.exists():
                    video_path = str(mp4)
                    source_uri = str(mp4.as_posix())
                elif frames_dir.exists():
                    # We don’t process frames-directories yet in builder; require user to convert to mp4.
                    # Still emit the path so we can fail fast with a helpful message in docs.
                    video_path = str(frames_dir)
                    source_uri = str(frames_dir.as_posix())
                else:
                    continue

                out.append(
                    SampleIndex(
                        sample_id=f"jester_{split}_{vid}",
                        dataset="jester",
                        split=split,  # type: ignore[arg-type]
                        label=label,
                        source_uri=source_uri,
                        video_path=video_path,
                    )
                )
    return out

