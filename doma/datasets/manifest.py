from __future__ import annotations

import csv
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

from .schema import SampleIndex


MANIFEST_FIELDS = [
    "sample_id",
    "dataset",
    "split",
    "label",
    "text",
    "source_uri",
    "video_path",
    "fps",
    "num_frames",
    # artifact paths (resolved at write time)
    "pose_npz",
    "optflow_npz",
    "quality_json",
    "rgb_mp4",
]


def write_manifest_csv(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def sample_to_row(sample: SampleIndex, artifacts: dict[str, str] | None = None) -> dict:
    d = asdict(sample)
    d["text"] = d.pop("text", None)
    if artifacts:
        d.update(artifacts)
    return d

