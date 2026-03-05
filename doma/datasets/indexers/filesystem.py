from __future__ import annotations

from pathlib import Path
from typing import Iterable

from ..schema import SampleIndex


def index_videos_under(
    raw_root: Path,
    *,
    dataset: str,
    split: str,
    label: str,
    glob: str = "**/*.mp4",
    subset_limit: int = 0,
) -> list[SampleIndex]:
    base = raw_root / dataset
    paths = sorted(base.glob(glob))
    out: list[SampleIndex] = []
    for i, p in enumerate(paths):
        if subset_limit and i >= subset_limit:
            break
        sample_id = p.stem
        out.append(
            SampleIndex(
                sample_id=f"{dataset}_{split}_{sample_id}",
                dataset=dataset,  # type: ignore[arg-type]
                split=split,  # type: ignore[arg-type]
                label=label,
                source_uri=str(p.as_posix()),
                video_path=str(p),
            )
        )
    return out

