from __future__ import annotations

import csv
from pathlib import Path

from ..schema import SampleIndex


def index_ipn_hand(
    raw_root: Path, cfg: dict, *, subset_limit: int = 0
) -> list[SampleIndex]:
    """
    IPN Hand is continuous; annotations vary by distribution.

    Supported minimal modes:
    - If `${raw_root}/ipn_hand/index.csv` exists, read it (preferred).
      Required columns: sample_id, split, label, video_path, source_uri
    - Else, fall back to indexing any video under
      `${raw_root}/ipn_hand/videos/**/*.(mp4|avi)`
      with label="unknown" and split="train".
    """
    base = raw_root / str(cfg.get("raw_dir", "ipn_hand"))
    index_csv = base / "index.csv"
    out: list[SampleIndex] = []

    if index_csv.exists():
        with index_csv.open("r", encoding="utf-8", newline="") as f:
            r = csv.DictReader(f)
            for row in r:
                if subset_limit and len(out) >= subset_limit:
                    break
                sample_id = row.get("sample_id") or ""
                split = (row.get("split") or "train").strip()
                label = (row.get("label") or "unknown").strip()
                video_path = row.get("video_path") or ""
                source_uri = row.get("source_uri") or video_path
                if not sample_id or not video_path:
                    continue
                p = (
                    (base / video_path)
                    if not Path(video_path).is_absolute()
                    else Path(video_path)
                )
                if not p.exists():
                    continue
                out.append(
                    SampleIndex(
                        sample_id=sample_id,
                        dataset="ipn_hand",
                        split=split,  # type: ignore[arg-type]
                        label=label,
                        source_uri=source_uri,
                        video_path=str(p),
                    )
                )
        return out

    vids = sorted((base / "videos").glob("**/*.mp4"))
    vids += sorted((base / "videos").glob("**/*.avi"))
    for i, p in enumerate(vids):
        if subset_limit and i >= subset_limit:
            break
        out.append(
            SampleIndex(
                sample_id=f"ipn_train_{p.stem}",
                dataset="ipn_hand",
                split="train",
                label="unknown",
                source_uri=str(p.as_posix()),
                video_path=str(p),
            )
        )
    return out

