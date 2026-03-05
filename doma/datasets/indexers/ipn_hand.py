from __future__ import annotations

import csv
import random
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

    # Prefer official IPN annotations if present (segment-aware).
    ann_cfg = cfg.get("annotations") if isinstance(cfg.get("annotations"), dict) else {}
    ann_root = base / str((ann_cfg or {}).get("root", "annotations"))
    ann_dir = _find_ipn_annotations_dir(ann_root)
    if ann_dir is not None:
        val_ratio = float(cfg.get("val_ratio", 0.1))
        split_seed = int(cfg.get("seed", 0))
        return _index_ipn_segments(
            base=base,
            ann_dir=ann_dir,
            val_ratio=val_ratio,
            seed=split_seed,
            subset_limit=subset_limit,
        )

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


def _find_ipn_annotations_dir(ann_root: Path) -> Path | None:
    """
    IPN annotations come as a folder containing:
    - Video_TrainList.txt / Video_TestList.txt
    - Annot_TrainList.txt / Annot_TestList.txt
    - classIdx.txt
    """
    if not ann_root.exists():
        return None
    candidates = []
    for p in ann_root.rglob("Annot_TrainList.txt"):
        d = p.parent
        needed = [
            d / "Annot_TestList.txt",
            d / "Video_TrainList.txt",
            d / "Video_TestList.txt",
            d / "classIdx.txt",
        ]
        if all(x.exists() for x in needed):
            candidates.append(d)
    if not candidates:
        return None
    # Pick the deepest folder (most specific) deterministically.
    candidates.sort(key=lambda x: (len(x.parts), str(x)))
    return candidates[-1]


def _index_ipn_segments(
    *,
    base: Path,
    ann_dir: Path,
    val_ratio: float,
    seed: int,
    subset_limit: int,
) -> list[SampleIndex]:
    """
    Build one SampleIndex per annotated segment.
    Splits:
    - official train/test from Video_*List.txt
    - val split created by sampling videos from official train (anti-leak)
    """
    train_videos = _read_video_list(ann_dir / "Video_TrainList.txt")
    test_videos = _read_video_list(ann_dir / "Video_TestList.txt")

    rng = random.Random(int(seed))
    train_vids = sorted(train_videos)
    rng.shuffle(train_vids)
    n_val = int(round(len(train_vids) * float(val_ratio)))
    val_set = set(train_vids[:n_val])
    train_set = set(train_vids[n_val:])

    # Map video_id -> actual file path (avi preferred).
    video_map = _build_video_map(base / "videos")

    train_segments: list[SampleIndex] = []
    val_segments: list[SampleIndex] = []
    test_segments: list[SampleIndex] = []

    # Parse segments
    for seg in _read_annot_list(ann_dir / "Annot_TrainList.txt"):
        video = seg["video"]
        if video not in train_videos:
            continue
        split = "val" if video in val_set else "train"
        sample = _make_segment_sample(
            base=base,
            video_map=video_map,
            ann_path=ann_dir / "Annot_TrainList.txt",
            split=split,
            **seg,
        )
        if sample is None:
            continue
        (val_segments if split == "val" else train_segments).append(sample)

    for seg in _read_annot_list(ann_dir / "Annot_TestList.txt"):
        video = seg["video"]
        if video not in test_videos:
            continue
        sample = _make_segment_sample(
            base=base,
            video_map=video_map,
            ann_path=ann_dir / "Annot_TestList.txt",
            split="test",
            **seg,
        )
        if sample is None:
            continue
        test_segments.append(sample)

    # Stable ordering for debugging: ensure early subset hits train/val/test.
    ordered: list[SampleIndex] = []
    if train_segments:
        ordered.append(train_segments[0])
    if val_segments:
        ordered.append(val_segments[0])
    if test_segments:
        ordered.append(test_segments[0])
    ordered.extend(train_segments[1:])
    ordered.extend(val_segments[1:])
    ordered.extend(test_segments[1:])

    if subset_limit:
        return ordered[: int(subset_limit)]
    return ordered


def _build_video_map(videos_root: Path) -> dict[str, Path]:
    out: dict[str, Path] = {}
    if not videos_root.exists():
        return out
    # Prefer avi if both exist.
    for p in videos_root.rglob("*.mp4"):
        out[p.stem] = p
    for p in videos_root.rglob("*.avi"):
        out[p.stem] = p
    return out


def _read_video_list(path: Path) -> set[str]:
    vids: set[str] = set()
    if not path.exists():
        return vids
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        # Format: "<video>\t<frames>"
        video = line.split()[0]
        vids.add(video)
    return vids


def _read_annot_list(path: Path) -> list[dict[str, int | str]]:
    """
    Supports:
    - With header (Annot_List.txt): video,label,id,t_start,t_end,frames
    - No header (Annot_TrainList/TestList): same columns, comma-separated
    """
    rows = []
    if not path.exists():
        return rows
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        return rows
    start_idx = 1 if lines[0].lower().startswith("video,") else 0
    for line in lines[start_idx:]:
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 6:
            continue
        video, label, _id, t_start, t_end, frames = parts[:6]
        rows.append(
            {
                "video": video,
                "label": label,
                "t_start": int(t_start),
                "t_end": int(t_end),
                "frames": int(frames),
            }
        )
    return rows


def _make_segment_sample(
    *,
    base: Path,
    video_map: dict[str, Path],
    ann_path: Path,
    split: str,
    video: str,
    label: str,
    t_start: int,
    t_end: int,
    frames: int,
) -> SampleIndex | None:
    vp = video_map.get(video)
    if vp is None or not vp.exists():
        return None
    # Use relative uri for readability; keep absolute video_path for IO.
    try:
        rel = vp.relative_to(base).as_posix()
    except Exception:
        rel = vp.as_posix()
    sample_id = f"ipn_{split}_{video}_{label}_{t_start}_{t_end}"
    return SampleIndex(
        sample_id=sample_id,
        dataset="ipn_hand",
        split=split,  # type: ignore[arg-type]
        label=label,
        source_uri=rel,
        video_path=str(vp),
        frame_start=int(t_start),
        frame_end=int(t_end),
        parent_video=video,
        source_annotation=str(ann_path.as_posix()),
        num_frames=int(frames),
    )

