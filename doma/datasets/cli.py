from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

from .builder import BuildOptions, build_dataset, process_sample
from .config import load_config
from .indexers.ipn_hand import index_ipn_hand
from .indexers.jester import index_jester
from .indexers.ms_asl import index_ms_asl
from .indexers.wlasl import index_wlasl


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Build unified dataset (pose tensors + optical-flow features)")
    p.add_argument("--config", default="config/datasets.yaml", help="Path to datasets config (.yaml/.toml/.json)")
    p.add_argument("--out", default="", help="Output processed root (defaults to config processed_root)")
    p.add_argument("--only", default="", help="Comma-separated dataset names to process (optional)")
    p.add_argument("--subset", type=int, default=0, help="Limit number of samples processed (debug)")
    p.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Limit frames per video (debug; 0 = full video)",
    )
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing artifacts")
    p.add_argument("--download", action="store_true", help="Download missing MS-ASL/WLASL videos via yt-dlp")
    args = p.parse_args(argv)

    cfg = load_config(Path(args.config))
    out_root = Path(args.out) if args.out else cfg.processed_root

    only = {s.strip() for s in args.only.split(",") if s.strip()} if args.only else None

    samples = []
    for name, dcfg in cfg.datasets.items():
        if only is not None and name not in only:
            continue
        if not bool(dcfg.get("enabled", False)):
            continue

        if name == "ipn_hand":
            samples.extend(index_ipn_hand(cfg.raw_root, dcfg, subset_limit=dcfg.get("subset_limit", 0) or 0))
        elif name == "jester":
            samples.extend(index_jester(cfg.raw_root, dcfg, subset_limit=dcfg.get("subset_limit", 0) or 0))
        elif name == "ms_asl":
            samples.extend(index_ms_asl(cfg.raw_root, dcfg, subset_limit=dcfg.get("subset_limit", 0) or 0))
        elif name == "wlasl":
            samples.extend(index_wlasl(cfg.raw_root, dcfg, subset_limit=dcfg.get("subset_limit", 0) or 0))

    if args.download:
        _download_missing(samples, cfg.raw_root, max_retries=3)
        # Drop rows that are still missing after download (avoids failing the whole build).
        kept = []
        for s in samples:
            if s.video_path and Path(s.video_path).exists():
                kept.append(s)
        samples = kept

    build_dataset(
        cfg,
        samples=samples,
        out_dir=out_root,
        opts=BuildOptions(
            overwrite=args.overwrite,
            subset_limit=args.subset,
            max_frames=args.max_frames,
        ),
    )
    return 0


def _download_missing(samples, raw_root: Path, *, max_retries: int = 3) -> None:
    yt = shutil.which("yt-dlp")
    if not yt:
        raise RuntimeError("yt-dlp not found in PATH. Install yt-dlp or run without --download.")
    ffmpeg = shutil.which("ffmpeg")

    for s in samples:
        if s.dataset not in {"ms_asl", "wlasl"}:
            continue
        if not s.video_path:
            continue
        out_path = Path(s.video_path)
        if out_path.exists():
            continue
        out_path.parent.mkdir(parents=True, exist_ok=True)

        tmp_tpl = str(out_path.with_suffix(".%(ext)s"))
        ok = False
        last_err = ""
        for _try in range(int(max_retries)):
            cmd = [yt, "-f", "mp4/best", "-o", tmp_tpl, s.source_uri]
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode == 0:
                ok = True
                break
            last_err = r.stderr or r.stdout or ""
        if not ok:
            # keep going; sample will be filtered out upstream
            continue

        # Prefer a produced mp4 if available.
        produced = next(out_path.parent.glob(out_path.stem + ".mp4"), None)
        if produced is None:
            produced = next(out_path.parent.glob(out_path.stem + ".*"), None)
        if produced is None:
            continue

        if produced.suffix.lower() == ".mp4":
            if produced.name != out_path.name:
                produced.rename(out_path)
            continue

        # Remux to mp4 when possible.
        if ffmpeg:
            cmd = [ffmpeg, "-y", "-i", str(produced), "-c", "copy", str(out_path)]
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode == 0 and out_path.exists():
                try:
                    produced.unlink()
                except OSError:
                    pass

