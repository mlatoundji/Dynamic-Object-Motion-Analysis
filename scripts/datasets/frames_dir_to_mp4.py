from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Convert a directory of frames (jpg/png) to an mp4 using ffmpeg concat")
    p.add_argument("frames_dir", help="Directory containing frames")
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--out", default="", help="Output mp4 path (default: <frames_dir>.mp4 next to dir)")
    args = p.parse_args(argv)

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise SystemExit("ffmpeg not found in PATH")

    d = Path(args.frames_dir)
    if not d.is_dir():
        raise SystemExit(f"Not a directory: {d}")

    frames = sorted(d.glob("*.jpg"))
    if not frames:
        frames = sorted(d.glob("*.png"))
    if not frames:
        raise SystemExit(f"No .jpg/.png frames in {d}")

    out = Path(args.out) if args.out else d.with_suffix(".mp4")
    out.parent.mkdir(parents=True, exist_ok=True)

    # Build concat list (works with arbitrary filenames)
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".txt", encoding="utf-8") as f:
        for fr in frames:
            f.write(f"file '{fr.as_posix()}'\n")
        list_path = f.name

    cmd = [
        ffmpeg,
        "-y",
        "-r",
        str(float(args.fps)),
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        list_path,
        "-pix_fmt",
        "yuv420p",
        str(out),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(r.stderr or r.stdout)

    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

