from __future__ import annotations

import argparse
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Quick validation of Jester raw layout")
    p.add_argument("--raw", default="data/raw/jester")
    args = p.parse_args(argv)

    base = Path(args.raw)
    missing = []
    for fname in ["train.csv", "validation.csv"]:
        if not (base / fname).exists():
            missing.append(fname)
    if missing:
        raise SystemExit(f"Missing files in {base}: {', '.join(missing)}")

    frames_root = base / "20bn-jester-v1"
    videos_root = base / "videos"
    if not frames_root.exists() and not videos_root.exists():
        raise SystemExit(f"Expected {frames_root} or {videos_root} to exist")

    print("OK:", base)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

