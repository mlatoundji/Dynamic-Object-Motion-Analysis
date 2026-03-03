from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Generate a minimal IPN Hand index.csv from a videos folder"
    )
    p.add_argument(
        "--raw",
        default="data/raw/ipn_hand",
        help="IPN Hand raw root (contains videos/)",
    )
    p.add_argument("--split", default="train", choices=["train", "val", "test"])
    p.add_argument("--label", default="unknown")
    p.add_argument(
        "--glob",
        default="videos/**/*.*",
        help="Glob for videos under raw root (e.g. videos/**/*.avi)",
    )
    args = p.parse_args(argv)

    raw = Path(args.raw)
    vids = sorted(raw.glob(args.glob))
    if not vids:
        raise SystemExit(f"No videos found under {raw} with glob={args.glob}")

    out = raw / "index.csv"
    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["sample_id", "split", "label", "video_path", "source_uri"],
        )
        w.writeheader()
        for v in vids:
            rel = v.relative_to(raw).as_posix()
            w.writerow(
                {
                    "sample_id": f"ipn_{args.split}_{v.stem}",
                    "split": args.split,
                    "label": args.label,
                    "video_path": rel,
                    "source_uri": rel,
                }
            )
    print(f"Wrote {out} with {len(vids)} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
