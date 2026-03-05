from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Split an index.csv into train/val/test (in-place)"
    )
    p.add_argument(
        "--index",
        default="data/raw/ipn_hand/index.csv",
        help="Path to index.csv",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--train", type=float, default=0.8)
    p.add_argument("--val", type=float, default=0.1)
    p.add_argument("--test", type=float, default=0.1)
    p.add_argument("--stratify", default="label", choices=["", "label"])
    args = p.parse_args(argv)

    if abs((args.train + args.val + args.test) - 1.0) > 1e-6:
        raise SystemExit("train+val+test must sum to 1.0")

    index_path = Path(args.index)
    rows = list(_read_csv(index_path))
    if not rows:
        raise SystemExit(f"Empty index: {index_path}")

    rng = random.Random(int(args.seed))

    groups: dict[str, list[int]] = defaultdict(list)
    if args.stratify == "label":
        for i, r in enumerate(rows):
            groups[str(r.get("label") or "unknown")].append(i)
    else:
        groups["__all__"] = list(range(len(rows)))

    splits = {}
    for _g, idxs in groups.items():
        rng.shuffle(idxs)
        n = len(idxs)
        n_train = int(round(n * float(args.train)))
        n_val = int(round(n * float(args.val)))
        # ensure totals don't exceed n
        n_train = min(n_train, n)
        n_val = min(n_val, n - n_train)

        for j in idxs[:n_train]:
            splits[j] = "train"
        for j in idxs[n_train: n_train + n_val]:
            splits[j] = "val"
        for j in idxs[n_train + n_val:]:
            splits[j] = "test"

    for i, r in enumerate(rows):
        r["split"] = splits.get(i, "train")

    _write_csv(index_path, rows)
    counts = {"train": 0, "val": 0, "test": 0}
    for r in rows:
        counts[str(r.get("split"))] = counts.get(str(r.get("split")), 0) + 1
    print(f"Wrote {index_path} with splits: {counts}")
    return 0


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        return [dict(row) for row in r]


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        return
    fields = list(rows[0].keys())
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow(row)
    tmp.replace(path)


if __name__ == "__main__":
    raise SystemExit(main())
