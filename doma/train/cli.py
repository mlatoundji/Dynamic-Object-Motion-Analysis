from __future__ import annotations

import argparse
from pathlib import Path

from .data import build_label_map, read_manifest_rows
from .report import generate_report
from .runner import TrainConfig, evaluate_checkpoint, train_run
from .utils import save_json


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Train/evaluate CNN-LSTM gesture classifier (IPN Hand)")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train", help="Train a model from a manifest.csv")
    p_train.add_argument("--manifest", default="data/processed/manifest.csv")
    p_train.add_argument("--out", default="runs")
    p_train.add_argument("--name", default="", help="Run name (default: timestamped)")
    p_train.add_argument("--seed", type=int, default=0)
    p_train.add_argument("--device", default="auto")
    p_train.add_argument("--dt-ms", type=float, default=33.333, help="Regular sampling interval used by live PoC")
    p_train.add_argument("--batch", type=int, default=32)
    p_train.add_argument("--epochs", type=int, default=20)
    p_train.add_argument("--lr", type=float, default=3e-4)
    p_train.add_argument("--weight-decay", type=float, default=1e-2)
    p_train.add_argument("--num-workers", type=int, default=0)
    p_train.add_argument("--no-class-weight", action="store_true")

    p_train.add_argument("--no-landmarks", action="store_true")
    p_train.add_argument("--no-optflow", action="store_true")
    p_train.add_argument("--no-pose", action="store_true")

    p_train.add_argument("--conv-ch", type=int, default=128)
    p_train.add_argument("--conv-layers", type=int, default=2)
    p_train.add_argument("--conv-kernel", type=int, default=5)
    p_train.add_argument("--lstm-hidden", type=int, default=256)
    p_train.add_argument("--lstm-layers", type=int, default=1)
    p_train.add_argument("--no-bidir", action="store_true")
    p_train.add_argument("--dropout", type=float, default=0.2)

    p_eval = sub.add_parser("eval", help="Evaluate a checkpoint on a split")
    p_eval.add_argument("--ckpt", required=True, help="Path to checkpoint (best.pt)")
    p_eval.add_argument("--manifest", default="data/processed/manifest.csv")
    p_eval.add_argument("--split", default="test", choices=["train", "val", "test"])
    p_eval.add_argument("--device", default="auto")
    p_eval.add_argument("--batch", type=int, default=64)
    p_eval.add_argument("--no-landmarks", action="store_true")
    p_eval.add_argument("--no-optflow", action="store_true")
    p_eval.add_argument("--no-pose", action="store_true")
    p_eval.add_argument("--out", default="", help="Write JSON metrics to this path (optional)")

    p_report = sub.add_parser("report", help="Generate a Markdown report from a run directory")
    p_report.add_argument("--run", required=True, help="Run directory (e.g. runs/classify_...)")
    p_report.add_argument("--out", default="docs/REPORT_CNN_LSTM.md", help="Markdown output path")

    args = p.parse_args(argv)

    if args.cmd == "train":
        manifest = Path(args.manifest)
        rows = read_manifest_rows(manifest)
        label_to_idx = build_label_map(rows)
        cfg = TrainConfig(
            manifest_csv=str(manifest.as_posix()),
            out_dir=str(Path(args.out).as_posix()),
            run_name=str(args.name),
            seed=int(args.seed),
            device=str(args.device),
            dt_ms=float(args.dt_ms),
            use_landmarks=not bool(args.no_landmarks),
            include_optflow=not bool(args.no_optflow),
            include_pose=not bool(args.no_pose),
            batch_size=int(args.batch),
            epochs=int(args.epochs),
            lr=float(args.lr),
            weight_decay=float(args.weight_decay),
            num_workers=int(args.num_workers),
            class_weight=not bool(args.no_class_weight),
            conv_channels=int(args.conv_ch),
            conv_layers=int(args.conv_layers),
            conv_kernel=int(args.conv_kernel),
            lstm_hidden=int(args.lstm_hidden),
            lstm_layers=int(args.lstm_layers),
            bidirectional=not bool(args.no_bidir),
            dropout=float(args.dropout),
        )
        run_dir = train_run(cfg, rows=rows, label_to_idx=label_to_idx)
        print(str(run_dir))
        return 0

    if args.cmd == "eval":
        out = evaluate_checkpoint(
            Path(args.ckpt),
            manifest_csv=Path(args.manifest),
            split=str(args.split),
            use_landmarks=not bool(args.no_landmarks),
            include_pose=not bool(args.no_pose),
            include_optflow=not bool(args.no_optflow),
            batch_size=int(args.batch),
            device=str(args.device),
        )
        if args.out:
            save_json(Path(args.out), out)
        else:
            print(out["basic"])
        return 0

    if args.cmd == "report":
        generate_report(Path(args.run), out_path=Path(args.out))
        print(str(Path(args.out)))
        return 0

    raise SystemExit("Unknown command")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

