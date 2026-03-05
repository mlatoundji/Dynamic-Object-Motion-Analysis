from __future__ import annotations

import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = object  # type: ignore[assignment]
    DataLoader = object  # type: ignore[assignment]

from .data import FeatureConfig, GestureDataset, NormStats, SampleRow, collate_padded, compute_norm_stats
from .metrics import classification_report_dict, compute_basic_metrics, confusion_matrix_counts
from .model import CNNLSTMClassifier, ModelConfig, export_onnx
from .plots import plot_confusion_matrix, plot_training_curves
from .utils import dataclass_to_json, save_json, set_seed


@dataclass(frozen=True)
class TrainConfig:
    manifest_csv: str = "data/processed/manifest.csv"
    out_dir: str = "runs"
    run_name: str = ""
    seed: int = 0
    device: str = "auto"
    dt_ms: float = 33.333

    # feature config
    use_landmarks: bool = True
    include_optflow: bool = True
    include_pose: bool = True

    # training
    batch_size: int = 32
    epochs: int = 20
    lr: float = 3e-4
    weight_decay: float = 1e-2
    num_workers: int = 0
    class_weight: bool = True

    # model
    conv_channels: int = 128
    conv_layers: int = 2
    conv_kernel: int = 5
    lstm_hidden: int = 256
    lstm_layers: int = 1
    bidirectional: bool = True
    dropout: float = 0.2


def _split_rows(rows: list[SampleRow]) -> tuple[list[SampleRow], list[SampleRow], list[SampleRow]]:
    tr = [r for r in rows if r.split == "train"]
    va = [r for r in rows if r.split == "val"]
    te = [r for r in rows if r.split == "test"]
    return tr, va, te


def _device_from_cfg(device: str) -> "torch.device":
    if torch is None:  # pragma: no cover
        raise RuntimeError("PyTorch required")
    d = (device or "auto").strip().lower()
    if d in {"", "auto"}:
        d = "cuda" if torch.cuda.is_available() else "cpu"
    if d == "cuda" and not torch.cuda.is_available():
        d = "cpu"
    return torch.device(d)


def _class_weights(labels_idx: list[int], *, num_classes: int) -> "torch.Tensor":
    if torch is None:  # pragma: no cover
        raise RuntimeError("PyTorch required")
    counts = np.zeros((int(num_classes),), dtype=np.float64)
    for y in labels_idx:
        if 0 <= int(y) < int(num_classes):
            counts[int(y)] += 1.0
    counts = np.maximum(counts, 1.0)
    inv = 1.0 / counts
    w = inv / float(np.mean(inv))
    return torch.tensor(w.astype(np.float32))


def _predict(model: CNNLSTMClassifier, loader: DataLoader, *, device: "torch.device") -> tuple[np.ndarray, np.ndarray]:
    if torch is None:  # pragma: no cover
        raise RuntimeError("PyTorch required")
    model.eval()
    ys: list[int] = []
    ps: list[int] = []
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            lengths = batch["lengths"].to(device)
            y = batch["y"].to(device)
            logits = model(x, lengths)
            pred = torch.argmax(logits, dim=1)
            ys.extend(y.detach().cpu().numpy().astype(int).tolist())
            ps.extend(pred.detach().cpu().numpy().astype(int).tolist())
    return np.asarray(ys, dtype=np.int64), np.asarray(ps, dtype=np.int64)


def train_run(cfg: TrainConfig, *, rows: list[SampleRow], label_to_idx: dict[str, int]) -> Path:
    if torch is None:  # pragma: no cover
        raise RuntimeError("PyTorch required (install optional extra 'train' or 'raft').")

    set_seed(int(cfg.seed))
    device = _device_from_cfg(cfg.device)

    feat_cfg = FeatureConfig(
        use_landmarks=bool(cfg.use_landmarks),
        include_optflow=bool(cfg.include_optflow),
        include_pose=bool(cfg.include_pose),
        angle_as_sincos=True,
    )

    train_rows, val_rows, test_rows = _split_rows(rows)
    norm = compute_norm_stats(train_rows, feat_cfg=feat_cfg, label_to_idx=label_to_idx, max_samples=0)

    ds_tr = GestureDataset(train_rows, label_to_idx=label_to_idx, feat_cfg=feat_cfg, norm=norm)
    ds_va = GestureDataset(val_rows, label_to_idx=label_to_idx, feat_cfg=feat_cfg, norm=norm)

    dl_tr = DataLoader(
        ds_tr,
        batch_size=int(cfg.batch_size),
        shuffle=True,
        num_workers=int(cfg.num_workers),
        collate_fn=collate_padded,
        drop_last=False,
    )
    dl_va = DataLoader(
        ds_va,
        batch_size=int(cfg.batch_size),
        shuffle=False,
        num_workers=int(cfg.num_workers),
        collate_fn=collate_padded,
        drop_last=False,
    )

    # Infer feature dimension from one batch.
    b0 = next(iter(dl_tr))
    in_features = int(b0["x"].shape[2])
    num_classes = int(len(label_to_idx))

    mcfg = ModelConfig(
        in_features=in_features,
        num_classes=num_classes,
        conv_channels=int(cfg.conv_channels),
        conv_layers=int(cfg.conv_layers),
        conv_kernel=int(cfg.conv_kernel),
        conv_dropout=float(cfg.dropout),
        lstm_hidden=int(cfg.lstm_hidden),
        lstm_layers=int(cfg.lstm_layers),
        bidirectional=bool(cfg.bidirectional),
        lstm_dropout=float(cfg.dropout),
        head_dropout=float(cfg.dropout),
    )
    model = CNNLSTMClassifier(mcfg).to(device)

    if bool(cfg.class_weight):
        labels_idx = [label_to_idx[r.label] for r in train_rows if r.label in label_to_idx]
        w = _class_weights(labels_idx, num_classes=num_classes).to(device)
    else:
        w = None

    loss_fn = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.lr), weight_decay=float(cfg.weight_decay))

    # Output structure
    run_name = cfg.run_name.strip() or time.strftime("classify_%Y%m%d-%H%M%S")
    run_dir = Path(cfg.out_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    dataclass_to_json(run_dir / "train_config.json", cfg)
    dataclass_to_json(run_dir / "model_config.json", mcfg)
    norm.to_npz(run_dir / "norm.npz")
    save_json(run_dir / "label_map.json", {"label_to_idx": label_to_idx, "idx_to_label": {str(v): k for k, v in label_to_idx.items()}})

    history: dict[str, list[float]] = {"train_loss": [], "val_loss": [], "val_accuracy": [], "val_macro_f1": []}
    best_macro_f1 = -1.0
    best_path = run_dir / "checkpoints" / "best.pt"

    for epoch in range(int(cfg.epochs)):
        model.train()
        losses = []
        for batch in dl_tr:
            x = batch["x"].to(device)
            lengths = batch["lengths"].to(device)
            y = batch["y"].to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(x, lengths)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu().item()))

        train_loss = float(np.mean(losses)) if losses else 0.0

        # Val loss + metrics
        model.eval()
        vlosses = []
        ys_all: list[int] = []
        ps_all: list[int] = []
        with torch.no_grad():
            for batch in dl_va:
                x = batch["x"].to(device)
                lengths = batch["lengths"].to(device)
                y = batch["y"].to(device)
                logits = model(x, lengths)
                vlosses.append(float(loss_fn(logits, y).detach().cpu().item()))
                pred = torch.argmax(logits, dim=1)
                ys_all.extend(y.detach().cpu().numpy().astype(int).tolist())
                ps_all.extend(pred.detach().cpu().numpy().astype(int).tolist())

        y_true = np.asarray(ys_all, dtype=np.int64)
        y_pred = np.asarray(ps_all, dtype=np.int64)
        metrics = compute_basic_metrics(y_true, y_pred, num_classes=num_classes)
        val_loss = float(np.mean(vlosses)) if vlosses else 0.0

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(float(metrics.accuracy))
        history["val_macro_f1"].append(float(metrics.macro_f1))

        # Best checkpoint
        if metrics.macro_f1 > best_macro_f1:
            best_macro_f1 = float(metrics.macro_f1)
            torch.save(
                {"model": model.state_dict(), "model_config": asdict(mcfg), "label_to_idx": label_to_idx},
                best_path,
            )

    plot_training_curves(history, out_path=run_dir / "training_curves.png")
    save_json(run_dir / "history.json", history)

    # Final test eval (if test split exists)
    if test_rows:
        ds_te = GestureDataset(test_rows, label_to_idx=label_to_idx, feat_cfg=feat_cfg, norm=norm)
        dl_te = DataLoader(ds_te, batch_size=int(cfg.batch_size), shuffle=False, num_workers=int(cfg.num_workers), collate_fn=collate_padded)

        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        y_true, y_pred = _predict(model, dl_te, device=device)

        labels_sorted = [lab for lab, _ in sorted(label_to_idx.items(), key=lambda kv: kv[1])]
        rep = classification_report_dict(y_true, y_pred, labels=labels_sorted)
        cm = confusion_matrix_counts(y_true, y_pred, num_classes=num_classes)
        plot_confusion_matrix(cm, labels=labels_sorted, out_path=run_dir / "confusion_matrix.png", normalize=True)

        save_json(run_dir / "test_metrics.json", {"basic": asdict(compute_basic_metrics(y_true, y_pred, num_classes=num_classes)), "report": rep})
        save_json(run_dir / "confusion_matrix_counts.json", {"labels": labels_sorted, "cm": cm.tolist()})

    # ONNX export for deployment/poc
    try:
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        export_onnx(model, out_path=run_dir / "model.onnx", opset=17, max_len=256)
    except Exception:
        # ONNX export is best-effort (keeps training usable if export fails on some envs)
        pass

    return run_dir


def evaluate_checkpoint(
    ckpt_path: Path,
    *,
    manifest_csv: Path,
    split: str = "test",
    use_landmarks: bool = True,
    include_pose: bool = True,
    include_optflow: bool = True,
    batch_size: int = 64,
    device: str = "auto",
) -> dict[str, Any]:
    if torch is None:  # pragma: no cover
        raise RuntimeError("PyTorch required")

    run_dir = ckpt_path.parent.parent if ckpt_path.name == "best.pt" else ckpt_path.parent
    norm = NormStats.from_npz(run_dir / "norm.npz")
    label_to_idx = (run_dir / "label_map.json")
    label_map = None
    if label_to_idx.exists():
        import json

        label_map = json.loads(label_to_idx.read_text(encoding="utf-8")).get("label_to_idx", {})
    if not isinstance(label_map, dict) or not label_map:
        raise ValueError("Missing label_map.json next to checkpoint")
    label_map = {str(k): int(v) for k, v in label_map.items()}

    from .data import read_manifest_rows

    rows = read_manifest_rows(manifest_csv)
    rows = [r for r in rows if r.split == split]

    feat_cfg = FeatureConfig(use_landmarks=use_landmarks, include_pose=include_pose, include_optflow=include_optflow, angle_as_sincos=True)
    ds = GestureDataset(rows, label_to_idx=label_map, feat_cfg=feat_cfg, norm=norm)
    dl = DataLoader(ds, batch_size=int(batch_size), shuffle=False, collate_fn=collate_padded)

    device_t = _device_from_cfg(device)

    ckpt = torch.load(ckpt_path, map_location=device_t)
    mcfg = ModelConfig(**ckpt["model_config"])
    model = CNNLSTMClassifier(mcfg).to(device_t)
    model.load_state_dict(ckpt["model"])

    y_true, y_pred = _predict(model, dl, device=device_t)
    labels_sorted = [lab for lab, _ in sorted(label_map.items(), key=lambda kv: kv[1])]
    rep = classification_report_dict(y_true, y_pred, labels=labels_sorted)
    cm = confusion_matrix_counts(y_true, y_pred, num_classes=len(labels_sorted))
    basic = compute_basic_metrics(y_true, y_pred, num_classes=len(labels_sorted))
    return {"basic": asdict(basic), "report": rep, "labels": labels_sorted, "cm": cm.tolist()}

