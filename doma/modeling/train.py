"""
Generic training script for gesture recognition models.

Model-agnostic: register models by name and run training with progress and
evaluation (accuracy, precision, recall, F1). Scalable for future models.
"""

from pathlib import Path
from datetime import datetime
from typing import Any, Callable, Optional
import json
import logging
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

try:
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        precision_recall_fscore_support,
    )
except ImportError:
    accuracy_score = None
    precision_recall_fscore_support = None
    classification_report = None

# Model registry: name -> (builder_fn, default_kwargs)
# builder_fn(num_classes, **kwargs) -> nn.Module
_MODEL_REGISTRY: dict[str, tuple[Callable[..., nn.Module], dict[str, Any]]] = {}


def register_model(
    name: str,
    builder: Callable[..., nn.Module],
    default_kwargs: Optional[dict[str, Any]] = None,
) -> None:
    """Register a model for training. builder(num_classes, **kwargs) -> nn.Module."""
    _MODEL_REGISTRY[name] = (builder, default_kwargs or {})


def get_model_builder(name: str) -> tuple[Callable[..., nn.Module], dict[str, Any]]:
    """Get (builder_fn, default_kwargs) for a registered model name."""
    if name not in _MODEL_REGISTRY:
        raise KeyError(
            f"Unknown model {name!r}. Registered: {list(_MODEL_REGISTRY.keys())}"
        )
    return _MODEL_REGISTRY[name]


# Register built-in models
def _register_builtin_models() -> None:
    from doma.models.temporal_transformer import TemporalTransformer, ModelConfig as TTConfig
    from doma.models.temporal_vit import TemporalViT, ModelConfig as ViTConfig
    from doma.models.cnn_lstm import CNNLSTM, ModelConfig as CNNLSTMConfig

    def _build_cnn_lstm(num_classes: int, in_features: int, **kwargs: Any) -> nn.Module:
        cfg = CNNLSTMConfig(num_classes=num_classes, in_features=in_features, **kwargs)
        return CNNLSTM(cfg)

    def _build_temporal_transformer(num_classes: int, in_features: int, **kwargs: Any) -> nn.Module:
        cfg = TTConfig(num_classes=num_classes, in_features=in_features, **kwargs)
        return TemporalTransformer(cfg)

    def _build_temporal_vit(num_classes: int, **kwargs: Any) -> nn.Module:
        cfg = ViTConfig(num_classes=num_classes, **kwargs)
        return TemporalViT(cfg)

    register_model(
        "temporal_transformer",
        _build_temporal_transformer,
        default_kwargs={
            "d_model": 192,
            "nhead": 4,
            "num_encoder_layers": 4,
            "dim_feedforward": 768,
            "dropout": 0.1,
        },
    )
    register_model(
        "temporal_vit",
        _build_temporal_vit,
        default_kwargs={
            "num_frames": 16,
            "img_size": 224,
            "patch_size": 16,
            "embed_dim": 384,
            "depth": 12,
            "num_heads": 6,
            "temporal_depth": 2,
            "drop_rate": 0.0,
            "pretrained": True,
        },
    )
    register_model(
        "cnn_lstm",
        _build_cnn_lstm,
        default_kwargs={
            "conv_channels": 128,
            "conv_layers": 2,
            "conv_kernel": 5,
            "conv_dropout": 0.1,
            "lstm_hidden": 256,
            "lstm_layers": 1,
            "bidirectional": True,
            "lstm_dropout": 0.1,
            "head_dropout": 0.2,
        },
    )


_register_builtin_models()

# Logger for training (configured per run in run_train)
_train_logger = logging.getLogger("doma.modeling.train")


def _setup_run_logging(log_path: Path) -> list[logging.Handler]:
    """Configure _train_logger to emit to console and log file. Returns handlers to remove later."""
    _train_logger.setLevel(logging.INFO)
    for h in _train_logger.handlers[:]:
        _train_logger.removeHandler(h)
    formatter = logging.Formatter("%(message)s")
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    _train_logger.addHandler(stream)
    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    _train_logger.addHandler(file_handler)
    return _train_logger.handlers[:]


def _clear_run_logging(handlers: list[logging.Handler]) -> None:
    for h in handlers:
        _train_logger.removeHandler(h)
        if getattr(h, "close", None):
            h.close()


def compute_metrics(
    y_true: list[int],
    y_pred: list[int],
    num_classes: Optional[int] = None,
    labels: Optional[list[int]] = None,
) -> dict[str, float]:
    """
    Compute accuracy, precision, recall, F1 (macro and weighted).

    Returns dict with: accuracy, precision_macro, recall_macro, f1_macro,
    precision_weighted, recall_weighted, f1_weighted.
    """
    if accuracy_score is None or precision_recall_fscore_support is None:
        raise ImportError("sklearn is required for metrics. Install with: pip install scikit-learn")

    y_true = [int(x) for x in y_true]
    y_pred = [int(x) for x in y_pred]

    acc = accuracy_score(y_true, y_pred)

    if labels is None and num_classes is not None:
        labels = list(range(num_classes))
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0, labels=labels
    )
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0, labels=labels
    )

    return {
        "accuracy": float(acc),
        "precision_macro": float(p_macro),
        "recall_macro": float(r_macro),
        "f1_macro": float(f1_macro),
        "precision_weighted": float(p_weighted),
        "recall_weighted": float(r_weighted),
        "f1_weighted": float(f1_weighted),
    }


def _batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    """Move batch tensors to device. Non-tensors (e.g. sample_id list) unchanged."""
    return {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in batch.items()
    }


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    desc: str = "Eval",
) -> tuple[list[int], list[int], float]:
    """
    Run model on loader and return all labels, all predictions, and mean loss.

    Model must accept model(batch=batch) and return logits (B, num_classes).
    Batch can be manifest-style (pose, optflow, length, label) or IPN-style (frames, length, label).
    """
    model.eval()
    all_labels: list[int] = []
    all_preds: list[int] = []
    total_loss = 0.0
    n = 0
    criterion = nn.CrossEntropyLoss(reduction="sum")

    with torch.no_grad():
        for batch in tqdm(loader, desc=desc, leave=False):
            batch_dev = _batch_to_device(batch, device)
            label = batch_dev.get("label", batch_dev.get("y"))

            logits = model(batch=batch_dev)
            loss = criterion(logits, label)

            total_loss += loss.item()
            n += label.size(0)
            pred = logits.argmax(dim=1).cpu().tolist()
            all_preds.extend(pred)
            all_labels.extend(label.cpu().tolist())

    mean_loss = total_loss / n if n else 0.0
    return all_labels, all_preds, mean_loss


def run_train(
    manifest_path: str | Path,
    root_dir: str | Path,
    model_name: str = "temporal_transformer",
    *,
    output_dir: str | Path = "models",
    split_mode: str = "train_val_test",
    batch_size: int = 32,
    epochs: int = 20,
    lr: float = 5e-4,
    max_len: Optional[int] = None,
    num_workers: int = 0,
    model_kwargs: Optional[dict[str, Any]] = None,
    save_best_by: str = "accuracy",
    device: Optional[torch.device] = None,
    weight_decay: float = 0.01,
    seed: Optional[int] = None,
) -> dict[str, Any]:
    """
    Train a gesture model and evaluate on validation.

    Args:
        manifest_path: Path to data/processed/manifest.csv.
        root_dir: Root directory for data paths.
        model_name: Name of a registered model (e.g. "temporal_transformer").
        output_dir: Directory to save checkpoints.
        split_mode: "train_val_test" = train on train, evaluate on val; "train_test" = train on train+val, evaluate on test.
        batch_size: Batch size.
        epochs: Number of epochs.
        lr: Learning rate.
        max_len: Optional max sequence length (truncation).
        num_workers: DataLoader workers.
        model_kwargs: Extra kwargs passed to the model builder.
        save_best_by: Metric to maximize for best checkpoint ("accuracy", "f1_macro", "f1_weighted").
        device: Device (default: cuda if available, else mps on Apple Silicon, else cpu).
        weight_decay: AdamW weight decay for regularization.
        seed: Optional random seed for reproducibility (model init + dataloader shuffle).

    Returns:
        Dict with training history and best metrics.
    """
    from doma.dataloaders import build_dataloaders

    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            device = torch.device("mps")
            # TransformerEncoder uses nested tensor ops not implemented on MPS; force CPU fallback.
            os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        else:
            device = torch.device("cpu")

    manifest_path = Path(manifest_path)
    root_dir = Path(root_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    dataloader_generator = torch.Generator().manual_seed(seed) if seed is not None else None

    # Single run ID for this training run (model, metrics, and log share it)
    run_dt = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{model_name}_{run_dt}"
    run_dir = output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = run_dir / "best.pt"
    metrics_path = run_dir / "metrics.json"
    log_path = run_dir / "train_log.log"

    run_handlers = _setup_run_logging(log_path)
    log = _train_logger

    log.info("Run ID: %s", run_id)
    log.info("Run dir: %s", run_dir)
    log.info("Log file: %s", log_path)
    log.info("Device: %s", device)
    log.info(
        "output_dir=%s  model=%s  split_mode=%s  batch_size=%s  epochs=%s  lr=%s  weight_decay=%s  seed=%s",
        output_dir, model_name, split_mode, batch_size, epochs, lr, weight_decay, seed,
    )
    log.info("")

    # Data (unified features for all manifest-based models)
    from doma.dataloaders import build_dataloaders

    loaders = build_dataloaders(
        manifest_path=manifest_path,
        root_dir=root_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        max_len=max_len,
        split_mode=split_mode,
        pin_memory=(device.type == "cuda"),
        generator=dataloader_generator,
    )

    if split_mode == "train_val_test":
        train_loader, val_loader, test_loader = loaders
        eval_loader = val_loader  # train on train, evaluate on val
    else:
        train_loader, test_loader = loaders
        eval_loader = test_loader  # train on train+val, evaluate on test

    # Num classes and feature dim from dataset / first batch
    ds = train_loader.dataset
    num_classes = ds.num_classes
    b0 = next(iter(train_loader))
    in_features = int(b0["x"].shape[2])

    # Model (all manifest models use unified batch with x, lengths)
    builder, defaults = get_model_builder(model_name)
    kwargs = {**defaults, **(model_kwargs or {})}
    if model_name == "cnn_lstm":
        model = builder(num_classes=num_classes, in_features=in_features, **kwargs).to(device)
    elif model_name == "temporal_transformer":
        model = builder(num_classes=num_classes, in_features=in_features, **kwargs).to(device)
    else:
        model = builder(num_classes=num_classes, **kwargs).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    # Build model_config for saving (reproducibility); all models now have .cfg
    from dataclasses import asdict
    model_config = asdict(model.cfg)
    with open(run_dir / "model_config.json", "w", encoding="utf-8") as f:
        json.dump(model_config, f, indent=2)

    train_config = {
        "manifest_path": str(manifest_path),
        "root_dir": str(root_dir),
        "model_name": model_name,
        "split_mode": split_mode,
        "batch_size": batch_size,
        "epochs": epochs,
        "lr": lr,
        "max_len": max_len,
        "num_workers": num_workers,
        "save_best_by": save_best_by,
        "weight_decay": weight_decay,
        "seed": seed,
        "dt_ms": 33.333,
    }
    with open(run_dir / "train_config.json", "w", encoding="utf-8") as f:
        json.dump(train_config, f, indent=2)

    # For live-classifier: save norm, label_map to run_dir
    if model_name in ("cnn_lstm", "temporal_transformer"):
        norm = getattr(ds, "_norm", None)
        label_to_idx = getattr(ds, "_label_to_idx", None)
        if norm is not None:
            norm.to_npz(run_dir / "norm.npz")
        if label_to_idx is not None:
            with open(run_dir / "label_map.json", "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "label_to_idx": label_to_idx,
                        "idx_to_label": {str(v): k for k, v in label_to_idx.items()},
                    },
                    f,
                    indent=2,
                )

    history: list[dict[str, float]] = []
    best_metric = -1.0
    best_epoch = -1
    best_metrics: Optional[dict[str, float]] = None

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_n = 0

        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{epochs}",
            leave=True,
        )
        for batch in pbar:
            batch_dev = _batch_to_device(batch, device)
            label = batch_dev.get("label", batch_dev.get("y"))

            optimizer.zero_grad()
            logits = model(batch=batch_dev)
            loss = criterion(logits, label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item() * label.size(0)
            train_n += label.size(0)
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        train_loss /= train_n if train_n else 1.0
        scheduler.step()

        # Evaluation (on val for train_val_test, on test for train_test)
        eval_labels, eval_preds, eval_loss = evaluate(
            model, eval_loader, device,
            desc="Val" if split_mode == "train_val_test" else "Test",
        )
        metrics = compute_metrics(
            eval_labels, eval_preds, num_classes=num_classes
        )
        metrics["train_loss"] = train_loss
        metrics["val_loss"] = eval_loss

        history.append(metrics)

        # Log
        log.info(
            "  train_loss=%.4f  val_loss=%.4f  accuracy=%.4f  precision(macro)=%.4f  "
            "recall(macro)=%.4f  f1(macro)=%.4f  f1(weighted)=%.4f",
            metrics["train_loss"], metrics["val_loss"], metrics["accuracy"],
            metrics["precision_macro"], metrics["recall_macro"], metrics["f1_macro"],
            metrics["f1_weighted"],
        )

        # Best checkpoint
        key = save_best_by
        if key not in metrics:
            key = "accuracy"
        value = metrics[key]
        if value > best_metric:
            best_metric = value
            best_epoch = epoch + 1
            best_metrics = dict(metrics)
            ckpt = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "metrics": metrics,
                "num_classes": num_classes,
                "model_name": model_name,
            }
            # Live-classifier expects "model" and "model_config"
            ckpt["model"] = ckpt["model_state_dict"]
            ckpt["model_config"] = model_config
            torch.save(ckpt, ckpt_path)
            log.info("  -> saved %s (best %s=%.4f)", ckpt_path.name, save_best_by, value)

    # Per-class metrics for best model (manifest-based: re-run eval with best ckpt)
    best_per_class: dict[str, Any] = {}
    if ckpt_path.exists() and best_metrics is not None and model_name in ("cnn_lstm", "temporal_transformer"):
        label_to_idx_best = getattr(ds, "_label_to_idx", None)
        if label_to_idx_best is not None and classification_report is not None:
            ckpt = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ckpt.get("model", ckpt.get("model_state_dict")), strict=True)
            eval_labels_best, eval_preds_best, _ = evaluate(
                model, eval_loader, device,
                desc="Eval (best)",
            )
            labels_sorted = [lab for lab, _ in sorted(label_to_idx_best.items(), key=lambda kv: kv[1])]
            report = classification_report(
                eval_labels_best,
                eval_preds_best,
                labels=list(range(num_classes)),
                target_names=labels_sorted,
                output_dict=True,
                zero_division=0,
            )
            best_per_class = {k: report[k] for k in labels_sorted if k in report}

    # Write metrics
    metrics_out = {
        "best_epoch": best_epoch,
        "best_metric_name": save_best_by,
        "best_metric_value": best_metric,
        "best_metrics": best_metrics if best_metrics is not None else {},
        "best_per_class": best_per_class,
        "history": history,
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_out, f, indent=2)
    log.info("Metrics written to %s", metrics_path)
    log.info("Best epoch: %s (%s=%.4f)", best_epoch, save_best_by, best_metric)
    _clear_run_logging(run_handlers)
    return {
        "history": history,
        "best_epoch": best_epoch,
        "best_metric": best_metric,
        "best_metric_name": save_best_by,
        "best_metrics": best_metrics if best_metrics is not None else {},
    }


def run_train_ipn(
    annotation_path: str | Path,
    root_dir: str | Path,
    model_name: str = "temporal_vit",
    *,
    output_dir: str | Path = "models",
    batch_size: int = 32,
    epochs: int = 20,
    lr: float = 1e-4,
    max_len: Optional[int] = None,
    num_workers: int = 0,
    model_kwargs: Optional[dict[str, Any]] = None,
    save_best_by: str = "accuracy",
    device: Optional[torch.device] = None,
    frame_size: Optional[tuple[int, int]] = None,
    max_frames: Optional[int] = None,
    flow_dir: Optional[str | Path] = None,
    weight_decay: float = 0.05,
) -> dict[str, Any]:
    """
    Train a gesture model on the IPN Hand flow dataset (ipnall_flo.json + flow frames).

    Uses doma.dataloaders.flow_dataloader.build_dataloaders. Model must accept batch with "frames"
    (B, T, C, H, W) and return logits (B, num_classes).
    """
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            device = torch.device("mps")
            os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        else:
            device = torch.device("cpu")

    annotation_path = Path(annotation_path)
    root_dir = Path(root_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    flow_dir = Path(flow_dir) if flow_dir is not None else root_dir / "flow"

    from doma.dataloaders import build_flow_dataloaders

    train_loader, val_loader = build_flow_dataloaders(
        annotation_path=annotation_path,
        flow_dir=flow_dir,
        num_frames=max_frames or 8,
        frame_size=frame_size or (224, 224),
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    run_dt = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{model_name}_{run_dt}"
    ckpt_path = output_dir / f"{run_id}.pt"
    metrics_path = output_dir / f"{run_id}.json"
    log_path = output_dir / f"{run_id}.log"

    run_handlers = _setup_run_logging(log_path)
    log = _train_logger

    log.info("Run ID: %s (IPN Hand flow, train/val only)", run_id)
    log.info("Log file: %s", log_path)
    log.info("Device: %s", device)
    log.info(
        "annotation=%s  root_dir=%s  model=%s  batch_size=%s  epochs=%s  lr=%s",
        annotation_path, root_dir, model_name, batch_size, epochs, lr,
    )
    log.info("")

    eval_loader = val_loader

    if train_loader is None:
        raise RuntimeError("No training samples found. Check annotation_path and root_dir (flow/ dir).")

    num_classes = train_loader.dataset.num_classes

    builder, defaults = get_model_builder(model_name)
    kwargs = {**defaults, **(model_kwargs or {})}
    model = builder(num_classes=num_classes, **kwargs).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    history: list[dict[str, float]] = []
    best_metric = -1.0
    best_epoch = -1
    best_metrics: Optional[dict[str, float]] = None

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_n = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=True)
        for batch in pbar:
            batch_dev = _batch_to_device(batch, device)
            label = batch_dev.get("label", batch_dev.get("y"))

            optimizer.zero_grad()
            logits = model(batch=batch_dev)
            loss = criterion(logits, label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item() * label.size(0)
            train_n += label.size(0)
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        train_loss /= train_n if train_n else 1.0
        scheduler.step()

        eval_labels, eval_preds, eval_loss = evaluate(
            model, eval_loader, device,
            desc="Val",
        )

        metrics = compute_metrics(eval_labels, eval_preds, num_classes=num_classes)
        metrics["train_loss"] = train_loss
        metrics["val_loss"] = eval_loss
        history.append(metrics)

        log.info(
            "  train_loss=%.4f  val_loss=%.4f  accuracy=%.4f  precision(macro)=%.4f  "
            "recall(macro)=%.4f  f1(macro)=%.4f  f1(weighted)=%.4f",
            metrics["train_loss"], metrics["val_loss"], metrics["accuracy"],
            metrics["precision_macro"], metrics["recall_macro"], metrics["f1_macro"],
            metrics["f1_weighted"],
        )

        key = save_best_by
        if key not in metrics:
            key = "accuracy"
        value = metrics[key]
        if value > best_metric:
            best_metric = value
            best_epoch = epoch + 1
            best_metrics = dict(metrics)
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "metrics": metrics,
                "num_classes": num_classes,
                "model_name": model_name,
            }, ckpt_path)
            log.info("  -> saved %s (best %s=%.4f)", ckpt_path.name, save_best_by, value)

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump({
            "best_epoch": best_epoch,
            "best_metric_name": save_best_by,
            "best_metric_value": best_metric,
            "best_metrics": best_metrics if best_metrics is not None else {},
            "history": history,
        }, f, indent=2)
    log.info("Metrics written to %s", metrics_path)
    log.info("Best epoch: %s (%s=%.4f)", best_epoch, save_best_by, best_metric)
    _clear_run_logging(run_handlers)
    return {
        "history": history,
        "best_epoch": best_epoch,
        "best_metric": best_metric,
        "best_metric_name": save_best_by,
        "best_metrics": best_metrics if best_metrics is not None else {},
    }


def main() -> None:
    import argparse

    # Enable MPS->CPU fallback for unsupported ops (e.g. transformer) before any torch ops run.
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

    parser = argparse.ArgumentParser(
        description="Train a gesture recognition model (scalable for multiple models)."
    )
    parser.add_argument(
        "--dataset-type",
        type=str,
        default="manifest",
        choices=["manifest", "ipn_flow"],
        help="manifest: use manifest.csv + pose/optflow NPZ; ipn_flow: use ipnall_flo.json + flow/ frames",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/processed/manifest.csv"),
        help="Path to manifest.csv (used when --dataset-type manifest)",
    )
    parser.add_argument(
        "--annotation",
        type=Path,
        default=None,
        help="Path to ipnall_flo.json (required when --dataset-type ipn_flow)",
    )
    parser.add_argument(
        "--root-dir",
        type=Path,
        default=Path("."),
        help="Root directory for data (manifest paths or flow/ dir)",
    )
    parser.add_argument(
        "--flow-dir",
        type=Path,
        default=None,
        help="Flow frames directory (default: root_dir / flow). Used when --dataset-type ipn_flow",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="temporal_transformer",
        choices=list(_MODEL_REGISTRY.keys()),
        help="Model name",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models"),
        help="Directory to save checkpoints",
    )
    parser.add_argument(
        "--split-mode",
        type=str,
        default="train_test",
        choices=["train_val_test", "train_test"],
        help="train_val_test: train on train, eval on val; train_test: train on train+val, eval on test",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate. For ipn_flow/temporal_vit recommend 1e-4.")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="AdamW weight decay (manifest: 0.01; ipn_flow uses 0.05)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--max-len", type=int, default=None, help="Max sequence length (truncate)")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--save-best-by",
        type=str,
        default="accuracy",
        choices=["accuracy", "f1_macro", "f1_weighted"],
        help="Metric to maximize for best checkpoint",
    )
    args = parser.parse_args()

    if args.dataset_type == "ipn_flow":
        if args.annotation is None:
            parser.error("--annotation is required when --dataset-type ipn_flow")
        # Use 1e-4 for ViT when user did not override lr (parser default 5e-4)
        lr_ipn = 1e-4 if args.lr == 5e-4 else args.lr
        weight_decay_ipn = 0.05 if args.weight_decay == 0.01 else args.weight_decay
        run_train_ipn(
            annotation_path=args.annotation,
            root_dir=args.root_dir,
            model_name=args.model,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            epochs=args.epochs,
            lr=lr_ipn,
            max_len=args.max_len,
            num_workers=args.num_workers,
            save_best_by=args.save_best_by,
            flow_dir=args.flow_dir,
            weight_decay=weight_decay_ipn,
        )
    else:
        run_train(
            manifest_path=args.manifest,
            root_dir=args.root_dir,
            model_name=args.model,
            output_dir=args.output_dir,
            split_mode=args.split_mode,
            batch_size=args.batch_size,
            epochs=args.epochs,
            lr=args.lr,
            max_len=args.max_len,
            num_workers=args.num_workers,
            save_best_by=args.save_best_by,
            weight_decay=args.weight_decay,
            seed=args.seed,
        )
