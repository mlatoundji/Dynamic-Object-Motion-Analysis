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

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    from sklearn.metrics import (
        accuracy_score,
        precision_recall_fscore_support,
    )
except ImportError:
    accuracy_score = None
    precision_recall_fscore_support = None

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
    from doma.models.temporal_transformer import TemporalTransformer

    register_model(
        "temporal_transformer",
        TemporalTransformer,
        default_kwargs={
            "d_model": 128,
            "nhead": 4,
            "num_encoder_layers": 3,
            "dim_feedforward": 256,
            "dropout": 0.1,
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


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    desc: str = "Eval",
) -> tuple[list[int], list[int], float]:
    """
    Run model on loader and return all labels, all predictions, and mean loss.

    Model must accept batch dict and return logits (B, num_classes).
    """
    model.eval()
    all_labels: list[int] = []
    all_preds: list[int] = []
    total_loss = 0.0
    n = 0
    criterion = nn.CrossEntropyLoss(reduction="sum")

    with torch.no_grad():
        for batch in tqdm(loader, desc=desc, leave=False):
            pose = batch["pose"].to(device)
            optflow = batch["optflow"].to(device)
            length = batch["length"].to(device)
            label = batch["label"].to(device)

            logits = model(pose=pose, optflow=optflow, length=length)
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

    Returns:
        Dict with training history and best metrics.
    """
    from doma.dataset import GestureDataset, create_dataloaders, collate_gesture_batch

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

    # Single run ID for this training run (model, metrics, and log share it)
    run_dt = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{model_name}_{run_dt}"
    ckpt_path = output_dir / f"{run_id}.pt"
    metrics_path = output_dir / f"{run_id}.json"
    log_path = output_dir / f"{run_id}.log"

    run_handlers = _setup_run_logging(log_path)
    log = _train_logger

    log.info("Run ID: %s", run_id)
    log.info("Log file: %s", log_path)
    log.info("Device: %s", device)
    log.info(
        "output_dir=%s  model=%s  split_mode=%s  batch_size=%s  epochs=%s  lr=%s",
        output_dir, model_name, split_mode, batch_size, epochs, lr,
    )
    log.info("")

    # Data
    loaders = create_dataloaders(
        manifest_path=manifest_path,
        root_dir=root_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        max_len=max_len,
        split_mode=split_mode,
        pin_memory=(device.type == "cuda"),
    )

    if split_mode == "train_val_test":
        train_loader, val_loader, test_loader = loaders
        eval_loader = val_loader  # train on train, evaluate on val
    else:
        train_loader, test_loader = loaders
        eval_loader = test_loader  # train on train+val, evaluate on test

    # Num classes from dataset
    ds = train_loader.dataset
    num_classes = ds.num_classes

    # Model
    builder, defaults = get_model_builder(model_name)
    kwargs = {**defaults, **(model_kwargs or {})}
    model = builder(num_classes=num_classes, **kwargs).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

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
            pose = batch["pose"].to(device)
            optflow = batch["optflow"].to(device)
            length = batch["length"].to(device)
            label = batch["label"].to(device)

            optimizer.zero_grad()
            logits = model(pose=pose, optflow=optflow, length=length)
            loss = criterion(logits, label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item() * label.size(0)
            train_n += label.size(0)
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        train_loss /= train_n if train_n else 1.0

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
            torch.save(ckpt, ckpt_path)
            log.info("  -> saved %s (best %s=%.4f)", ckpt_path.name, save_best_by, value)

    # Write metrics
    metrics_out = {
        "best_epoch": best_epoch,
        "best_metric_name": save_best_by,
        "best_metric_value": best_metric,
        "best_metrics": best_metrics if best_metrics is not None else {},
        "history": history,
    }
    with open(metrics_path, "w") as f:
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


def main() -> None:
    import argparse

    # Enable MPS->CPU fallback for unsupported ops (e.g. transformer) before any torch ops run.
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

    parser = argparse.ArgumentParser(
        description="Train a gesture recognition model (scalable for multiple models)."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/processed/manifest.csv"),
        help="Path to manifest.csv",
    )
    parser.add_argument(
        "--root-dir",
        type=Path,
        default=Path("."),
        help="Root directory for data paths",
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
    parser.add_argument("--lr", type=float, default=5e-4)
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
    )
