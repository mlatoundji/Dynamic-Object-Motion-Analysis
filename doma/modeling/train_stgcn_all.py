"""
Generic training script for gesture recognition models.

Supports:
- Temporal Transformer
- STGCN
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
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
except ImportError:
    accuracy_score = None
    precision_recall_fscore_support = None


# =========================================================
# Model registry
# =========================================================

_MODEL_REGISTRY: dict[str, tuple[Callable[..., nn.Module], dict[str, Any]]] = {}


def register_model(
    name: str,
    builder: Callable[..., nn.Module],
    default_kwargs: Optional[dict[str, Any]] = None,
) -> None:
    _MODEL_REGISTRY[name] = (builder, default_kwargs or {})


def get_model_builder(name: str) -> tuple[Callable[..., nn.Module], dict[str, Any]]:
    if name not in _MODEL_REGISTRY:
        raise KeyError(
            f"Unknown model {name!r}. Registered: {list(_MODEL_REGISTRY.keys())}"
        )
    return _MODEL_REGISTRY[name]


# =========================================================
# Register models
# =========================================================

def _register_builtin_models():
    from doma.models.stgcn import STGCN
    def build_stgcn(num_classes: int, **kwargs):

        n = 26
        Lk = torch.eye(n).unsqueeze(0)

        model = STGCN(
            ks=1,
            kt=3,
            bs=[[3, 64, 64], [64, 64, 128]],
            # bs=[[3, 64, 64], [64, 64, 128]],
            # bs=[[3, 64, 64], [64, 64, 128], [128, 128, 256]],
            T=kwargs.get("T", 478),
            n=n,
            Lk=Lk,
            p=0.5,
            num_classes=num_classes,
        )
        return model

    register_model("stgcn", build_stgcn)


_register_builtin_models()


# =========================================================
# Logger
# =========================================================

_train_logger = logging.getLogger("doma.modeling.train")


def _setup_run_logging(log_path: Path):

    _train_logger.setLevel(logging.INFO)

    for h in _train_logger.handlers[:]:
        _train_logger.removeHandler(h)

    formatter = logging.Formatter("%(message)s")

    stream = logging.StreamHandler()
    stream.setFormatter(formatter)

    file_handler = logging.FileHandler(log_path, mode="w")

    file_handler.setFormatter(formatter)

    _train_logger.addHandler(stream)
    _train_logger.addHandler(file_handler)

    return _train_logger.handlers[:]


def _clear_run_logging(handlers):

    for h in handlers:
        _train_logger.removeHandler(h)
        if getattr(h, "close", None):
            h.close()


# =========================================================
# Metrics
# =========================================================

def compute_metrics(y_true, y_pred, num_classes=None):

    acc = accuracy_score(y_true, y_pred)

    labels = list(range(num_classes))

    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", labels=labels, zero_division=0
    )

    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", labels=labels, zero_division=0
    )

    return dict(
        accuracy=float(acc),
        precision_macro=float(p_macro),
        recall_macro=float(r_macro),
        f1_macro=float(f1_macro),
        precision_weighted=float(p_weighted),
        recall_weighted=float(r_weighted),
        f1_weighted=float(f1_weighted),
    )


# =========================================================
# Evaluation
# =========================================================

def evaluate(model, loader, device, desc="Eval"):

    model.eval()

    all_labels = []
    all_preds = []

    total_loss = 0
    n = 0

    criterion = nn.CrossEntropyLoss(reduction="sum")

    with torch.no_grad():

        for batch in tqdm(loader, desc=desc, leave=False):

            pose = batch["pose"].to(device)
            optflow = batch["optflow"].to(device)
            label = batch["label"].to(device)
            x = torch.cat([pose, optflow], dim=-1)         # (B, T, 78)
            B, T, C = x.shape
            x = x.view(B, T, 26, 3)      # split xyz
            x = x.permute(0, 3, 1, 2)    # x = pose
            x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
            # x = skeleton

            logits = model(x)
            loss = criterion(logits, label)

            total_loss += loss.item()

            pred = logits.argmax(dim=1)

            all_preds.extend(pred.cpu().tolist())
            all_labels.extend(label.cpu().tolist())

            n += label.size(0)

    mean_loss = total_loss / n if n else 0

    return all_labels, all_preds, mean_loss


# =========================================================
# Training
# =========================================================

def run_train(
    manifest_path,
    root_dir,
    model_name="stgcn",
    output_dir="models",
    split_mode="train_test",
    batch_size=32,
    epochs=20,
    lr=5e-4,
    max_len=None,
    num_workers=0,
):

    from doma.dataset import create_dataloaders

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    manifest_path = Path(manifest_path)
    root_dir = Path(root_dir)

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    run_dt = datetime.now().strftime("%Y%m%d_%H%M%S")

    run_id = f"{model_name}_{run_dt}"

    ckpt_path = output_dir / f"{run_id}.pt"
    metrics_path = output_dir / f"{run_id}.json"
    log_path = output_dir / f"{run_id}.log"

    handlers = _setup_run_logging(log_path)
    log = _train_logger

    log.info("Run ID: %s", run_id)
    log.info("Device: %s", device)

    loaders = create_dataloaders(
        manifest_path=manifest_path,
        root_dir=root_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        max_len=max_len,
        split_mode=split_mode,
    )

    if split_mode == "train_val_test":
        train_loader, val_loader, test_loader = loaders
        eval_loader = val_loader
    else:
        train_loader, test_loader = loaders
        eval_loader = test_loader

    num_classes = train_loader.dataset.num_classes

    builder, defaults = get_model_builder(model_name)

    model = builder(num_classes=num_classes, **defaults).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    criterion = nn.CrossEntropyLoss()

    best_metric = -1

    history = []

    for epoch in range(epochs):

        model.train()

        train_loss = 0
        train_n = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

        for batch in pbar:

            pose = batch["pose"].to(device)
            optflow = batch["optflow"].to(device)
            label = batch["label"].to(device)

            x = torch.cat([pose, optflow], dim=-1)         # (B, T, 78)
            B, T, C = x.shape
            x = x.view(B, T, 26, 3)      # split xyz
            x = x.permute(0, 3, 1, 2)    # (B,3,T,26)
            x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
            # x = pose

            optimizer.zero_grad()

            logits = model(x)

            loss = criterion(logits, label)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            train_loss += loss.item() * label.size(0)
            train_n += label.size(0)

            pbar.set_postfix(loss=loss.item())

        train_loss /= train_n

        labels, preds, val_loss = evaluate(model, eval_loader, device)

        metrics = compute_metrics(labels, preds, num_classes)

        metrics["train_loss"] = train_loss
        metrics["val_loss"] = val_loss

        history.append(metrics)

        log.info(
            "train_loss=%.4f val_loss=%.4f acc=%.4f f1=%.4f",
            train_loss,
            val_loss,
            metrics["accuracy"],
            metrics["f1_macro"],
        )

        if metrics["accuracy"] > best_metric:

            best_metric = metrics["accuracy"]

            torch.save(
                dict(
                    model_state_dict=model.state_dict(),
                    optimizer_state_dict=optimizer.state_dict(),
                    metrics=metrics,
                    epoch=epoch,
                ),
                ckpt_path,
            )

            log.info("Saved best checkpoint")

    with open(metrics_path, "w") as f:

        json.dump(dict(history=history), f, indent=2)

    _clear_run_logging(handlers)


# =========================================================
# CLI
# =========================================================

def main():

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--manifest", type=Path, default="data/processed/manifest.csv")

    parser.add_argument("--root-dir", type=Path, default=".")

    parser.add_argument("--model", type=str, default="stgcn")

    parser.add_argument("--batch-size", type=int, default=32)

    parser.add_argument("--epochs", type=int, default=100)

    parser.add_argument("--lr", type=float, default=1e-4)

    parser.add_argument("--num-workers", type=int, default=0)

    args = parser.parse_args()

    run_train(
        manifest_path=args.manifest,
        root_dir=args.root_dir,
        model_name=args.model,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()