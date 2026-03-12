"""
Generate train loss / val loss vs epoch plots from training runs.

- Per-model figures from models/*/metrics.json (and Temporal ViT train_log.log)
- One joint, publication-style figure comparing all models

Run from repo root:

    MPLBACKEND=Agg MPLCONFIGDIR=.mplconfig uv run python scripts/plot_train_val_loss.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def model_display_name(folder_name: str) -> str:
    if folder_name.startswith("cnn_lstm"):
        return "CNN-LSTM"
    if folder_name.startswith("temporal_transformer"):
        return "Temporal Transformer"
    if folder_name.startswith("stgcn_opt"):
        return "ST-GCN-Opt"
    if folder_name.startswith("stgcn"):
        return "ST-GCN"
    if folder_name.startswith("temporal_vit"):
        return "Temporal ViT"
    return folder_name.split("_")[0].replace("_", " ").title()


def load_history_from_metrics(metrics_path: Path) -> tuple[list[float], list[float]]:
    with open(metrics_path, encoding="utf-8") as f:
        data = json.load(f)
    history = data.get("history", [])
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]
    return train_loss, val_loss


def load_history_from_temporal_vit_log(log_path: Path) -> tuple[list[float], list[float]]:
    train_loss: list[float] = []
    val_loss: list[float] = []
    pattern = re.compile(
        r"train_loss=([\d.]+)\s+val_loss=([\d.]+)"
    )
    with open(log_path, encoding="utf-8") as f:
        for line in f:
            m = pattern.search(line)
            if m:
                train_loss.append(float(m.group(1)))
                val_loss.append(float(m.group(2)))
    return train_loss, val_loss


def plot_train_val_loss(
    train_loss: list[float],
    val_loss: list[float],
    title: str,
    out_path: Path,
) -> None:
    """Single-model plot (kept mainly for debugging)."""
    epochs = np.arange(1, len(train_loss) + 1, dtype=int)
    plt.style.use("seaborn-v0_8-whitegrid")  # clean, article-like
    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    ax.plot(epochs, train_loss, label="Train", color="C0", linewidth=1.8)
    ax.plot(epochs, val_loss, label="Validation", color="C1", linestyle="--", linewidth=1.8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend(frameon=False)
    ax.set_xlim(left=1, right=max(epochs))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_all_models_joint(
    curves: dict[str, tuple[list[float], list[float]]],
    out_path: Path,
) -> None:
    """
    Joint figure: all models' train/val loss on the same axes.
    Style: closer to academic plots (no markers, consistent colors, legend outside).
    """
    if not curves:
        return

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(9, 4))

    colors = {
        "CNN-LSTM": "C0",
        "Temporal Transformer": "C1",
        "ST-GCN-Opt": "C2",
        "ST-GCN": "C3",
        "Temporal ViT": "C4",
    }

    # First pass to find max epoch
    max_epoch = 0
    for train_loss, _ in curves.values():
        max_epoch = max(max_epoch, len(train_loss))

    # Plot each model
    for name, (train_loss, val_loss) in curves.items():
        epochs = np.arange(1, len(train_loss) + 1, dtype=int)
        c = colors.get(name, None)
        ax.plot(
            epochs,
            train_loss,
            label=f"{name} – train",
            color=c,
            linewidth=1.8,
        )
        ax.plot(
            epochs,
            val_loss,
            label=f"{name} – val",
            color=c,
            linestyle="--",
            linewidth=1.8,
        )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_xlim(left=1, right=max_epoch)
    ax.set_title("Évolution de la loss (train / val) par modèle")
    ax.grid(True, alpha=0.3)

    # Legend outside on the right
    ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.5))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0.0, 0.0, 0.82, 1.0))
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    models_dir = repo_root / "models"
    out_dir = repo_root / "docs" / "images"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect curves for joint figure
    joint_curves: dict[str, tuple[list[float], list[float]]] = {}

    # 1) From metrics.json (manifest-based models)
    for metrics_path in sorted(models_dir.glob("*/metrics.json")):
        run_dir = metrics_path.parent
        name = model_display_name(run_dir.name)
        # slug: cnn_lstm, temporal_transformer, stgcn_opt, stgcn
        if run_dir.name.startswith("cnn_lstm"):
            slug = "cnn_lstm"
        elif run_dir.name.startswith("temporal_transformer"):
            slug = "temporal_transformer"
        elif run_dir.name.startswith("stgcn_opt"):
            slug = "stgcn_opt"
        elif run_dir.name.startswith("stgcn"):
            slug = "stgcn"
        else:
            slug = run_dir.name.split("_")[0]
        try:
            train_loss, val_loss = load_history_from_metrics(metrics_path)
        except Exception as e:
            print(f"Skip {run_dir.name}: {e}")
            continue
        if not train_loss or not val_loss:
            print(f"Skip {run_dir.name}: empty history")
            continue
        out_path = out_dir / f"train_val_loss_{slug}.png"
        plot_train_val_loss(train_loss, val_loss, name, out_path)
        print(f"Saved {out_path}")
        joint_curves[name] = (train_loss, val_loss)

    # 2) Temporal ViT from train_log.log (flow-based model)
    for log_path in sorted(models_dir.glob("temporal_vit*/train_log.log")):
        run_dir = log_path.parent
        name = "Temporal ViT"
        try:
            train_loss, val_loss = load_history_from_temporal_vit_log(log_path)
        except Exception as e:
            print(f"Skip {run_dir.name}: {e}")
            continue
        if not train_loss or not val_loss:
            print(f"Skip {run_dir.name}: empty log")
            continue
        out_path = out_dir / "train_val_loss_temporal_vit.png"
        plot_train_val_loss(train_loss, val_loss, name, out_path)
        print(f"Saved {out_path}")
        joint_curves[name] = (train_loss, val_loss)
        break

    # 3) Joint, publication-style figure
    joint_path = out_dir / "train_val_loss_all_models.png"
    plot_all_models_joint(joint_curves, joint_path)
    print(f"Saved {joint_path}")


if __name__ == "__main__":
    main()
