from __future__ import annotations

from pathlib import Path

import numpy as np


def plot_confusion_matrix(
    cm: np.ndarray,
    *,
    labels: list[str],
    out_path: Path,
    normalize: bool = True,
) -> None:
    import matplotlib.pyplot as plt

    cm = np.asarray(cm, dtype=np.float64)
    if normalize:
        row = cm.sum(axis=1, keepdims=True)
        cm = np.divide(cm, np.maximum(row, 1.0))

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(1, 1, 1)
    im = ax.imshow(cm, interpolation="nearest")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Confusion Matrix" + (" (normalized)" if normalize else ""))
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    ticks = np.arange(len(labels))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_training_curves(
    history: dict[str, list[float]], *, out_path: Path
) -> None:
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(10, 4))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    if "train_loss" in history:
        ax1.plot(history["train_loss"], label="train_loss")
    if "val_loss" in history:
        ax1.plot(history["val_loss"], label="val_loss")
    ax1.set_title("Loss")
    ax1.set_xlabel("epoch")
    ax1.legend()

    if "val_macro_f1" in history:
        ax2.plot(history["val_macro_f1"], label="val_macro_f1")
    if "val_accuracy" in history:
        ax2.plot(history["val_accuracy"], label="val_accuracy")
    ax2.set_title("Val metrics")
    ax2.set_xlabel("epoch")
    ax2.legend()

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
