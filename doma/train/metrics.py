from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class EvalMetrics:
    accuracy: float
    macro_f1: float
    micro_f1: float


def confusion_matrix_counts(
    y_true: np.ndarray, y_pred: np.ndarray, *, num_classes: int
) -> np.ndarray:
    cm = np.zeros((int(num_classes), int(num_classes)), dtype=np.int64)
    for t, p in zip(
        y_true.astype(int).tolist(),
        y_pred.astype(int).tolist(),
        strict=False,
    ):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t, p] += 1
    return cm


def compute_basic_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, *, num_classes: int
) -> EvalMetrics:
    # Prefer sklearn when available (more robust), else fall back.
    try:
        from sklearn.metrics import f1_score

        labs = list(range(int(num_classes)))
        macro = float(f1_score(y_true, y_pred, average="macro", labels=labs))
        micro = float(f1_score(y_true, y_pred, average="micro", labels=labs))
    except Exception:
        cm = confusion_matrix_counts(
            y_true, y_pred, num_classes=num_classes
        ).astype(np.float64)
        tp = np.diag(cm)
        fp = np.sum(cm, axis=0) - tp
        fn = np.sum(cm, axis=1) - tp
        f1s = []
        for k in range(int(num_classes)):
            denom = 2 * tp[k] + fp[k] + fn[k]
            f1s.append(float((2 * tp[k] / denom) if denom > 0 else 0.0))
        macro = float(np.mean(f1s)) if f1s else 0.0
        denom = 2 * tp.sum() + fp.sum() + fn.sum()
        micro = float((2 * tp.sum() / denom) if denom > 0 else 0.0)

    acc = (
        float(np.mean((y_true == y_pred).astype(np.float64)))
        if y_true.size
        else 0.0
    )
    return EvalMetrics(accuracy=acc, macro_f1=macro, micro_f1=micro)


def classification_report_dict(
    y_true: np.ndarray, y_pred: np.ndarray, *, labels: list[str]
) -> dict[str, Any]:
    """
    Returns a JSON-serializable classification report. Uses sklearn if available.
    """
    try:
        from sklearn.metrics import classification_report

        return classification_report(
            y_true,
            y_pred,
            labels=list(range(len(labels))),
            target_names=labels,
            output_dict=True,
            zero_division=0,
        )
    except Exception:
        # Minimal report (per-class precision/recall/f1/support).
        cm = confusion_matrix_counts(
            y_true, y_pred, num_classes=len(labels)
        ).astype(np.float64)
        tp = np.diag(cm)
        fp = np.sum(cm, axis=0) - tp
        fn = np.sum(cm, axis=1) - tp
        support = np.sum(cm, axis=1)
        out: dict[str, Any] = {}
        for k, name in enumerate(labels):
            prec = float(tp[k] / (tp[k] + fp[k])) if (tp[k] + fp[k]) > 0 else 0.0
            rec = float(tp[k] / (tp[k] + fn[k])) if (tp[k] + fn[k]) > 0 else 0.0
            f1 = (
                float((2 * prec * rec / (prec + rec)))
                if (prec + rec) > 0
                else 0.0
            )
            out[name] = {
                "precision": prec,
                "recall": rec,
                "f1-score": f1,
                "support": int(support[k]),
            }
        out["accuracy"] = (
            float(np.mean((y_true == y_pred).astype(np.float64)))
            if y_true.size
            else 0.0
        )
        return out
