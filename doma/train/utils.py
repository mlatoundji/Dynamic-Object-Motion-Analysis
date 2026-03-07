from __future__ import annotations

import json
import os
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

try:  # torch is optional
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    os.environ["PYTHONHASHSEED"] = str(int(seed))
    if torch is not None:
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))


def pick_device(device: str) -> str:
    if torch is None:  # pragma: no cover
        return "cpu"
    d = (device or "auto").strip().lower()
    if d in {"auto", ""}:
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"
    if d == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return d


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def dataclass_to_json(path: Path, obj: Any) -> None:
    save_json(path, asdict(obj))
