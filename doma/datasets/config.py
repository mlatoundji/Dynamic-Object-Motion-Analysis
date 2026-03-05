from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DatasetsConfig:
    raw_root: Path
    processed_root: Path
    dt_ms: float
    roi_size: tuple[int, int]
    mediapipe: dict[str, Any]
    outputs: dict[str, bool]
    datasets: dict[str, dict[str, Any]]


def load_config(path: Path) -> DatasetsConfig:
    data = _load_any_config(path)
    raw_root = Path(data.get("raw_root", "data/raw"))
    processed_root = Path(data.get("processed_root", "data/processed"))

    proc = data.get("processing", {}) or {}
    dt_ms = float(proc.get("dt_ms", 33.333))
    roi_size = tuple(int(x) for x in proc.get("roi_size", [224, 224]))
    mediapipe = proc.get("mediapipe", {}) or {}

    outputs = data.get("outputs", {}) or {}
    datasets = data.get("datasets", {}) or {}
    return DatasetsConfig(
        raw_root=raw_root,
        processed_root=processed_root,
        dt_ms=dt_ms,
        roi_size=(int(roi_size[0]), int(roi_size[1])),
        mediapipe=dict(mediapipe),
        outputs={str(k): bool(v) for k, v in outputs.items()},
        datasets={str(k): dict(v) for k, v in datasets.items()},
    )


def _load_any_config(path: Path) -> dict[str, Any]:
    suf = path.suffix.lower()
    if suf in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError("PyYAML is required to read .yaml configs. Install it or use .toml/.json.") from e
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return dict(data or {})
    if suf == ".toml":
        import tomllib

        return dict(tomllib.loads(path.read_text(encoding="utf-8")))
    if suf == ".json":
        import json

        return dict(json.loads(path.read_text(encoding="utf-8")))
    raise ValueError(f"Unsupported config extension: {path}")

