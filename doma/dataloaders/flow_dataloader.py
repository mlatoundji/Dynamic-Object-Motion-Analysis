"""
Flow DataLoader: IPN Hand flow (ipnall_flo.json + flow frame directories).
Builds train/val DataLoaders. Dataset class is internal.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from config.labels import LABEL_TO_ID


def _key_to_flow_dir(key: str, flow_dir: Path) -> Path:
    """Map DB key (e.g. ./flow/1CM1_4_R_#229^2) to flow directory path."""
    rest = key.removeprefix("./flow/").removeprefix("flow/")
    base = rest.split("^")[0] if "^" in rest else rest
    return flow_dir / base


def _get_frame_path(flow_dir: Path, seq_id: str, frame_num: int) -> Path:
    """Path to frame image; frame_num is 1-based."""
    return flow_dir / f"{seq_id}_{frame_num:06d}.jpg"


class _GestureFlowDataset(Dataset[dict[str, Any]]):
    """Internal. Dataset of gesture clips from optical flow frames."""

    def __init__(
        self,
        annotation_path: str | Path,
        flow_dir: str | Path,
        subset: Literal["training", "validation"],
        num_frames: int = 8,
        frame_size: tuple[int, int] = (224, 224),
        sample_mode: Literal["uniform", "consecutive"] = "uniform",
    ):
        self.annotation_path = Path(annotation_path)
        self.flow_dir = Path(flow_dir)
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.sample_mode = sample_mode

        with open(self.annotation_path, encoding="utf-8") as f:
            data = json.load(f)

        self.label_to_id = LABEL_TO_ID
        self.samples = []
        for key, entry in data["database"].items():
            if entry["subset"] != subset:
                continue
            ann = entry["annotations"]
            flow_path = _key_to_flow_dir(key, self.flow_dir)
            if not flow_path.is_dir():
                continue
            start = int(ann["start_frame"])
            end = int(ann["end_frame"])
            label = ann["label"]
            if label not in self.label_to_id:
                continue
            rest = key.removeprefix("./flow/").removeprefix("flow/")
            seq_id = rest.split("^")[0] if "^" in rest else rest
            self.samples.append(
                {
                    "flow_dir": flow_path,
                    "seq_id": seq_id,
                    "start_frame": start,
                    "end_frame": end,
                    "label": label,
                }
            )

    def __len__(self) -> int:
        return len(self.samples)

    @property
    def num_classes(self) -> int:
        return len(self.label_to_id)

    def _sample_frame_indices(self, start: int, end: int) -> list[int]:
        length = end - start + 1
        if length <= self.num_frames:
            indices = list(range(start, end + 1))
            while len(indices) < self.num_frames:
                indices.append(indices[-1] if indices else start)
            return indices[: self.num_frames]
        if self.sample_mode == "uniform":
            indices = np.linspace(start, end, self.num_frames, dtype=int).tolist()
        else:
            step = (length - self.num_frames) / max(1, self.num_frames - 1)
            indices = [start + int(i * step) for i in range(self.num_frames)]
        return indices

    def __getitem__(self, idx: int) -> dict[str, Any]:
        s = self.samples[idx]
        flow_dir = s["flow_dir"]
        seq_id = s["seq_id"]
        start = s["start_frame"]
        end = s["end_frame"]
        label_id = self.label_to_id[s["label"]]

        frame_indices = self._sample_frame_indices(start, end)
        frames = []
        for fi in frame_indices:
            path = _get_frame_path(flow_dir, seq_id, fi)
            img = cv2.imread(str(path))
            if img is None:
                img = np.zeros((self.frame_size[0], self.frame_size[1], 3), dtype=np.uint8)
            else:
                img = cv2.resize(img, (self.frame_size[1], self.frame_size[0]))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            frames.append(img)

        x = np.stack(frames, axis=0).astype(np.float32) / 255.0
        x = torch.from_numpy(x).permute(0, 3, 1, 2)
        return {
            "frames": x,
            "label": label_id,
            "length": self.num_frames,
            "sample_id": str(idx),
        }


def _collate_flow_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    if not batch:
        return {}
    return {
        "frames": torch.stack([s["frames"] for s in batch]),
        "label": torch.tensor([s["label"] for s in batch], dtype=torch.long),
        "length": torch.tensor([s["length"] for s in batch], dtype=torch.long),
        "sample_id": [s["sample_id"] for s in batch],
    }


def build_dataloaders(
    annotation_path: str | Path = "data/ipnall_flo.json",
    flow_dir: str | Path = "data/flow",
    num_frames: int = 8,
    frame_size: tuple[int, int] = (224, 224),
    batch_size: int = 32,
    num_workers: int = 0,
    pin_memory: bool = True,
):
    """Build training and validation DataLoaders for IPN flow (ipnall_flo.json + flow frames)."""
    from torch.utils.data import DataLoader

    train_ds = _GestureFlowDataset(
        annotation_path=annotation_path,
        flow_dir=flow_dir,
        subset="training",
        num_frames=num_frames,
        frame_size=frame_size,
    )
    val_ds = _GestureFlowDataset(
        annotation_path=annotation_path,
        flow_dir=flow_dir,
        subset="validation",
        num_frames=num_frames,
        frame_size=frame_size,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=_collate_flow_batch,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=_collate_flow_batch,
    )
    return train_loader, val_loader
