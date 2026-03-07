# Training guide: dataset, training script, and Temporal Transformer

This guide explains how to use the gesture dataset and training pipeline to train a model, and how the **Temporal Transformer** works in brief.

---

## 1. Prerequisites

- **Manifest**: `data/processed/manifest.csv` with columns `sample_id`, `dataset`, `split`, `label`, `pose_npz`, `optflow_npz`.
- **Processed data**: For each row, the paths in `pose_npz` and `optflow_npz` point to `.npz` files (relative to the project root). Labels must match `config.labels.LABEL_TO_ID`.

Run from the **project root** so paths resolve correctly.

---

## 2. Data loaders (`doma.dataloaders`)

The package **`doma.dataloaders`** exposes DataLoader builders; dataset classes are internal.

- **`doma.dataloaders.flow_dataloader`**: `build_dataloaders` — IPN flow (ipnall_flo.json + flow frames) → train_loader, val_loader.
- **`doma.dataloaders.dataloader`**: `build_dataloaders`, `collate_gesture_batch` — manifest.csv (pose/optflow NPZ) → train/val/test loaders.

### build_dataloaders (`doma.dataloaders.dataloader`)

Builds DataLoaders from `manifest.csv` (pose/optflow NPZ paths):

- **Purpose**: Loads one sample per row from the manifest, reads pose and optflow NPZ, returns batches with padding.
- **Key arguments**:
  - `manifest_path`: path to `manifest.csv`
  - `root_dir`: project root (e.g. `"."`)
  - `split_mode`: `"train_val_test"` (train, val, test) or `"train_test"` (train+val, test)
  - `batch_size`, `num_workers`, `max_len`, `pad_value`, `pin_memory`, `label_to_id`
- **Per sample in batch**:
  - `pose`: (B, T_max, 72), `optflow`: (B, T_max, 6), `label`, `length`, `sample_id`, `valid_pose`, `valid_optflow`

Returns `(train_loader, val_loader, test_loader)` or `(train_loader, test_loader)`.

### build_dataloaders (`doma.dataloaders.flow_dataloader`)

Builds train/val DataLoaders for the IPN Hand flow dataset (`ipnall_flo.json` + flow frame directories):

- **Arguments**: `annotation_path`, `flow_dir`, `num_frames`, `frame_size`, `batch_size`, `num_workers`, `pin_memory`
- **Batch**: `frames` (B, T, C, H, W), `label`, `length`, `sample_id`

Returns `(train_loader, val_loader)`.

### collate_gesture_batch

Pads variable-length manifest samples to the max length in the batch (or `max_len`). Use as `collate_fn` when building a DataLoader manually.

---

## 3. Training (`doma/modeling/train.py`)

### CLI: `doma-train`

From the project root:

```bash
poetry run doma-train --model temporal_transformer --epochs 20 --batch-size 32 --output-dir models
```

or

```bash
uv run doma-train --model temporal_transformer --epochs 20 --batch-size 32 --output-dir models
```

**Useful options**:

| Option | Default | Description |
|--------|--------|-------------|
| `--manifest` | `data/processed/manifest.csv` | Manifest path |
| `--root-dir` | `.` | Root for data paths |
| `--model` | `temporal_transformer` | Model name (from registry) |
| `--output-dir` | `models` | Where to save checkpoint, metrics, log |
| `--split-mode` | `train_test` | `train_val_test` or `train_test` |
| `--batch-size` | 32 | Batch size |
| `--epochs` | 20 | Number of epochs |
| `--lr` | 5e-4 | Learning rate |
| `--max-len` | None | Max sequence length (truncation) |
| `--save-best-by` | `accuracy` | Metric for best checkpoint: `accuracy`, `f1_macro`, `f1_weighted` |

**Split behaviour**:

- **train_val_test**: train on **train**, evaluate each epoch on **val**. Best checkpoint by val metric.
- **train_test**: train on **train+val**, evaluate each epoch on **test**. Best checkpoint by test metric.

### From Python

```python
from pathlib import Path
from doma.modeling.train import run_train

run_train(
    manifest_path=Path("data/processed/manifest.csv"),
    root_dir=Path("."),
    model_name="temporal_transformer",
    output_dir=Path("models"),
    split_mode="train_test",  # or "train_val_test"
    batch_size=32,
    epochs=20,
    lr=5e-4,
    max_len=256,  # optional
    save_best_by="accuracy",
)
```

### Outputs per run

Each run gets a single timestamped ID: `{model_name}_{YYYYmmdd_HHMMSS}`. In `output_dir` you get:

- **`{run_id}.pt`** — Best checkpoint (model state, optimizer, metrics, `num_classes`, `model_name`).
- **`{run_id}.json`** — Metrics: `best_epoch`, `best_metrics`, `history` (per-epoch metrics).
- **`{run_id}.log`** — Training log (same lines as console, also in the file).

Logging goes to both console and the log file via Python’s `logging`.

---

## 4. Temporal Transformer

**File**: `doma/models/temporal_transformer.py`

**Idea**: Treat each time step as a token (pose + optflow), run a Transformer over time, then pool and classify.

**Input**: Batch of sequences — `pose` (B, T, 72), `optflow` (B, T, 6), `length` (B,). Sequences are padded; `length` gives the true length per sample.

**Forward steps**:

1. **Concat** pose and optflow → (B, T, 78).
2. **Sanitize** `nan`/`inf`; scale optflow block by 1/5000 so large values (e.g. `n_pixels`) don’t blow up the next layer.
3. **LayerNorm(78)** on the 78-dim input for stable scale.
4. **Linear(78 → d_model)** + LayerNorm → (B, T, d_model).
5. **Sinusoidal positional encoding** + dropout.
6. **TransformerEncoder** with `src_key_padding_mask` from `length` so padding is ignored.
7. **Masked mean over time**: mean over non-padded positions → (B, d_model).
8. **Linear(d_model → num_classes)** → logits (B, num_classes).

**Default config**: `d_model=128`, `nhead=4`, `num_encoder_layers=3`, `dim_feedforward=256`, `dropout=0.1`. The model is registered as `"temporal_transformer"` and used by the training script when `--model temporal_transformer`.

---

## 5. Adding another model

1. Implement an `nn.Module` whose `forward` accepts `pose`, `optflow`, `length` (or a batch dict) and returns logits (B, num_classes).
2. In `doma/modeling/train.py`, call `register_model("my_model", MyModelClass, default_kwargs={...})` inside `_register_builtin_models()` (or from your code before training).
3. Run with `--model my_model`.

The training loop and metrics are model-agnostic; only the forward signature and registry are required.

---

## 6. IPN Hand flow and Temporal ViT

For the IPN Hand dataset (flow frames + `ipnall_flo.json`): use **`doma.dataloaders.flow_dataloader.build_dataloaders`** (or `from doma.dataloaders import build_dataloaders`) and the **Temporal ViT** model (`doma/models/temporal_vit.py`, registered as `temporal_vit`). Frames live under `flow_dir` (default `root_dir/flow`) as `<seq_id>_<frame:06d>.jpg`. Train with:

```bash
uv run doma-train --dataset-type ipn_flow --annotation data/raw/ipn_hand/ipnall_flo.json --root-dir data/raw/ipn_hand --model temporal_vit --output-dir models
```

`--annotation` is mandatory when `--dataset-type ipn_flow`. The IPN dataset has only train/val splits; no `--split-mode` is used.
