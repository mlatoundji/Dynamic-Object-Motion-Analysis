# Training guide: dataset, training script, and models

This guide explains how to use the gesture dataset and training pipeline, which models are available, and how the **Temporal Transformer**, **ST-GCN**, and **ST-GCN-Opt** fit in.

---

## 1. Prerequisites

- **Manifest**: `data/processed/manifest.csv` with columns `sample_id`, `dataset`, `split`, `label`, `pose_npz`, `optflow_npz`.
- **Processed data**: For each row, the paths in `pose_npz` and `optflow_npz` point to `.npz` files (relative to the project root). Labels must match the label set used in the manifest.

Run from the **project root** so paths resolve correctly.

---

## 2. Data loaders (`doma.dataloaders`)

The package **`doma.dataloaders`** exposes DataLoader builders; dataset classes are internal.

- **`build_dataloaders`** (`doma.dataloaders.dataloader`): Manifest (pose/optflow NPZ) → unified feature vectors `x` (B, T, F), `lengths`, `label`. Used by **temporal_transformer**, **cnn_lstm**.
- **`build_dataloaders_stgcn`** (`doma.dataloaders.stgcn_dataloader`): Same manifest → batches with **skeleton** (B, 3, T, n), **motion** (B, 3, T, n), **track** (B, T, 9), **optflow** (B, T, 6), `label`, `lengths`. Used by **stgcn**, **stgcn_opt**.
- **`build_flow_dataloaders`** (`doma.dataloaders.flow_dataloader`): IPN flow (ipnall_flo.json + flow frames) → train_loader, val_loader. Used by **temporal_vit** when `--dataset-type ipn_flow`.

### build_dataloaders (unified features)

Builds DataLoaders from `manifest.csv` (pose/optflow NPZ paths):

- **Purpose**: Loads one sample per row, reads pose and optflow NPZ, returns batches with padded sequences and optional normalization.
- **Key arguments**: `manifest_path`, `root_dir`, `split_mode` (`"train_val_test"` or `"train_test"`), `batch_size`, `num_workers`, `max_len`, `pin_memory`, `generator`.
- **Per batch**: `x` (B, T_max, F), `lengths` (B,), `label` (B,), optional `sample_id`.

Returns `(train_loader, val_loader, test_loader)` or `(train_loader, test_loader)`.

### build_dataloaders_stgcn (ST-GCN / ST-GCN-Opt)

Builds DataLoaders for **stgcn** and **stgcn_opt** from the same manifest:

- **Purpose**: Loads skeleton, motion, track, and optflow from pose/optflow NPZ per sample; collates to fixed-length batches.
- **Key arguments**: `manifest_path`, `root_dir`, `split_mode`, `batch_size`, `num_workers`, `max_len`, `use_landmarks` (default True → 21 keypoints), `pin_memory`, `generator`.
- **Per batch**: `skeleton` (B, 3, T, n), `motion` (B, 3, T, n), `track` (B, T, 9), `optflow` (B, T, 6), `label` (B,), `lengths` (B,).

Returns the same split structure as `build_dataloaders`.

### build_dataloaders (flow_dataloader — IPN Hand)

Builds train/val DataLoaders for the IPN Hand flow dataset (`ipnall_flo.json` + flow frame directories):

- **Arguments**: `annotation_path`, `flow_dir`, `num_frames`, `frame_size`, `batch_size`, `num_workers`, `pin_memory`
- **Batch**: `frames` (B, T, C, H, W), `label`, `length`, `sample_id`

Returns `(train_loader, val_loader)`.

### collate_gesture_batch / collate_padded

Pads variable-length manifest samples to the max length in the batch (or `max_len`). Use as `collate_fn` when building a DataLoader manually.

---

## 3. Training (`doma/modeling/train.py`)

All models are trained through a **single script** and **model registry**. The same CLI and `run_train()` API apply; the script chooses the right dataloader and batch format per model.

### Registered models

| Model name | Description | Data |
|------------|-------------|------|
| `temporal_transformer` | Transformer over time (pose+optflow tokens) | Manifest, unified `x` |
| `cnn_lstm` | CNN + LSTM on unified features | Manifest, unified `x` |
| `stgcn` | Spatial-temporal GCN (skeleton + motion) | Manifest, ST-GCN loader |
| `stgcn_opt` | Multi-branch: ST-GCN + track + optflow (concat/attention/gated fusion) | Manifest, ST-GCN loader |
| `temporal_vit` | Vision Transformer on flow frames | IPN flow (`--dataset-type ipn_flow`) |

### CLI: `doma-train`

From the project root:

```bash
uv run doma-train --model temporal_transformer --epochs 20 --batch-size 32 --output-dir models
```

or

```bash
uv run doma-train --model temporal_transformer --epochs 20 --batch-size 32 --output-dir models
```

**Examples by model**:

```bash
# Temporal Transformer (default)
uv run doma-train --model temporal_transformer --epochs 20 --batch-size 32 --output-dir models

# ST-GCN (skeleton + motion from same manifest)
uv run doma-train --model stgcn --epochs 20 --batch-size 32 --output-dir models

# ST-GCN-Opt (skeleton + motion + track + optflow, concat fusion)
uv run doma-train --model stgcn_opt --epochs 20 --batch-size 32 --output-dir models
```

**Useful options**:

| Option | Default | Description |
|--------|--------|-------------|
| `--manifest` | `data/processed/manifest.csv` | Manifest path |
| `--root-dir` | `.` | Root for data paths |
| `--model` | `temporal_transformer` | Model name (from registry) |
| `--output-dir` | `models` | Directory for run subfolders |
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
    model_name="temporal_transformer",  # or "stgcn", "stgcn_opt", "cnn_lstm"
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

Each run gets a timestamped directory: **`output_dir / {model_name}_{YYYYmmdd_HHMMSS}`**. Inside that directory:

- **`best.pt`** — Best checkpoint (model state, optimizer, metrics, `num_classes`, `model_name`, `model_config`).
- **`metrics.json`** — Best metrics, `best_epoch`, and full `history` (per-epoch metrics).
- **`train_log.log`** — Training log (same as console).
- **`model_config.json`** — Model config used (for reproducibility).
- **`train_config.json`** — Training args (manifest, lr, batch_size, etc.).
- **`label_map.json`** — (Manifest models) `label_to_idx` / `idx_to_label` for inference.
- **`norm.npz`** — (cnn_lstm / temporal_transformer only) Normalization stats for live classifier.

Logging goes to both console and the log file.

---

## 4. Temporal Transformer

**File**: `doma/models/temporal_transformer.py`

**Idea**: Treat each time step as a token (pose + optflow), run a Transformer over time, then pool and classify.

**Input**: Batch with `x` (B, T, F), `lengths` (B,). The model uses **`forward(batch=batch)`**; the training loop passes the full batch dict.

**Forward steps** (in brief): concat pose/optflow → sanitize → LayerNorm → Linear → positional encoding → TransformerEncoder with padding mask → masked mean over time → classifier. Default config: `d_model`, `nhead`, `num_encoder_layers`, `dim_feedforward`, `dropout`. Registered as **`temporal_transformer`**.

---

## 5. ST-GCN and ST-GCN-Opt

**Files**: `doma/models/stgcn.py`, `doma/models/stgcn_opt.py`

**ST-GCN** uses skeleton (3, T, n) and motion (3, T, n) from the same manifest NPZ. Input is concatenated to (B, 6, T, n); spatial-temporal graph convolution and pooling produce logits. It accepts **`forward(batch=batch)`** with `batch["skeleton"]` and `batch["motion"]`, or legacy **`forward(x)`** with x (B, 6, T, n). Config: **`ModelConfig`** (`num_classes`, `num_keypoints`, `dropout`, `ks`, `kt`, `channel_config`). Registered as **`stgcn`**.

**ST-GCN-Opt** adds **track** (B, T, 9) and **optflow** (B, T, 6) branches and fuses them (concat, weighted, attention, or gated). Same manifest and **`build_dataloaders_stgcn`** provide all four inputs. **`forward(batch=batch)`** with `skeleton`, `motion`, `track`, `optflow`. Config: **`ModelConfig`** (`num_classes`, `num_keypoints`, `dropout`, `fusion_type`, `use_lstm_for_track`, `use_lstm_for_optflow`). Registered as **`stgcn_opt`**. Variants (e.g. attention fusion) can be set via **`model_kwargs`** when calling `run_train`.

---

## 6. Adding another model

1. Implement an **`nn.Module`** with a **`ModelConfig`** dataclass and **`forward(batch=...)`** that returns logits (B, num_classes). Batch may contain `x`/`lengths` (unified features) or `skeleton`/`motion`/`track`/`optflow` (ST-GCN style).
2. In **`doma/modeling/train.py`**, inside **`_register_builtin_models()`**, register with **`register_model("name", builder_fn, default_kwargs={...})`**. The builder should take `num_classes` and optional `in_features` (for manifest feature models) or other kwargs, and return an instance with **`.cfg`**.
3. If the model needs a different dataloader (e.g. ST-GCN), branch in **`run_train()`** on `model_name` to call the appropriate **`build_dataloaders_*`** and pass the resulting batches into **`model(batch=batch_dev)`**.
4. Run with **`--model name`**.

The training loop and metrics are model-agnostic; only the batch interface and registry are required.

---

## 7. IPN Hand flow and Temporal ViT

For the IPN Hand dataset (flow frames + `ipnall_flo.json`): use **`doma.dataloaders.flow_dataloader.build_dataloaders`** (or `from doma.dataloaders import build_flow_dataloaders`) and the **Temporal ViT** model (`doma/models/temporal_vit.py`, registered as `temporal_vit`). Frames live under `flow_dir` (default `root_dir/flow`) as `<seq_id>_<frame:06d>.jpg`. Train with:

```bash
uv run doma-train --dataset-type ipn_flow --annotation data/raw/ipn_hand/ipnall_flo.json --root-dir data/raw/ipn_hand --model temporal_vit --output-dir models
```

`--annotation` is mandatory when `--dataset-type ipn_flow`. The IPN dataset has only train/val splits; no `--split-mode` is used.
