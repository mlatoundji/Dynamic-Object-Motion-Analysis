## Création du jeu de données unifié (IPN Hand, Jester, MS-ASL, WLASL)

Ce projet fournit un pipeline reproductible qui transforme des vidéos/frames en **artefacts normalisés**:
- **Pose/Kinematics** (spec §5.2): `pose_tensor.npz`
- **Optical-flow features**: `optflow_features.npz`
- `quality.json` + `manifest.csv`

### 0) Prérequis

- Python 3.12+ et Poetry
- Pour MS-ASL/WLASL (download):
  - `yt-dlp` dans le PATH
  - `ffmpeg` dans le PATH

### 1) Installation

```bash
poetry install -E dataset -E hand
```

### MediaPipe (important)

Dans certains environnements (Windows notamment), `mediapipe` peut être fourni en mode **Tasks-only** (pas de `mediapipe.solutions`).  
Le projet gère ce cas en téléchargeant automatiquement le modèle `hand_landmarker.task` dans `.mediapipe_models/` (si réseau disponible).

URL modèle (référence): `https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task`

### 2) Placer les données brutes dans `data/raw/`

#### IPN Hand

- Télécharger et extraire IPN Hand (voir page officielle IPN Hand: `https://gibranbenitez.github.io/IPN_Hand/`).
- Placer les vidéos sous:
  - `data/raw/ipn_hand/videos/**/*.(avi|mp4)` (OpenCV lit les deux)
- Si tu as les **annotations officielles** (fichiers `Annot_*List.txt` + `Video_*List.txt` + `classIdx.txt`),
  place-les sous `data/raw/ipn_hand/annotations/.../` (le pipeline les détecte automatiquement) :
  - **1 sample = 1 segment** (`t_start`/`t_end` en frames, inclusifs)
  - **splits**: train/test officiels, et **val** créé en découpant le TRAIN **au niveau vidéo** via
    `datasets.ipn_hand.val_ratio` + `datasets.ipn_hand.seed` dans `config/datasets.yaml`.
- Sinon, tu peux créer `data/raw/ipn_hand/index.csv` (mode legacy “1 vidéo = 1 sample”):
  - colonnes: `sample_id,split,label,video_path,source_uri`
  - `video_path` peut être relatif à `data/raw/ipn_hand/` (ex: `videos/xx.avi`)

Script d’aide:

```bash
python scripts/datasets/make_ipn_index.py --raw data/raw/ipn_hand --glob "videos/**/*.avi" --split train --label unknown
python scripts/datasets/split_index_csv.py --index data/raw/ipn_hand/index.csv --train 0.8 --val 0.1 --test 0.1 --seed 0 --stratify label
```

#### Jester (20BN-Jester)

https://www.kaggle.com/datasets/toxicmender/20bn-jester?resource=download 


Layout typique:
- `data/raw/jester/train.csv`
- `data/raw/jester/validation.csv`
- `data/raw/jester/20bn-jester-v1/<video_id>/*.jpg`

Validation rapide:

```bash
python scripts/datasets/validate_jester_layout.py --raw data/raw/jester
```

Si tu veux convertir un dossier de frames en mp4:

```bash
python scripts/datasets/frames_dir_to_mp4.py data/raw/jester/20bn-jester-v1/<video_id> --fps 30 --out data/raw/jester/videos/<video_id>.mp4
```

#### MS-ASL

- Placer les annotations sous:
  - `data/raw/ms_asl/annotations/MSASL_train.json`
  - `data/raw/ms_asl/annotations/MSASL_val.json`
  - `data/raw/ms_asl/annotations/MSASL_test.json`
- Les vidéos seront téléchargées sous `data/raw/ms_asl/videos/` via `--download`.

#### WLASL

- Placer `data/raw/wlasl/annotations/WLASL2000.json`
- Les vidéos seront téléchargées sous `data/raw/wlasl/videos/` via `--download`.

### 3) Construire le dataset (processed)

#### Build local (IPN/Jester, sans download)

```bash
poetry run doma-build-dataset --config config/datasets.yaml --only ipn_hand,jester --subset 50
```

Pour debug/perf, tu peux limiter le nombre de frames par vidéo:

```bash
poetry run doma-build-dataset --config config/datasets.yaml --only ipn_hand --subset 1 --max-frames 200
```

#### Build avec download (MS-ASL/WLASL)

```bash
poetry run doma-build-dataset --config config/datasets.yaml --only ms_asl,wlasl --download --subset 100
```

### 4) Sorties

Exemple:

```
data/processed/
  manifest.csv
  jester/train/jester_train_<id>/
    pose_tensor.npz
    optflow_features.npz
    quality.json
```

### 5) Synthétique (Blender/Unreal)

- Blender: voir `synthetic/blender/README.md`
- Unreal: voir `synthetic/unreal/README.md`

