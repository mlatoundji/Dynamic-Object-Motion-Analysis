## Dataset build scripts

Ces scripts préparent et construisent un dataset unifié sous `data/processed/` en produisant, **par sample**:
- `pose_tensor.npz` (tenseur cinématique compatible spec §5.2)
- `optflow_features.npz` (features flot optique dans la ROI main)
- `quality.json` (indicateurs de couverture/validité)
- `manifest.csv` (index global)

### Layout attendu (non versionné)

```
data/
  raw/
    ipn_hand/
      videos/...
      index.csv            # optionnel mais recommandé (voir ci-dessous)
    jester/
      train.csv
      validation.csv
      test.csv
      videos/<id>.mp4                 # optionnel
      20bn-jester-v1/<id>/*.jpg       # courant (frames)
    ms_asl/
      annotations/MSASL_train.json
      annotations/MSASL_val.json
      annotations/MSASL_test.json
      videos/                          # téléchargés via yt-dlp
    wlasl/
      annotations/WLASL2000.json
      videos/                          # téléchargés via yt-dlp
```

### IPN Hand (`data/raw/ipn_hand/index.csv`)

Le format le plus robuste est un `index.csv` que vous contrôlez.

Colonnes minimales:
- `sample_id`
- `split` (train|val|test)
- `label`
- `video_path` (chemin relatif à `data/raw/ipn_hand/` ou chemin absolu)
- `source_uri`

### Build

Depuis la racine du repo:

```bash
poetry install -E dataset -E hand
poetry run doma-build-dataset --config configs/datasets.yaml --only ipn_hand,jester --subset 50
```

### MS-ASL / WLASL (download)

Prérequis:
- `yt-dlp` dans le PATH
- `ffmpeg` dans le PATH (recommandé; remux automatique vers mp4)

```bash
poetry install -E dataset -E hand
poetry run doma-build-dataset --config configs/datasets.yaml --only ms_asl,wlasl --download --subset 100
```

