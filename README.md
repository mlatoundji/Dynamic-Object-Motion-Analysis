# Dynamic Object Motion Analysis (DOMA)

**Détection et analyse du mouvement par flot optique guidé par objets** — combinaison de **détection de main** (MediaPipe / YOLO) et **flot optique** (Farnebäck / RAFT) pour estimer vitesse, direction et amplitude du mouvement, puis **classification de gestes** par modèles séquence→classe (CNN-LSTM, Temporal Transformer, ST-GCN).

---

## Table des matières

- [Dynamic Object Motion Analysis (DOMA)](#dynamic-object-motion-analysis-doma)
  - [Table des matières](#table-des-matières)
  - [À propos](#à-propos)
  - [Structure du projet](#structure-du-projet)
  - [Prérequis et installation](#prérequis-et-installation)
  - [Données](#données)
    - [Labels IPN Hand](#labels-ipn-hand)
    - [Aperçu du dataset IPN Hand (original)](#aperçu-du-dataset-ipn-hand-original)
    - [Sorties du pipeline](#sorties-du-pipeline)
    - [Création du dataset](#création-du-dataset)
  - [Entraînement](#entraînement)
    - [Modèles](#modèles)
    - [Entraînement](#entraînement-1)
  - [Inférence en direct](#inférence-en-direct)
  - [Analyse des logs](#analyse-des-logs)
  - [Documentation](#documentation)
  - [Références](#références)

---

## À propos

Le pipeline repose sur :

1. **Brique sémantique** : détection de la main (ROI manuelle, MediaPipe Hands, ou YOLO) pour restreindre le flot à une zone utile.
2. **Brique dynamique** : flot optique (Farnebäck ou RAFT), filtrage (seuillage, soustraction du fond), puis extraction de **métriques** (vitesse, direction dominante, concentration).
3. **Jeu de données unifié** : pose/cinématique + features de flot par sample, alimentant un **manifest** (`manifest.csv`) et des modèles de classification.
4. **Modèles** : CNN-LSTM, Temporal Transformer, ST-GCN, ST-GCN-Opt, Temporal ViT (flow frames), entraînés via un script unique (`doma-train`) et évalués en live via `doma-live-classifier`.

Les labels de gestes IPN Hand sont définis dans `config/labels.py` (LABELS, LABEL_TO_TEXT, LABEL_TO_ID).

---

## Structure du projet

```
├── config/                 # Configuration (labels, datasets)
├── data/
│   ├── raw/                # Données brutes (vidéos, index, annotations)
│   ├── processed/          # manifest.csv, pose/*.npz, optflow/*.npz
│   └── interim/            # Artefacts intermédiaires
├── doma/
│   ├── cli.py              # doma-live (démo flot)
│   ├── detectors.py        # Manual ROI, MediaPipe, YOLO
│   ├── flow.py             # Farnebäck, RAFT
│   ├── motion.py           # Métriques de mouvement
│   ├── dataloaders/        # build_dataloaders, build_dataloaders_stgcn
│   ├── datasets/           # Construction dataset (IPN Hand)
│   ├── modeling/           # train.py (registry, run_train)
│   ├── models/             # cnn_lstm, temporal_transformer, stgcn, stgcn_opt, temporal_vit
│   ├── live_classifier.py  # doma-live-classifier (inférence temps réel)
│   └── tools/              # analyze_live_logs
├── docs/                   # Guides (création dataset, entraînement, rapports)
├── models/                 # Runs d’entraînement (run_id/best.pt, metrics.json, …)
├── notebooks/              # Exploration et vérifications
├── scripts/                # Utilitaires (datasets, analyse)
└── pyproject.toml          # Dépendances et points d’entrée CLI
```

---

## Prérequis et installation

- **Python** 3.12+
- **Poetry** ou **uv** pour l’environnement

```bash
# Installation de base
uv sync

# Avec options (détection main, dataset, entraînement)
uv sync --extra hand --extra dataset --extra train
```

Commandes CLI exposées : `doma-live`, `doma-build-dataset`, `doma-train`, `doma-live-classifier`.

---

## Données

Le projet utilise le jeu de données **[IPN Hand](https://gibranbenitez.github.io/IPN_Hand/)**, un benchmark vidéo pour la reconnaissance continue de gestes de la main : plus de **4 000 séquences de gestes** et **800 000 frames** (50 sujets), conçu pour l’interaction avec des écrans sans contact. Les vidéos sont en **640×480 à 30 fps** ; le pipeline produit, par sample, des artefacts normalisés à partir de ces vidéos.

### Labels IPN Hand

Les **14 classes** (1 non-geste + 13 gestes) et leur description sont définies dans `config/labels.py` (LABELS, LABEL_TO_TEXT, LABEL_TO_ID) :

| Id | Label | Description |
|----|-------|-------------|
| 0 | D0X | Non-gesture |
| 1 | B0A | Pointing with one finger |
| 2 | B0B | Pointing with two fingers |
| 3 | G01 | Click with one finger |
| 4 | G02 | Click with two fingers |
| 5 | G03 | Throw up |
| 6 | G04 | Throw down |
| 7 | G05 | Throw left |
| 8 | G06 | Throw right |
| 9 | G07 | Open twice |
| 10 | G08 | Double click with one finger |
| 11 | G09 | Double click with two fingers |
| 12 | G10 | Zoom in |
| 13 | G11 | Zoom out |

Référence : [IPN Hand – Details](https://gibranbenitez.github.io/IPN_Hand/), article ICPR 2020.

### Aperçu du dataset IPN Hand (original)

Exemples de gestes :

<p align="center">
  <img src="docs/images/c1.gif" width="140" /> <img src="docs/images/c2.gif" width="140" /> <img src="docs/images/c3.gif" width="140" /> <img src="docs/images/c4.gif" width="140" /> <img src="docs/images/c5.gif" width="140" />
</p>
<p align="center">
  <img src="docs/images/c6.gif" width="140" /> <img src="docs/images/c7.gif" width="140" /> <img src="docs/images/c8.gif" width="140" /> <img src="docs/images/c9.gif" width="140" /> <img src="docs/images/c10.gif" width="140" />
</p>
<p align="center">
  <img src="docs/images/c11.gif" width="140" /> <img src="docs/images/c12.gif" width="140" /> <img src="docs/images/c13.gif" width="140" />
</p>

### Sorties du pipeline

Pour chaque sample (segment vidéo IPN Hand), le pipeline génère :

- **Pose / cinématique** : `pose_*.npz` — position, vitesse et accélération du track 3D, optionnellement les 21 landmarks main (MediaPipe).
- **Flot optique** : `optflow_*.npz` — métriques temporelles (vitesse moyenne/max, direction dominante, concentration, seuil, etc.).

Un **manifest** (`data/processed/manifest.csv`) regroupe tous les samples avec les colonnes : `sample_id`, `split`, `label`, `pose_npz`, `optflow_npz`. Les chemins sont relatifs à la racine du projet.

### Création du dataset

- **Guide détaillé** : `docs/DATASET_CREATION.md`.
- **Configuration** : `config/datasets.yaml` (section `ipn_hand`).

Exemple de construction d’un sous-ensemble à partir des vidéos brutes dans `data/raw/ipn_hand/` :

```bash
uv run doma-build-dataset --config config/datasets.yaml --only ipn_hand --subset 10
```

---

## Entraînement

Un seul point d’entrée : **`doma-train`**. Le script choisit le dataloader et le format de batch selon le modèle.

### Modèles

| Modèle                 | Description                                      | Données        |
|------------------------|--------------------------------------------------|----------------|
| `temporal_transformer` | Transformer sur séquence pose+optflow            | Manifest |
| `cnn_lstm`             | CNN + LSTM sur features unifiées                 | Manifest |
| `stgcn`                | GCN spatio-temporel (skeleton + motion)          | Manifest |
| `stgcn_opt`            | Multi-branches (ST-GCN + track + optflow)        | Manifest |
| `temporal_vit`         | Vision Transformer sur frames de flot            | Raw IPN flow |

### Entraînement

```bash
# Depuis la racine du projet
uv run doma-train --model temporal_transformer --manifest data/processed/manifest.csv --epochs 20 --batch-size 32 --output-dir models
uv run doma-train --model stgcn --epochs 20 --batch-size 32 --output-dir models
uv run doma-train --model cnn_lstm --epochs 20 --batch-size 32 --output-dir models
```

Options utiles : `--root-dir`, `--split-mode` (train_val_test | train_test), `--max-len`, `--lr`, `--save-best-by` (accuracy | f1_macro | f1_weighted).

Chaque run crée un répertoire horodaté sous `models/` (ex. `models/stgcn_20260307_224459/`) contenant : `best.pt`, `metrics.json`, `train_log.log`, `model_config.json`, `train_config.json`, `label_map.json`, et selon le modèle `norm.npz`.

**Guide complet** : `docs/TRAINING_GUIDE.md`.

---

## Inférence en direct

Classification de gestes en temps réel à partir de la webcam (ou d’une vidéo), avec un run entraîné (CNN-LSTM ou Temporal Transformer recommandés pour le live) :

```bash
uv run doma-live-classifier --run models/<run_id> --source 0
```

- **Quitter** : touche `q`
- **Reset** : touche `r`

Options utiles : `--window-ms`, `--infer-every-ms`, `--ema`, `--d0x-thr`, `--mirror-view` / `--flip-features`, `--log` / `--no-log`, `--log-dir`. Les sessions sont enregistrées par défaut sous `doma/sessions/` (CSV, TXT, optionnel NPZ).

---

## Analyse des logs

Résumé d’une session live (métriques, latence) :

```bash
uv run python -m doma.tools.analyze_live_logs --csv doma/sessions/<session>/report_*.csv --out doma/sessions/<session>/analysis_summary.json
```

Avec segments annotés (CSV : `t_start_ms,t_end_ms,label`) pour confusion et latence :

```bash
uv run python -m doma.tools.analyze_live_logs --csv doma/sessions/<session>/report_*.csv --segments segments.csv --out doma/sessions/<session>/analysis_with_segments.json
```

---

## Documentation

| Document | Contenu |
|----------|---------|
| `docs/DATASET_CREATION.md` | Création du dataset IPN Hand |
| `docs/TRAINING_GUIDE.md`   | Dataloaders, entraînement, modèles (ST-GCN, Temporal Transformer, etc.) |
| `config/labels.py`         | Labels de gestes (LABELS, LABEL_TO_TEXT, LABEL_TO_ID) |

---

## Références

- **IPN Hand** : [gibranbenitez.github.io/IPN_Hand](https://gibranbenitez.github.io/IPN_Hand/)
- Détection : MediaPipe Hands, YOLO (Ultralytics). Flot : Farnebäck (OpenCV), RAFT.

---

*Projet DOMA — Analyse du mouvement par flot optique et classification de gestes.*
