# Rapport d’entraînement — CNN‑LSTM (IPN Hand)
- **Run**: `classify_20260305-211230`
- **Artefacts**: `C:/Users/moura/Documents/Studies/TP/Flot_Optique/Dynamic Object Motion Analysis/runs/classify_20260305-211230`

---
## Contexte et objectifs
Ce rapport documente l’entraînement d’un classifieur temporel **CNN‑LSTM** destiné à prédire un label de geste discret (incluant **`D0X` non‑gesture**) à partir de séries temporelles issues du pipeline DOMA.

## Données et labels
- **Source**: IPN Hand (segments annotés)
- **Nombre de classes**: 14
- **Classes**: `B0A`, `B0B`, `D0X`, `G01`, `G02`, `G03`, `G04`, `G05`, `G06`, `G07`, `G08`, `G09`, `G10`, `G11`

### Index utilisé
- `manifest.csv`: `data/processed/manifest.csv`

## Représentation d’entrée
- **Pose**: `track_pos_xyz` + `track_vel_xyz` + `track_acc_xyz` (et option `landmarks_xyz` aplati)
- **Flot optique**: `avg_speed`, `max_speed`, `dominant_angle_deg` (sin/cos), `direction_concentration`, `n_pixels`, `threshold`
- **Masquage**: suppression des timestamps invalides (`valid`) avant padding.

## Modèle
### Architecture
- **Conv1D temporel** → **LSTM** → pooling moyen masqué → tête fully-connected (softmax)

### Hyperparamètres (run)
```json
{
  "model_config": {
    "bidirectional": true,
    "conv_channels": 128,
    "conv_dropout": 0.2,
    "conv_kernel": 5,
    "conv_layers": 2,
    "head_dropout": 0.2,
    "in_features": 79,
    "lstm_dropout": 0.2,
    "lstm_hidden": 256,
    "lstm_layers": 1,
    "num_classes": 14
  },
  "train_config": {
    "batch_size": 32,
    "bidirectional": true,
    "class_weight": true,
    "conv_channels": 128,
    "conv_kernel": 5,
    "conv_layers": 2,
    "device": "auto",
    "dropout": 0.2,
    "dt_ms": 33.333,
    "epochs": 20,
    "include_optflow": true,
    "include_pose": true,
    "lr": 0.0003,
    "lstm_hidden": 256,
    "lstm_layers": 1,
    "manifest_csv": "data/processed/manifest.csv",
    "num_workers": 0,
    "out_dir": "runs",
    "run_name": "",
    "seed": 0,
    "use_landmarks": true,
    "weight_decay": 0.01
  }
}
```

## Entraînement
![Training curves](C:/Users/moura/Documents/Studies/TP/Flot_Optique/Dynamic Object Motion Analysis/runs/classify_20260305-211230/training_curves.png)

## Résultats (test)
- **Accuracy**: 0.7527950310559006
- **Macro‑F1**: 0.6944308669880951
- **Micro‑F1**: 0.7527950310559006

![Confusion matrix](C:/Users/moura/Documents/Studies/TP/Flot_Optique/Dynamic Object Motion Analysis/runs/classify_20260305-211230/confusion_matrix.png)

### Détails par classe
```json
{
  "B0A": {
    "f1-score": 0.9355432780847146,
    "precision": 0.9136690647482014,
    "recall": 0.9584905660377359,
    "support": 265.0
  },
  "B0B": {
    "f1-score": 0.9545454545454546,
    "precision": 0.9545454545454546,
    "recall": 0.9545454545454546,
    "support": 264.0
  },
  "D0X": {
    "f1-score": 0.6961869618696187,
    "precision": 0.930921052631579,
    "recall": 0.555992141453831,
    "support": 509.0
  },
  "G01": {
    "f1-score": 0.4375,
    "precision": 0.32407407407407407,
    "recall": 0.6730769230769231,
    "support": 52.0
  },
  "G02": {
    "f1-score": 0.5979381443298969,
    "precision": 0.6444444444444445,
    "recall": 0.5576923076923077,
    "support": 52.0
  },
  "G03": {
    "f1-score": 0.6226415094339622,
    "precision": 0.6111111111111112,
    "recall": 0.6346153846153846,
    "support": 52.0
  },
  "G04": {
    "f1-score": 0.7777777777777778,
    "precision": 0.75,
    "recall": 0.8076923076923077,
    "support": 52.0
  },
  "G05": {
    "f1-score": 0.6929133858267716,
    "precision": 0.5866666666666667,
    "recall": 0.8461538461538461,
    "support": 52.0
  },
  "G06": {
    "f1-score": 0.7049180327868853,
    "precision": 0.6142857142857143,
    "recall": 0.8269230769230769,
    "support": 52.0
  },
  "G07": {
    "f1-score": 0.7627118644067796,
    "precision": 0.6818181818181818,
    "recall": 0.8653846153846154,
    "support": 52.0
  },
  "G08": {
    "f1-score": 0.6,
    "precision": 0.625,
    "recall": 0.5769230769230769,
    "support": 52.0
  },
  "G09": {
    "f1-score": 0.7155963302752294,
    "precision": 0.6842105263157895,
    "recall": 0.75,
    "support": 52.0
  },
  "G10": {
    "f1-score": 0.7894736842105263,
    "precision": 0.7258064516129032,
    "recall": 0.8653846153846154,
    "support": 52.0
  },
  "G11": {
    "f1-score": 0.4342857142857143,
    "precision": 0.3089430894308943,
    "recall": 0.7307692307692307,
    "support": 52.0
  },
  "accuracy": 0.7527950310559006,
  "macro avg": {
    "f1-score": 0.6944308669880951,
    "precision": 0.6682497022632153,
    "recall": 0.7574031104751716,
    "support": 1610.0
  },
  "weighted avg": {
    "f1-score": 0.761079172259273,
    "precision": 0.8129763052517116,
    "recall": 0.7527950310559006,
    "support": 1610.0
  }
}
```

## Études d’ablation (à compléter)
- **Sans accélération**: désactiver `track_acc_xyz`
- **Sans flot optique**: `--no-optflow`
- **Sans landmarks**: `--no-landmarks`

Chaque ablation doit être relancée avec le même seed et comparée via Accuracy + Macro‑F1.

## Reproductibilité
- Commande type:

```bash
poetry install -E train -E dataset -E hand
poetry run doma-train train --manifest data/processed/manifest.csv --epochs 20 --batch 32
```
