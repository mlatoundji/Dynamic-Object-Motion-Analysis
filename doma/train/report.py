from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def generate_report(run_dir: Path, *, out_path: Path) -> None:
    """
    Generate a Markdown report from a training run folder.

    Expected files in run_dir:
    - train_config.json
    - model_config.json
    - label_map.json
    - history.json
    - training_curves.png
    - test_metrics.json (optional)
    - confusion_matrix.png (optional)
    """
    run_dir = run_dir.resolve()
    out_path = out_path.resolve()

    train_cfg = _read_json(run_dir / "train_config.json")
    model_cfg = _read_json(run_dir / "model_config.json")
    label_map = _read_json(run_dir / "label_map.json")
    _read_json(run_dir / "history.json")
    test_metrics = _read_json(run_dir / "test_metrics.json")

    labels = []
    l2i = label_map.get("label_to_idx")
    if isinstance(l2i, dict):
        labels = [lab for lab, _ in sorted(((k, int(v)) for k, v in l2i.items()), key=lambda kv: kv[1])]

    # Use relative paths for markdown embedding.
    rel_run = out_path.parent.resolve()

    def rel(p: Path) -> str:
        try:
            return str(p.resolve().relative_to(rel_run)).replace("\\", "/")
        except Exception:
            return str(p.as_posix())

    curves_png = run_dir / "training_curves.png"
    cm_png = run_dir / "confusion_matrix.png"

    basic = {}
    report = {}
    if isinstance(test_metrics.get("basic"), dict):
        basic = test_metrics["basic"]
    if isinstance(test_metrics.get("report"), dict):
        report = test_metrics["report"]

    md = []
    md.append("# Rapport d’entraînement — CNN‑LSTM (IPN Hand)\n")
    md.append(f"- **Run**: `{run_dir.name}`\n")
    md.append(f"- **Artefacts**: `{rel(run_dir)}`\n")
    md.append("\n---\n")

    md.append("## Contexte et objectifs\n")
    md.append(
        "Ce rapport documente l’entraînement d’un classifieur temporel **CNN‑LSTM** "
        "destiné à prédire un label de geste discret (incluant **`D0X` non‑gesture**) "
        "à partir de séries temporelles issues du pipeline DOMA.\n"
    )

    md.append("\n## Données et labels\n")
    md.append("- **Source**: IPN Hand (segments annotés)\n")
    if labels:
        md.append(f"- **Nombre de classes**: {len(labels)}\n")
        md.append("- **Classes**: " + ", ".join(f"`{x}`" for x in labels) + "\n")
    md.append("\n### Index utilisé\n")
    md.append(
        "- `manifest.csv`: "
        f"`{train_cfg.get('manifest_csv', 'data/processed/manifest.csv')}`\n"
    )

    md.append("\n## Représentation d’entrée\n")
    md.append(
        "- **Pose**: `track_pos_xyz` + `track_vel_xyz` + `track_acc_xyz` (et option `landmarks_xyz` aplati)\n"
        "- **Flot optique**: `avg_speed`, `max_speed`, `dominant_angle_deg` (sin/cos), "
        "`direction_concentration`, `n_pixels`, `threshold`\n"
        "- **Masquage**: suppression des timestamps invalides (`valid`) avant padding.\n"
    )

    md.append("\n## Modèle\n")
    md.append("### Architecture\n")
    md.append("- **Conv1D temporel** → **LSTM** → pooling moyen masqué → tête fully-connected (softmax)\n")
    md.append("\n### Hyperparamètres (run)\n")
    md.append("```json\n")
    md.append(
        json.dumps(
            {"train_config": train_cfg, "model_config": model_cfg},
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    md.append("\n```\n")

    md.append("\n## Entraînement\n")
    if curves_png.exists():
        md.append(f"![Training curves]({rel(curves_png)})\n")
    else:
        md.append("- Courbes non disponibles (fichier `training_curves.png` manquant)\n")

    md.append("\n## Résultats (test)\n")
    if basic:
        md.append("- **Accuracy**: " + str(basic.get("accuracy")) + "\n")
        md.append("- **Macro‑F1**: " + str(basic.get("macro_f1")) + "\n")
        md.append("- **Micro‑F1**: " + str(basic.get("micro_f1")) + "\n")
    else:
        md.append("- Résultats test non disponibles (fichier `test_metrics.json` manquant)\n")

    if cm_png.exists():
        md.append(f"\n![Confusion matrix]({rel(cm_png)})\n")

    if report:
        md.append("\n### Détails par classe\n")
        md.append("```json\n")
        md.append(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
        md.append("\n```\n")

    md.append("\n## Études d’ablation (à compléter)\n")
    md.append(
        "- **Sans accélération**: désactiver `track_acc_xyz`\n"
        "- **Sans flot optique**: `--no-optflow`\n"
        "- **Sans landmarks**: `--no-landmarks`\n"
        "\nChaque ablation doit être relancée avec le même seed et comparée via Accuracy + Macro‑F1.\n"
    )

    md.append("\n## Reproductibilité\n")
    md.append(
        "- Commande type:\n\n"
        "```bash\n"
        "poetry install -E train -E dataset -E hand\n"
        "poetry run doma-train train --manifest data/processed/manifest.csv --epochs 20 --batch 32\n"
        "```\n"
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("".join(md), encoding="utf-8")
