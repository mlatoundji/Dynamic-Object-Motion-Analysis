import json
import os

# Liste des fichiers JSON
list_files = [
    "models/attention_20260308_111806/attention_20260308_111806.json",
    "models/default_20260308_102325/default_20260308_102325.json",
    "models/lstm_20260308_121906/lstm_20260308_121906.json",
    "models/lstm_attention_20260307_213259/lstm_attention_20260307_213259.json",
    "models/lstm_gated_20260308_132658/lstm_gated_20260308_132658.json",
    "models/stgcn_20260307_185936/stgcn_20260307_185936.json",
    "models/stgcn_20260310_011351/stgcn_20260310_011351.json",
]

for file_path in list_files:
    # Charger le JSON
    with open(file_path, "r") as f:
        data = json.load(f)
    
    history = data.get("history", [])
    if not history:
        print(f"Aucun historique dans {file_path}")
        continue
    
    # Trouver l'indice du meilleur accuracy
    best_epoch = max(range(len(history)), key=lambda i: history[i]["accuracy"])
    best_metrics = history[best_epoch]
    
    # Construire le dictionnaire final
    output = {
        "best_epoch": best_epoch,
        "best_metric_name": "accuracy",
        "best_metric_value": best_metrics["accuracy"],
        "best_metrics": best_metrics
    }
    
    # Déterminer le dossier et sauvegarder metrics.json
    folder = os.path.dirname(file_path)
    output_path = os.path.join(folder, "metrics.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"Metrics sauvegardées dans {output_path}")