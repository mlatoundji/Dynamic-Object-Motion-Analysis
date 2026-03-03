## Unreal Engine synthetic generator (squelette)

Objectif: produire des séquences synthétiques photoréalistes avec **angles caméra complexes** et exporter simultanément:
- RGB (frames / vidéo)
- segmentation (CustomDepth/Stencil)
- ground-truth articulations (bones transforms)

### Prérequis (UE5)

- Unreal Engine 5.x installé
- Plugin **Python Editor Script** activé
- Movie Render Queue (MRQ) activé
- (Optionnel) Plugin/outil de capture de segmentation via CustomDepth + Stencil

### Stratégie recommandée

- **Animation**: Control Rig / Animation Blueprint sur une main riggée
- **Domain randomization**:
  - positions caméras (azimuth/elevation/distance/FOV)
  - éclairage (HDRI, intensités, couleurs)
  - matériaux (peau, gants)
  - motion blur / DOF
- **Export**:
  - MRQ pour frames RGB
  - Passes: CustomDepth/Stencil pour masks
  - Python: exporter transforms bones à chaque frame (CSV/JSON)

### Script Python UE (à adapter)

Le script `synthetic/unreal/export_dataset.py` sert de point de départ: il décrit les étapes et des APIs UE usuelles (MRQ + randomization + export bones).
