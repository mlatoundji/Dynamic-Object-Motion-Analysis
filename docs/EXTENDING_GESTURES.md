## Étendre le modèle à de nouveaux gestes (lettres, chiffres, formes)

Ce document décrit une stratégie concrète pour ajouter de nouveaux gestes au système (dataset → entraînement → PoC), en restant aligné sur le pipeline DOMA:

- `pose_tensor.npz` (pos/vel/acc + landmarks + valid)
- `optflow_features.npz` (stats de flot dans ROI main + valid)
- `manifest.csv` (index global)

### Objectif cible (classification discrète)

On vise d’abord une tâche **fermée** (classes finies): ex. lettres A–Z, chiffres 0–9, formes (cercle, carré, triangle), ou intentions (swipe left/right, zoom, click).

### 1) Collecte des données (réel)

- **Capteur**: webcam RGB (30 fps idéal).
- **Diversité**:
  - sujets multiples (morphologies/peaux)
  - environnements multiples (lumière, fond, distance caméra)
  - variations d’exécution (vitesse, amplitude, orientation de main)
- **Consignes**:
  - définir une *grammaire* claire de chaque geste (début/fin, main dominante, doigt utilisé)
  - intégrer des séquences *non‑gestes* pour enrichir `D0X` (mouvements parasites).

### 2) Annotation / segmentation

Le point critique est de produire **1 sample = 1 geste** (segment temporel), comme IPN Hand.

Options:
- **Annotation par segments**: (recommandé) marquer `t_start/t_end` (frames inclusives) + label.
- **Annotation par clip isolé**: chaque vidéo contient un seul geste (plus simple, moins flexible).

À l’issue, on vise un `index.csv` ou des annotations converties vers un `manifest.csv` cohérent.

### 3) Génération des artefacts DOMA

Une fois les vidéos segmentées, on réutilise votre pipeline existant:
- extraction MediaPipe (wrist + 21 landmarks)
- normalisation spatiale (origin) + rotationnelle (alignement)
- dérivées (vel/acc) sur une grille temporelle régulière
- features optflow (ROI main) + validité

### 4) Entraînement (fine‑tuning vs from‑scratch)

#### Option A — Fine‑tuning CNN‑LSTM (recommandé)

Si vous restez sur des gestes “tracés”/trajectoires (lettres, formes), le CNN‑LSTM sur tenseurs cinématiques est très adapté:
- rapide à entraîner
- robuste aux backgrounds (car on n’utilise pas directement les pixels RGB)
- interprétable via ablations (pos vs vel vs acc vs optflow)

Approche:
- ré‑entraîner la tête de classification (nouveau `num_classes`)
- ré‑entraîner tout le réseau si le nouveau domaine diffère fortement
- conserver une classe `D0X` comme “idle”/non‑intention

#### Option B — Modèle RGB (ResNet‑50 / ResNeXt‑101 + tête temporelle)

Préférable si:
- la **configuration de la main** (handshape) est discriminante et les landmarks sont instables
- on veut capturer des détails d’apparence (occlusions, doigts fins, textures)

Dans ce cas, la recette courante est:
- backbone 2D (ResNet/ResNeXt) par frame
- agrégation temporelle (LSTM/Transformer/Temporal Conv) sur la séquence d’embeddings

Coût:
- plus lourd en calcul (GPU recommandé)
- plus sensible à la diversité des backgrounds

#### Option C — “Optical Flow model” (RAFT / FlowFormer) en tant que classifieur

À privilégier **comme module de perception** (extraction de mouvement robuste), pas comme classifieur final:
- le flot dense peut être utilisé pour produire de meilleurs features (ou stabiliser le tracking)
- la classification finale reste plus simple sur vecteurs (pose/kinematics) ou sur embeddings RGB

### 5) Stratégie “données” (réel + synthétique)

Le synthétique est surtout utile pour:
- angles caméra rares
- occlusions contrôlées
- vérité terrain parfaite (traj + masque)

Mais il faut toujours recaler sur le réel (fine‑tuning) pour éviter un gap de domaine.

### 6) Critères de décision (quel modèle choisir ?)

- **Gestes = trajectoires (formes/lettres)** → CNN‑LSTM/Transformer léger sur `pose_tensor` (souvent suffisant).
- **Gestes = handshape statique (alphabet manuel) + détails doigts** → RGB model (ResNet/ResNeXt) ou squelettique enrichi (landmarks + graph conv).
- **Contrainte FPS strict CPU** → CNN‑LSTM sur vecteurs (le plus léger).
- **Occlusions sévères / tracking instable** → améliorer d’abord perception (ROI + flow + re‑ID), puis re‑entraîner.

### Faisabilité

Oui, la tâche est faisable. Le point dur n’est pas tant l’architecture que:
- la définition d’une taxonomie de gestes non ambiguë
- la segmentation/annotation fiable
- la diversité de données (éviter sur‑apprentissage à un seul décor/sujet)

Pour IPN Hand en particulier, la source officielle contient déjà des segments annotés et des labels standards: voir la page officielle IPN Hand: `https://gibranbenitez.github.io/IPN_Hand/`.

