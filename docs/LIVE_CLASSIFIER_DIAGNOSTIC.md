# Live classifier — diagnostic & réglages (CNN‑LSTM)

Ce document explique **les mismatchs les plus fréquents** entre l’entraînement et l’inférence live, et donne des **réglages recommandés** pour améliorer latence/stabilité/confusions.

## 1) Mismatchs live vs dataset/training (causes racines)

### 1.1 Flip miroir (gauche/droite)
Le live peut appliquer un `cv2.flip(frame, 1)` (vue miroir webcam). Si le dataset a été construit **sans flip**, alors les gestes directionnels peuvent se dégrader:
- `Throw left/right` (swap)
- plus généralement tout ce qui dépend de la direction en x

**Correctif**: rendre le flip configurable et, par défaut, **flip uniquement l’affichage** (HUD), pas les features.

### 1.2 État persistant non réinitialisé (effet “ordre main”)
En live, plusieurs états vivent sur toute la session:
- **pose**: origine/normalisation spatiale (référence “premier poignet”)
- **optflow**: `prev_roi` (image précédente dans la ROI)
- **fenêtre**: buffer temporel de la séquence
- **post‑traitement**: EMA des probabilités

Si on commence par une main puis on passe à l’autre sans reset, la fenêtre contient un mélange de contextes → performances qui semblent “s’inverser”.

**Correctif**: reset automatique lorsque la main est perdue (bbox absente) ou lorsqu’un saut indique un changement de main.

### 1.3 “Drop” des timesteps low‑motion (D0X / clicks)
Avec optflow activé, beaucoup de frames faiblement mobiles peuvent être marquées invalides, et les timesteps invalides sont supprimés avant l’inférence.
Conséquences typiques:
- `D0X` rarement affiché (l’idle est sous‑représenté dans la séquence)
- clicks/double‑click (événements courts) amortis ou perdus

**Correctif** (progressif):
- court terme: conserver les timesteps pose valides même si l’optflow est invalide (features optflow “zéro” + flag)
- moyen terme: aligner training/live en ajoutant une feature `has_motion` et en évitant de supprimer systématiquement les timesteps

## 2) Réglages live recommandés (point de départ)

Les valeurs ci‑dessous sont un **point de départ**; le logging live permet de les optimiser quantitativement.

### 2.1 Pour faible latence (clicks / doubles clicks)
- **Fenêtre**: 500–900 ms
- **Stride** (`infer‑every`): 50–100 ms
- **EMA**: 0.0–0.3 (sinon les pics courts sont “lissés”)
- **Décision**: ajouter un “hold” court (ex: 2–3 inférences) au lieu d’un EMA fort

### 2.2 Pour stabilité (throws / zoom)
- **Fenêtre**: 900–1500 ms
- **Stride**: 100–200 ms
- **EMA**: 0.5–0.7
- **Décision**: hystérésis (entrée/sortie) sur `D0X` et sur les classes gestuelles

### 2.3 Seuils optflow
Si les clicks/idle sont pénalisés, tester:
- `threshold_method=mad` (souvent plus stable en faible motion)
- diminuer `min_pixels`
- ajuster `fixed_threshold` si mode fixed

## 3) Benchmark live reproductible

Objectif: comparer des réglages / modèles sur une entrée identique.

- enregistrer une vidéo “scriptée” (2–3 minutes) avec ordre fixe (inclure des pauses `D0X`)
- rejouer via `--source <video>`
- exploiter les logs pour:
  - latence (début du geste → première prédiction stable)
  - stabilité (nombre de flips)
  - confusion sur segments annotés (annotation unique)

