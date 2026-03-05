## Blender synthetic dataset generator

Objectif: générer des séquences **RGB + mask** + **ground-truth joints (zero-noise)** avec **domain randomization** (caméra complexe, éclairage, etc.), puis les ingérer dans le même schéma que `doma-build-dataset`.

### Prérequis

- Blender 4.x installé
- Exécution headless disponible (`blender -b ...`)

### Préparer un template `.blend`

Le script attend un `.blend` contenant au minimum:
- Une **armature** (main riggée) animable (actions ou animation sur la timeline)
- Un **mesh** (la main) parenté à l’armature

Conventions recommandées (modifiable via arguments):
- Armature: `HandArmature`
- Mesh: `HandMesh`

### Exécution (headless)

Depuis la racine du repo (exemple):

```bash
blender -b synthetic/blender/template.blend ^
  -P synthetic/blender/render_dataset.py -- ^
  --out data/raw/synthetic_blender ^
  --num_sequences 10 ^
  --frames 90 ^
  --fps 30 ^
  --seed 0 ^
  --armature HandArmature ^
  --mesh HandMesh
```

### Sortie

```
data/raw/synthetic_blender/
  seq_000000/
    rgb/0000.png ...
    mask/0000.png ...
    joints/0000.json ...
    camera.json
```

Vous pouvez ensuite convertir ces séquences en mp4 et construire des tenseurs via `doma-build-dataset` (en indexant ces clips dans `data/raw/synthetic/` au même format `index.csv` que IPN).

