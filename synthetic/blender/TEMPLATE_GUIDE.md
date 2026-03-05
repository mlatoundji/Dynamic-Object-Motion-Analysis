## Créer `template.blend`

Le script `render_dataset.py` est volontairement agnostique du rig: il pilote une scène Blender existante.

### Étapes minimales

1. Ouvrir Blender et créer/importer une main riggée (armature + mesh).\n
2. Renommer (recommandé):\n
   - Armature: `HandArmature`\n
   - Mesh: `HandMesh`\n
3. Ajouter une animation (Action) ou une animation sur la timeline:\n
   - au moins `frames` images (ex: 90)\n
4. Sauvegarder le fichier sous `synthetic/blender/template.blend` (non fourni ici car binaire).\n

### Vérifier

Avant exécution headless, vérifier que:\n
- la timeline joue l’animation correctement\n
- la main est visible depuis l’origine (0,0,0)\n
- l’armature est bien de type ARMATURE\n

