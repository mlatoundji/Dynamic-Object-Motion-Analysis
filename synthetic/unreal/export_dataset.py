"""
Unreal Engine 5 Editor Python script (skeleton).

This file is not directly runnable from standard Python.
Run inside UE Editor with the Python plugin enabled.

Goal:
- Randomize camera & lighting (domain randomization)
- Render frames via Movie Render Queue
- Export per-frame bone transforms for the hand rig (zero-noise GT)
- Export segmentation masks via CustomDepth/Stencil

You will need to adapt object/asset names to your project.
"""

# Pseudo-code outline (adapt to UE Python API):
#
# import unreal
#
# OUT_DIR = "D:/data/raw/synthetic_unreal"
# NUM_SEQUENCES = 100
# FRAMES = 90
# FPS = 30
#
# def randomize_camera(camera_actor):
#     # set location/rotation/FOV
#     pass
#
# def export_bones(skeletal_mesh_actor, frame_idx, out_path):
#     # read bone transforms (component/world) and write json/csv
#     pass
#
# def setup_mrq(sequence, out_dir):
#     # configure Movie Render Queue job (output settings, passes)
#     pass
#
# for seq_idx in range(NUM_SEQUENCES):
#     # 1) Spawn/locate camera & hand actor(s)
#     # 2) Randomize parameters (camera, light, materials)
#     # 3) Play animation / set sequence range
#     # 4) Render with MRQ to OUT_DIR/seq_xxxxxx/rgb/
#     # 5) Export masks (CustomDepth/Stencil) to OUT_DIR/seq_xxxxxx/mask/
#     # 6) Export bone transforms per frame to OUT_DIR/seq_xxxxxx/joints/
#     pass

