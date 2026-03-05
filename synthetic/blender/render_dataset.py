from __future__ import annotations

# Blender Python script. Run via:
# blender -b template.blend -P synthetic/blender/render_dataset.py -- --out ... --num_sequences ...

import argparse
import json
import math
import random
from pathlib import Path

import bpy
import mathutils


def _parse_args() -> argparse.Namespace:
    argv = []
    if "--" in bpy.app.argv:
        argv = bpy.app.argv[bpy.app.argv.index("--") + 1 :]

    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--num_sequences", type=int, default=10)
    p.add_argument("--frames", type=int, default=90)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--armature", default="HandArmature")
    p.add_argument("--mesh", default="HandMesh")
    p.add_argument("--img_size", type=int, default=512)
    p.add_argument("--cam_dist_min", type=float, default=0.6)
    p.add_argument("--cam_dist_max", type=float, default=1.6)
    p.add_argument("--cam_elev_min_deg", type=float, default=-10.0)
    p.add_argument("--cam_elev_max_deg", type=float, default=75.0)
    p.add_argument("--cam_fov_min_deg", type=float, default=35.0)
    p.add_argument("--cam_fov_max_deg", type=float, default=85.0)
    return p.parse_args(argv)


def _ensure_camera() -> bpy.types.Object:
    cam = bpy.data.objects.get("Camera")
    if cam and cam.type == "CAMERA":
        return cam
    bpy.ops.object.camera_add()
    cam = bpy.context.active_object
    cam.name = "Camera"
    return cam


def _ensure_light() -> bpy.types.Object:
    light = bpy.data.objects.get("KeyLight")
    if light and light.type == "LIGHT":
        return light
    bpy.ops.object.light_add(type="AREA")
    light = bpy.context.active_object
    light.name = "KeyLight"
    return light


def _look_at(obj: bpy.types.Object, target: mathutils.Vector) -> None:
    direction = target - obj.location
    rot_quat = direction.to_track_quat("-Z", "Y")
    obj.rotation_euler = rot_quat.to_euler()


def _setup_compositor_for_mask(mesh_obj: bpy.types.Object, out_dir: Path) -> None:
    scene = bpy.context.scene
    scene.use_nodes = True
    tree = scene.node_tree
    tree.nodes.clear()

    view_layer = scene.view_layers["ViewLayer"]
    view_layer.use_pass_object_index = True
    mesh_obj.pass_index = 1

    rl = tree.nodes.new("CompositorNodeRLayers")
    idmask = tree.nodes.new("CompositorNodeIDMask")
    idmask.index = 1
    idmask.use_antialiasing = True

    file_rgb = tree.nodes.new("CompositorNodeOutputFile")
    file_rgb.label = "RGB_OUT"
    file_rgb.base_path = str((out_dir / "rgb").resolve())
    file_rgb.format.file_format = "PNG"

    file_mask = tree.nodes.new("CompositorNodeOutputFile")
    file_mask.label = "MASK_OUT"
    file_mask.base_path = str((out_dir / "mask").resolve())
    file_mask.format.file_format = "PNG"
    file_mask.format.color_mode = "BW"

    tree.links.new(rl.outputs["Image"], file_rgb.inputs["Image"])
    tree.links.new(rl.outputs["IndexOB"], idmask.inputs["ID value"])
    tree.links.new(idmask.outputs["Alpha"], file_mask.inputs["Image"])

    # Use consistent naming; Blender appends frame number automatically.
    file_rgb.file_slots[0].path = ""
    file_mask.file_slots[0].path = ""


def _export_camera(camera: bpy.types.Object) -> dict:
    cam_data = camera.data
    return {
        "location": list(camera.location),
        "rotation_euler": list(camera.rotation_euler),
        "lens_unit": cam_data.lens_unit,
        "angle": float(cam_data.angle),
        "clip_start": float(cam_data.clip_start),
        "clip_end": float(cam_data.clip_end),
    }


def _export_joints(armature: bpy.types.Object) -> dict:
    bones = []
    for b in armature.pose.bones:
        m = armature.matrix_world @ b.matrix
        head = armature.matrix_world @ b.head
        tail = armature.matrix_world @ b.tail
        bones.append(
            {
                "name": b.name,
                "head_world": [float(head.x), float(head.y), float(head.z)],
                "tail_world": [float(tail.x), float(tail.y), float(tail.z)],
                "matrix_world": [float(v) for row in m for v in row],
            }
        )
    return {"bones": bones}


def main() -> None:
    args = _parse_args()
    random.seed(int(args.seed))

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    scene = bpy.context.scene
    scene.render.resolution_x = int(args.img_size)
    scene.render.resolution_y = int(args.img_size)
    scene.render.fps = int(args.fps)

    cam = _ensure_camera()
    light = _ensure_light()

    arm = bpy.data.objects.get(args.armature)
    mesh = bpy.data.objects.get(args.mesh)
    if arm is None or arm.type != "ARMATURE":
        raise RuntimeError(f"Armature not found or wrong type: {args.armature}")
    if mesh is None:
        raise RuntimeError(f"Mesh not found: {args.mesh}")

    target = mathutils.Vector((0.0, 0.0, 0.0))

    for seq in range(int(args.num_sequences)):
        seq_dir = out_root / f"seq_{seq:06d}"
        (seq_dir / "rgb").mkdir(parents=True, exist_ok=True)
        (seq_dir / "mask").mkdir(parents=True, exist_ok=True)
        (seq_dir / "joints").mkdir(parents=True, exist_ok=True)

        # Domain randomization
        dist = random.uniform(float(args.cam_dist_min), float(args.cam_dist_max))
        elev = math.radians(random.uniform(float(args.cam_elev_min_deg), float(args.cam_elev_max_deg)))
        az = math.radians(random.uniform(0.0, 360.0))
        cam.location = mathutils.Vector(
            (
                dist * math.cos(elev) * math.cos(az),
                dist * math.cos(elev) * math.sin(az),
                dist * math.sin(elev),
            )
        )
        _look_at(cam, target)

        cam.data.angle = math.radians(random.uniform(float(args.cam_fov_min_deg), float(args.cam_fov_max_deg)))
        light.location = cam.location + mathutils.Vector((0.2, 0.2, 0.2))
        light.data.energy = random.uniform(200.0, 2500.0)

        _setup_compositor_for_mask(mesh, seq_dir)

        (seq_dir / "camera.json").write_text(json.dumps(_export_camera(cam), indent=2), encoding="utf-8")

        scene.frame_start = 0
        scene.frame_end = int(args.frames) - 1

        for frame in range(scene.frame_start, scene.frame_end + 1):
            scene.frame_set(frame)
            bpy.ops.render.render(write_still=False)

            j = _export_joints(arm)
            (seq_dir / "joints" / f"{frame:04d}.json").write_text(json.dumps(j), encoding="utf-8")


if __name__ == "__main__":
    main()

