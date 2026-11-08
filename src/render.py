import blenderproc as bproc

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

CONFIG = yaml.safe_load(Path("configs/default.yaml").read_text())
MESH_DIR = Path("data/meshes")
OUT_DIR = Path("data/renders")

MANIFEST = json.loads((MESH_DIR / "manifest.json").read_text())
MESH_NAMES = list(MANIFEST)
MESH_PROBS = [MANIFEST[name]["probability"] for name in MESH_NAMES]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-id", type=int, required=True)
    return parser.parse_args()


def make_materials():
    alu = bproc.material.create("aluminium")
    alu.set_principled_shader_value("Base Color", [0.91, 0.92, 0.92, 1.0])  # light grey
    alu.set_principled_shader_value("Metallic", 1.0)

    table = bproc.material.create("table")
    table.set_principled_shader_value("Base Color", [0.35, 0.35, 0.38, 1.0])  # medium grey
    table.set_principled_shader_value("Roughness", 0.8)

    return alu, table


def setup_camera():
    cam = CONFIG["camera"]
    w, h = cam["width"], cam["height"]
    fx = (w / 2) / np.tan(np.deg2rad(cam["hfov_deg"]) / 2)
    K = np.array([[fx, 0, w / 2], [0, fx, h / 2], [0, 0, 1]])
    bproc.camera.set_intrinsics_from_K_matrix(K, w, h)

    return K


def make_table(material):
    half = CONFIG["scene"]["table_half_size_m"]
    table = bproc.object.create_primitive("PLANE", scale=[half, half, 1])
    table.replace_materials(material)
    table.set_cp("category_id", 0)
    table.enable_rigidbody(active=False, collision_shape="BOX")
    return table


def make_lights():
    lights = []
    for _ in range(CONFIG["scene"]["light_count"]):
        light = bproc.types.Light()
        light.set_type("POINT")
        lights.append(light)
    return lights


def randomize_lights(lights, rng):
    s = CONFIG["scene"]
    for light in lights:
        light.set_location(
            bproc.sampler.shell(
                center=[0, 0, 0],
                radius_min=s["light_radius_min_m"],
                radius_max=s["light_radius_max_m"],
                elevation_min=s["light_elevation_min_deg"],
                elevation_max=s["light_elevation_max_deg"],
            )
        )
        light.set_energy(rng.uniform(s["light_energy_min_w"], s["light_energy_max_w"]))


def load_parts(rng, material):
    s = CONFIG["scene"]
    n_parts = rng.integers(s["n_parts_min"], s["n_parts_max"] + 1)
    parts = []
    for i in range(n_parts):
        mesh_name = str(rng.choice(MESH_NAMES, p=MESH_PROBS))
        obj = bproc.loader.load_obj(str(MESH_DIR / MANIFEST[mesh_name]["file"]))[0]

        dims_mm = np.array(obj.blender_obj.dimensions) * 1000
        assert np.allclose(dims_mm, MANIFEST[mesh_name]["extents_mm"], atol=0.1), (
            f"{mesh_name}: model frame changed on import, dims={dims_mm.round(1)}"
        )

        obj.set_name(f"{mesh_name}_{i}")
        obj.replace_materials(material)
        obj.set_cp("category_id", MANIFEST[mesh_name]["category_id"])
        obj.set_cp("mesh", mesh_name)

        xy = s["spawn_xy_m"]
        z = s["spawn_z_start_m"] + s["spawn_z_step_m"] * i
        obj.set_location([rng.uniform(-xy, xy), rng.uniform(-xy, xy), z])
        obj.set_rotation_euler(rng.uniform(0, 2 * np.pi, size=3))
        obj.enable_rigidbody(
            active=True,
            collision_shape="CONVEX_HULL",
            friction=s["physics_friction"],
            linear_damping=s["physics_linear_damping"],
            angular_damping=s["physics_angular_damping"],
        )
        parts.append(obj)

    return parts


def add_camera_views(parts):
    s = CONFIG["scene"]
    poi = bproc.object.compute_poi(parts)
    poses = []
    for _ in range(s["views_per_scene"]):
        location = bproc.sampler.shell(
            center=poi,
            radius_min=s["cam_radius_min_m"],
            radius_max=s["cam_radius_max_m"],
            elevation_min=s["cam_elevation_min_deg"],
            elevation_max=s["cam_elevation_max_deg"],
        )
        rotation = bproc.camera.rotation_from_forward_vec(poi - location)
        cam2world = bproc.math.build_transformation_mat(location, rotation)
        bproc.camera.add_camera_pose(cam2world)
        poses.append(cam2world)

    return poses


def on_table(obj, margin=0.1):
    """False if a part fell off the table or got launched by the physics."""
    edge = CONFIG["scene"]["table_half_size_m"] - margin
    x, y, z = obj.get_location()
    return bool(abs(x) < edge and abs(y) < edge and -0.01 < z < 0.5)


def visible_pixels(data, parts):
    """Visible pixel count of every part in every view (0 if fully hidden)."""
    names = [p.get_name() for p in parts]
    per_view = []
    for seg, attributes in zip(
        data["instance_segmaps"], data["instance_attribute_maps"], strict=True
    ):
        idx_to_name = {a["idx"]: a["name"] for a in attributes}
        ids, counts = np.unique(seg, return_counts=True)
        px = {idx_to_name.get(int(i)): int(c) for i, c in zip(ids, counts, strict=True)}
        per_view.append({name: px.get(name, 0) for name in names})

    return per_view


def render_scene(scene_id, K, alu, lights):
    s = CONFIG["scene"]
    seed = s["seed"] + scene_id
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    roughness = float(rng.uniform(s["alu_roughness_min"], s["alu_roughness_max"]))
    alu.set_principled_shader_value("Roughness", roughness)
    randomize_lights(lights, rng)
    parts = load_parts(rng, alu)

    bproc.object.simulate_physics_and_fix_final_poses(
        min_simulation_time=s["physics_min_s"],
        max_simulation_time=s["physics_max_s"],
        check_object_interval=1,
    )
    cam_poses = add_camera_views(parts)

    bproc.renderer.enable_depth_output(activate_antialiasing=False)
    bproc.renderer.enable_normals_output()
    bproc.renderer.enable_segmentation_output(map_by=["instance", "category_id", "name"])
    data = bproc.renderer.render()

    # Guard against silently empty outputs.
    assert np.ptp(data["normals"][0]) > 0, f"scene {scene_id}: normals output is constant"
    px_per_view = visible_pixels(data, parts)
    assert any(px > 0 for view in px_per_view for px in view.values()), (
        f"scene {scene_id}: no part appears in the segmentation"
    )

    scene_dir = OUT_DIR / f"scene_{scene_id:03d}"
    bproc.writer.write_hdf5(str(scene_dir), data)

    meta = {
        "scene_id": scene_id,
        "seed": seed,
        "alu_roughness": roughness,
        "K": K.tolist(),
        "width": CONFIG["camera"]["width"],
        "height": CONFIG["camera"]["height"],
        "cam2world_blender": [p.tolist() for p in cam_poses],
        "objects": [
            {
                "name": o.get_name(),
                "mesh": o.get_cp("mesh"),
                "category_id": o.get_cp("category_id"),
                "local2world": o.get_local2world_mat().tolist(),
                "on_table": on_table(o),
            }
            for o in parts
        ],
        "visible_px": px_per_view,
        "config": CONFIG,
    }
    (scene_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    n_brackets = sum(o["mesh"] == "bracket" for o in meta["objects"])
    n_off_table = sum(not o["on_table"] for o in meta["objects"])
    print(
        f"[scene {scene_id:03d}] parts={len(parts)} brackets={n_brackets} "
        f"off_table={n_off_table} roughness={roughness:.2f}"
    )


def main():
    args = parse_args()
    if (OUT_DIR / f"scene_{args.scene_id:03d}" / "meta.json").exists():
        print(f"[scene {args.scene_id:03d}] already rendered, skipping")
        return

    bproc.init()
    K = setup_camera()
    alu, table_mat = make_materials()
    make_table(table_mat)
    lights = make_lights()
    bproc.renderer.set_max_amount_of_samples(CONFIG["render"]["samples"])
    render_scene(args.scene_id, K, alu, lights)


main()
