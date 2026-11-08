import argparse
import csv
import json
from pathlib import Path

import numpy as np
import open3d as o3d
import trimesh
import yaml
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from camera_model import simulate_frame
from dataset import RENDER_DIR, backproject, load_frame

CONFIG = yaml.safe_load(Path("configs/default.yaml").read_text())
CAM, MODEL, EXP = CONFIG["camera"], CONFIG["camera_model"], CONFIG["experiment"]
MESH_DIR = Path("data/meshes")
OUT_DIR = Path("output/experiment")


def conditions():
    full = set(MODEL["effects"])
    conds = {"ideal": None, "full": full}

    for effect in MODEL["effects"]:
        conds[f"only_{effect}"] = {effect}
        conds[f"without_{effect}"] = full - {effect}

    return conds


def load_model(mesh_name):
    mesh = trimesh.load(MESH_DIR / f"{mesh_name}.obj", force="mesh")
    points, faces = trimesh.sample.sample_surface(mesh, EXP["model_points"], seed=EXP["seed"])
    normals = mesh.face_normals[faces]
    eval_points = points[: EXP["eval_points"]]

    return points, normals, eval_points, float(np.linalg.norm(mesh.extents))


def visible_side(points, normals, T, margin_deg):
    """Back-face culling: keep model points facing the camera under pose T.
    Faces up to margin_deg past edge-on are kept. Never uses the true pose."""
    p_cam = points @ T[:3, :3].T + T[:3, 3]
    n_cam = normals @ T[:3, :3].T
    to_camera = -p_cam / np.linalg.norm(p_cam, axis=1, keepdims=True)
    facing = (n_cam * to_camera).sum(axis=1) > -np.sin(np.deg2rad(margin_deg))
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points[facing]))
    cloud.normals = o3d.utility.Vector3dVector(normals[facing])
    return cloud


def perturb(T, rng, rot_deg, trans_m):
    axis = rng.normal(size=3)
    rotation = Rotation.from_rotvec(np.deg2rad(rot_deg) * axis / np.linalg.norm(axis))
    shift = rng.normal(size=3)
    T_init = T.copy()
    T_init[:3, :3] = rotation.as_matrix() @ T[:3, :3]
    T_init[:3, 3] += trans_m * shift / np.linalg.norm(shift)

    return T_init


def add_s(points, T_est, T_gt):
    gt = points @ T_gt[:3, :3].T + T_gt[:3, 3]
    est = points @ T_est[:3, :3].T + T_est[:3, 3]
    return float(cKDTree(est).query(gt)[0].mean())


# 180 deg about the diagonal between the legs maps the L-bracket onto itself:
# (x, y, z) -> (y, x, -z).
BRACKET_SYMMETRY = Rotation.from_rotvec(np.pi * np.array([1, 1, 0]) / np.sqrt(2)).as_matrix()


def rotation_error_deg(R_est, R_gt, symmetries=(np.eye(3), BRACKET_SYMMETRY)):
    """Smallest rotation between the estimate and any pose that looks the same."""
    return min(np.degrees(Rotation.from_matrix(R_est.T @ R_gt @ S).magnitude()) for S in symmetries)


def run_icp(points_cam, model_cloud, T_init, max_corr_m):
    source = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points_cam))
    result = o3d.pipelines.registration.registration_icp(
        source,
        model_cloud,
        max_corr_m,
        np.linalg.inv(T_init),
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=EXP["icp_max_iter"]),
    )
    return np.linalg.inv(result.transformation), result.fitness


def evaluate_frame(scene_id, view, meta, conds, models, init):
    frame = load_frame(scene_id, view)
    world2cam = np.linalg.inv(frame.cam2world)
    name_to_id = {name: i for i, name in frame.names.items()}

    # one depth image (as 3D points + valid mask) per condition, shared by all brackets.
    observations = {}
    for cond, effects in conds.items():
        depth = frame.depth if effects is None else simulate_frame(frame, CAM, MODEL, effects)
        observations[cond] = (backproject(depth, frame.K), depth > 0)

    rows = []
    for obj in meta["objects"]:
        if (
            obj["mesh"] != "bracket"
            or not obj["on_table"]
            or meta["visible_px"][view][obj["name"]] < EXP["min_visible_px"]
        ):
            continue
        model_points, model_normals, eval_points, diameter = models[obj["mesh"]]
        T_gt = world2cam @ np.array(obj["local2world"])  # object to camera (OpenCV)
        inst = name_to_id[obj["name"]]
        rng = np.random.default_rng([EXP["seed"], scene_id, view, inst])
        # The same starting guess for every condition.
        T_init = perturb(T_gt, rng, init["rot_deg"], init["trans_m"])
        mask = frame.instance == inst

        for cond, (cam_points, valid) in observations.items():
            observed = cam_points[mask & valid]
            row = {
                "scene": scene_id,
                "view": view,
                "object": obj["name"],
                "condition": cond,
                "n_points": len(observed),
                "distance_m": float(np.linalg.norm(T_gt[:3, 3])),
                "roughness": meta["alu_roughness"],
                "init_add_s_mm": 1000 * add_s(eval_points, T_init, T_gt),
            }
            if len(observed) < EXP["min_points"]:
                row.update(
                    add_s_mm=np.nan,
                    t_err_mm=np.nan,
                    r_err_deg=np.nan,
                    t_err_x_mm=np.nan,
                    t_err_y_mm=np.nan,
                    t_err_z_mm=np.nan,
                    fitness=0.0,
                    success=False,
                    strict_success=False,
                    failure="too_few_points",
                    T_est="",
                )
            else:
                # re-cull with the current estimate before each round
                T_est = T_init
                for i in range(EXP["icp_rounds"]):
                    margin = init["rot_deg"] if i == 0 else EXP["cull_margin_deg"]
                    target = visible_side(model_points, model_normals, T_est, margin)
                    T_est, fitness = run_icp(observed, target, T_est, init["max_corr_m"])
                err = add_s(eval_points, T_est, T_gt)
                success = err < EXP["success_add_s_frac"] * diameter
                r_err = rotation_error_deg(T_est[:3, :3], T_gt[:3, :3])
                t_err = T_est[:3, 3] - T_gt[:3, 3]
                # translation error in the bracket's own axes (z = extrusion direction)
                t_err_obj = 1000 * np.abs(T_gt[:3, :3].T @ t_err)
                strict = (
                    1000 * np.linalg.norm(t_err) < EXP["strict_t_mm"]
                    and r_err < EXP["strict_r_deg"]
                )
                row.update(
                    add_s_mm=1000 * err,
                    t_err_mm=1000 * float(np.linalg.norm(t_err)),
                    r_err_deg=float(r_err),
                    t_err_x_mm=float(t_err_obj[0]),
                    t_err_y_mm=float(t_err_obj[1]),
                    t_err_z_mm=float(t_err_obj[2]),
                    fitness=fitness,
                    success=bool(success),
                    strict_success=bool(strict),
                    failure="" if success else "icp",
                    # saved so any metric can be recomputed later without rerunning ICP.
                    T_est=json.dumps(np.round(T_est, 6).tolist()),
                )
            rows.append(row)
    return rows


def summarize(rows, conds):
    summary = {}
    for cond in conds:
        r = [x for x in rows if x["condition"] == cond]
        ran = [x for x in r if x["failure"] != "too_few_points"]

        def stat(key, q, ran=ran):
            values = np.array([x[key] for x in ran])
            return float(np.percentile(values, q)) if values.size else float("nan")

        summary[cond] = {
            "views": len(r),
            "success_rate": float(np.mean([x["success"] for x in r])),
            "strict_success_rate": float(np.mean([x["strict_success"] for x in r])),
            "too_few_points": sum(x["failure"] == "too_few_points" for x in r),
            "median_points": float(np.median([x["n_points"] for x in r])),
            **{
                f"{name}_{key}": stat(key, q)
                for key in ("add_s_mm", "t_err_mm", "r_err_deg", "t_err_z_mm")
                for name, q in (("median", 50), ("p90", 90))
            },
        }
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenes", type=int, default=None, help="only the first N scenes")
    parser.add_argument("--init-rot-deg", type=float, default=EXP["init_rot_deg"])
    parser.add_argument("--init-trans-m", type=float, default=EXP["init_trans_m"])
    parser.add_argument("--conditions", default=None, help="comma-separated subset")
    parser.add_argument("--tag", default="main", help="name of the output files")
    parser.add_argument(
        "--noise-corr-px",
        type=float,
        default=None,
        help="override camera_model.noise_corr_px (sensitivity to this assumption)",
    )
    args = parser.parse_args()

    if args.noise_corr_px is not None:
        MODEL["noise_corr_px"] = args.noise_corr_px  # read by simulate_frame via MODEL

    conds = conditions()
    if args.conditions:
        conds = {name: conds[name] for name in args.conditions.split(",")}

    # search radius has to reach past the starting error
    init = {
        "rot_deg": args.init_rot_deg,
        "trans_m": args.init_trans_m,
        "max_corr_m": max(EXP["icp_max_corr_m"], 1.5 * args.init_trans_m),
    }

    models = {"bracket": load_model("bracket")}
    meta_paths = sorted(RENDER_DIR.glob("scene_*/meta.json"))[: args.scenes]

    rows = []
    for i, path in enumerate(meta_paths, start=1):
        meta = json.loads(path.read_text())
        for view in range(len(meta["cam2world_blender"])):
            rows += evaluate_frame(meta["scene_id"], view, meta, conds, models, init)
        print(f"[{i}/{len(meta_paths)}] scene {meta['scene_id']:03d}: {len(rows)} rows")

    for row in rows:
        row.update(
            init_rot_deg=init["rot_deg"],
            init_trans_m=init["trans_m"],
            noise_corr_px=MODEL["noise_corr_px"],
        )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / f"results_{args.tag}.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = summarize(rows, conds)
    (OUT_DIR / f"summary_{args.tag}.json").write_text(
        json.dumps(
            {"init": init, "noise_corr_px": MODEL["noise_corr_px"], "conditions": summary},
            indent=2,
        )
    )

    print(
        f"\ninit: {init['rot_deg']:.0f} deg, {100 * init['trans_m']:.0f} cm\n"
        f"{'condition':24s} {'views':>5s} {'strict':>7s} {'few':>4s} {'t med':>6s} "
        f"{'t p90':>6s} {'tz med':>7s} {'r med':>6s} {'r p90':>6s} {'points':>7s}"
    )
    for cond, s in summary.items():
        print(
            f"{cond:24s} {s['views']:5d} {s['strict_success_rate']:7.1%} "
            f"{s['too_few_points']:4d} {s['median_t_err_mm']:6.2f} {s['p90_t_err_mm']:6.2f} "
            f"{s['median_t_err_z_mm']:7.2f} {s['median_r_err_deg']:6.2f} "
            f"{s['p90_r_err_deg']:6.2f} {s['median_points']:7.0f}"
        )


if __name__ == "__main__":
    main()
