"""
Load rendered frames and convert them to OpenCV camera conventions.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

RENDER_DIR = Path("data/renders")
BLENDER_TO_CV = np.array([1.0, -1.0, -1.0])  # flip y and z axes


@dataclass
class Frame:
    scene_id: int
    view: int
    rgb: np.ndarray
    depth: np.ndarray
    normals: np.ndarray
    instance: np.ndarray
    category: np.ndarray
    names: dict
    K: np.ndarray
    cam2world: np.ndarray
    meta: dict


def decode_normals(encoded):
    """BlenderProc stores 0.5 * n + 0.5 in the Blender camera frame"""
    n = (2.0 * encoded - 1.0) * BLENDER_TO_CV
    return n / np.linalg.norm(n, axis=-1, keepdims=True).clip(min=1e-9)


def load_frame(scene_id, view, render_dir=RENDER_DIR):
    scene_dir = Path(render_dir / f"scene_{scene_id:03d}")
    meta = json.loads((scene_dir / "meta.json").read_text())

    with h5py.File(scene_dir / f"{view}.hdf5", "r") as f:
        rgb = np.array(f["colors"])
        depth = np.array(f["depth"])
        normals = decode_normals(np.array(f["normals"]))
        instance = np.array(f["instance_segmaps"])
        category = np.array(f["category_id_segmaps"])
        attributes = json.loads(f["instance_attribute_maps"][()])

    cam2world = np.array(meta["cam2world_blender"][view])
    cam2world[:3, :3] = cam2world[:3, :3] * BLENDER_TO_CV

    return Frame(
        scene_id=scene_id,
        view=view,
        rgb=rgb,
        depth=depth,
        normals=normals,
        instance=instance,
        category=category,
        names={a["idx"]: a["name"] for a in attributes},
        K=np.array(meta["K"]),
        cam2world=cam2world,
        meta=meta,
    )


def backproject(depth, K):
    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    x = (u - K[0, 2]) / K[0, 0] * depth
    y = (v - K[1, 2]) / K[1, 1] * depth

    return np.stack([x, y, depth], axis=-1)
