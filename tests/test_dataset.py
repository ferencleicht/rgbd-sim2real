"""The table is the world plane z = 0 with normal +z, so it checks K, the depth and the
camera pose conversion all at once."""

import numpy as np
import pytest

from dataset import RENDER_DIR, backproject, load_frame

pytestmark = pytest.mark.skipif(
    not (RENDER_DIR / "scene_000" / "meta.json").exists(), reason="render scene 000 first"
)


def table_mask(frame):
    table_ids = [i for i, name in frame.names.items() if name == "Plane"]
    return np.isin(frame.instance, table_ids)


@pytest.mark.parametrize("view", [0, 1, 2])
def test_table_points_lie_on_world_plane(view):
    frame = load_frame(0, view)
    points = backproject(frame.depth, frame.K)[table_mask(frame)]
    world = points @ frame.cam2world[:3, :3].T + frame.cam2world[:3, 3]

    assert np.abs(world[:, 2]).max() < 0.002, "table points are not on z = 0"


@pytest.mark.parametrize("view", [0, 1, 2])
def test_table_normals_match_camera_pose(view):
    frame = load_frame(0, view)
    measured = np.median(frame.normals[table_mask(frame)], axis=0)
    expected = frame.cam2world[:3, :3].T @ np.array([0.0, 0.0, 1.0])  # world +z in camera frame
    cos = measured @ expected / np.linalg.norm(measured)
    assert np.degrees(np.arccos(np.clip(cos, -1, 1))) < 1.0, "normals decoded incorrectly"
