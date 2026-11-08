"""Camera model vs Intel's formulas (flat walls) and the datasheet fill rate."""

from pathlib import Path

import numpy as np
import pytest
import yaml

from camera_model import simulate_depth, simulate_frame
from dataset import RENDER_DIR, load_frame

CONFIG = yaml.safe_load(Path("configs/default.yaml").read_text())
CAM, MODEL = CONFIG["camera"], CONFIG["camera_model"]
W, H = CAM["width"], CAM["height"]
FX = (W / 2) / np.tan(np.deg2rad(CAM["hfov_deg"]) / 2)
K = np.array([[FX, 0, W / 2], [0, FX, H / 2], [0, 0, 1]])
FB = FX * CAM["baseline_m"]


def wall(z):
    return np.full((H, W), z, dtype=np.float32)


def wall_with_box(z_wall=0.6, z_box=0.4, cols=(400, 500)):
    depth = wall(z_wall)
    depth[:, cols[0] : cols[1]] = z_box
    return depth


def run(depth, effects):
    return simulate_depth(depth, K, CAM, MODEL, seed=0, enabled=effects)


@pytest.mark.parametrize("z", [0.3, 0.5, 0.7])
def test_noise_rms_matches_intel_formula(z):
    """[BKM] 3e: RMS error = Z^2 * subpixel / (f * B)."""
    out = run(wall(z), {"noise"})
    expected = z**2 * CAM["subpixel"] / FB
    assert np.std(out - z) == pytest.approx(expected, rel=0.05)


@pytest.mark.parametrize("z", [0.3, 0.5, 0.7])
def test_left_band_width_matches_datasheet(z):
    """[DS] 4.5: invalid band width in pixels = B * F / Z."""
    invalid_cols = int((run(wall(z), {"left_band"})[0] == 0).sum())
    assert invalid_cols == pytest.approx(FB / z, abs=1)


def test_min_depth_invalidates_close_range():
    assert (run(wall(0.15), {"min_depth"}) == 0).all()


def test_quantization_step():
    assert np.allclose(run(wall(0.5004), {"quantization"}), 0.500)


def test_stereo_shadow_left_of_closer_object():
    """The strip hidden from the right imager is f*B*(1/z_box - 1/z_wall) wide and
    lies directly left of the box, never right of it."""
    row = run(wall_with_box(), {"stereo_shadow"})[0]
    shadow = np.flatnonzero(row == 0)
    assert shadow.size == pytest.approx(FB * (1 / 0.4 - 1 / 0.6), abs=1.5)
    assert shadow.max() < 400 and shadow.min() > 300


def test_edges_only_near_depth_jumps():
    row = run(wall_with_box(), {"edges"})[0]
    invalid = set(np.flatnonzero(row == 0))
    near_jumps = set(range(397, 403)) | set(range(497, 503))
    assert invalid and invalid <= near_jumps


def test_effects_use_independent_random_streams():
    """Switching specular holes on or off must not change the noise pattern, otherwise
    ablation conditions would differ in more than one effect."""
    depth = wall(0.5)
    normals = np.zeros((H, W, 3))
    normals[..., 2] = -1.0
    metal = np.zeros((H, W), dtype=bool)
    common = dict(normals=normals, metal=metal, roughness=0.25)
    a = simulate_depth(depth, K, CAM, MODEL, 0, {"noise"}, **common)
    b = simulate_depth(depth, K, CAM, MODEL, 0, {"noise", "specular"}, **common)
    assert np.array_equal(a, b)


@pytest.mark.skipif(
    not (RENDER_DIR / "scene_000" / "meta.json").exists(), reason="render scene 000 first"
)
def test_fill_rate_table_vs_metal():
    """[DS] Table 4-14: fill rate >= 99% holds for matte targets, so the table must keep
    it under specular dropout, while aluminium loses a clear share of its pixels."""
    frame = load_frame(0, 0)
    out = simulate_frame(frame, CAM, MODEL, enabled={"specular"})
    table, metal = frame.category == 0, frame.category > 0
    assert (out[table] > 0).mean() >= 0.99
    assert 0.2 < (out[metal] > 0).mean() < 0.95
