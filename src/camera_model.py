"""Simulate Intel RealSense D435.

Every effect can be switched on or off in the config. Invalid pixels are 0, like on the
real camera.
"""

import numpy as np
from scipy.ndimage import binary_dilation, gaussian_filter
from scipy.special import ndtr

from dataset import backproject

# masks first, then noise
EFFECTS = (
    "min_depth",
    "left_band",
    "stereo_shadow",
    "edges",
    "specular",
    "noise",
    "quantization",
)


def effect_rng(seed, effect):
    """One RNG per effect, so turning one off doesn't change the others."""
    return np.random.default_rng([*np.atleast_1d(seed), EFFECTS.index(effect)])


def disparity_px(depth, K, baseline):
    return K[0, 0] * baseline / np.maximum(depth, 1e-6)


def min_depth_mask(depth, min_depth):
    """[DS] Table 4-11: no depth closer than Min-Z."""
    return depth >= min_depth


def left_band_mask(depth, K, baseline):
    """[DS] 4.5: no depth if the match at u - f*B/Z is outside the right image."""
    u = np.arange(depth.shape[1])[None, :]
    return u >= disparity_px(depth, K, baseline)


def stereo_shadow_mask(depth, K, baseline, tol_px):
    """No depth where a closer pixel further right covers this one in the right image."""
    u = np.arange(depth.shape[1])[None, :]
    u_right = u - disparity_px(depth, K, baseline)

    min_to_right = np.minimum.accumulate(u_right[:, ::-1], axis=1)[:, ::-1]
    min_to_right = np.concatenate(
        [min_to_right[:, 1:], np.full((depth.shape[0], 1), np.inf)], axis=1
    )
    return min_to_right >= u_right - tol_px


def edge_mask(depth, jump_m, width_px):
    """No depth next to depth jumps (the matching window covers two depths)."""
    dz_x = np.abs(np.diff(depth, axis=1, append=depth[:, -1:]))
    dz_y = np.abs(np.diff(depth, axis=0, append=depth[-1:, :]))
    jumps = (dz_x > jump_m) | (dz_y > jump_m)
    return ~binary_dilation(jumps, iterations=width_px)


def specular_drop_probability(depth, normals, K, roughness, model):
    """Hole probability on metal: high facing the camera (projector glare, wider for
    rougher metal), low in between, high again near edge-on (pattern reflected away)."""
    points = backproject(depth, K)
    to_camera = -points / np.linalg.norm(points, axis=-1, keepdims=True).clip(min=1e-9)
    cos = np.clip((normals * to_camera).sum(axis=-1), -1.0, 1.0)
    angle = np.degrees(np.arccos(cos))

    lobe_deg = model["highlight_lobe_deg_per_roughness"] * roughness
    highlight = model["highlight_p"] * np.exp(-0.5 * (angle / lobe_deg) ** 2)
    start = model["grazing_start_deg"]
    grazing = model["grazing_p"] * np.clip((angle - start) / (90.0 - start), 0.0, 1.0)
    return np.maximum(highlight, grazing)


def specular_mask(depth, normals, metal, K, roughness, model, rng):
    """Holes in blobs: smoothed noise through the normal CDF is still uniform per pixel,
    so each pixel keeps its probability, but neighbours drop out together."""
    p = specular_drop_probability(depth, normals, K, roughness, model)
    field = gaussian_filter(rng.standard_normal(depth.shape), model["hole_blob_px"])
    uniform = ndtr(field / field.std())
    return ~(metal & (uniform < p))


def add_disparity_noise(depth, K, baseline, subpixel, corr_px, rng):
    """[BKM] 3e: noise on the disparity, so the depth error grows with Z^2.
    Smoothed into bumps and rescaled to subpixel RMS."""
    fb = K[0, 0] * baseline
    field = gaussian_filter(rng.standard_normal(depth.shape), corr_px)
    field /= field.std()
    return fb / (disparity_px(depth, K, baseline) + subpixel * field)


def quantize(depth, unit):
    """[BKM] 8a: 1 mm depth steps."""
    return np.round(depth / unit) * unit


def simulate_depth(
    depth, K, cam, model, seed, enabled=None, normals=None, metal=None, roughness=None
):
    """Ideal depth to D435 depth."""
    enabled = set(model["effects"] if enabled is None else enabled)
    unknown = enabled - set(EFFECTS)
    assert not unknown, f"unknown camera-model effects: {sorted(unknown)}"

    depth = depth.astype(np.float64)
    valid = np.isfinite(depth) & (depth > 0)

    if "min_depth" in enabled:
        valid &= min_depth_mask(depth, cam["min_depth_m"])
    if "left_band" in enabled:
        valid &= left_band_mask(depth, K, cam["baseline_m"])
    if "stereo_shadow" in enabled:
        valid &= stereo_shadow_mask(depth, K, cam["baseline_m"], model["shadow_tol_px"])
    if "edges" in enabled:
        valid &= edge_mask(depth, model["edge_jump_m"], model["edge_width_px"])
    if "specular" in enabled:
        assert normals is not None and metal is not None and roughness is not None
        valid &= specular_mask(
            depth, normals, metal, K, roughness, model, effect_rng(seed, "specular")
        )

    if "noise" in enabled:
        depth = add_disparity_noise(
            depth,
            K,
            cam["baseline_m"],
            cam["subpixel"],
            model["noise_corr_px"],
            effect_rng(seed, "noise"),
        )
    if "quantization" in enabled:
        depth = quantize(depth, cam["depth_unit_m"])

    return np.where(valid, depth, 0.0).astype(np.float32)


def simulate_frame(frame, cam, model, enabled=None):
    """simulate_depth for a rendered frame, seeded per scene and view."""
    return simulate_depth(
        frame.depth,
        frame.K,
        cam,
        model,
        seed=(model["seed"], frame.scene_id, frame.view),
        enabled=enabled,
        normals=frame.normals,
        metal=frame.category > 0,
        roughness=frame.meta["alu_roughness"],
    )
