"""
Figure: ideal vs simulated D435 depth for one frame, with the cause of every hole.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from camera_model import simulate_frame  # noqa: E402
from dataset import load_frame  # noqa: E402

CONFIG = yaml.safe_load(Path("configs/default.yaml").read_text())
CAM, MODEL = CONFIG["camera"], CONFIG["camera_model"]
OUT_DIR = Path("output/figures")


HOLE_EFFECTS = ["min_depth", "left_band", "stereo_shadow", "edges", "specular"]
HOLE_COLORS = ["#7f7f7f", "#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"]


def hole_causes(frame):
    causes = np.zeros(frame.depth.shape, dtype=int)
    for i, effect in enumerate(HOLE_EFFECTS, start=1):
        removed = simulate_frame(frame, CAM, MODEL, enabled={effect}) == 0
        causes[(causes == 0) & removed] = i
    return causes


def largest_bracket_crop(frame, margin=25):
    ids = [i for i, name in frame.names.items() if name.startswith("bracket")]
    assert ids, "no bracket in this frame; pick another scene or view"
    best = max(ids, key=lambda i: (frame.instance == i).sum())
    v, u = np.nonzero(frame.instance == best)
    return (
        slice(max(v.min() - margin, 0), v.max() + margin),
        slice(max(u.min() - margin, 0), u.max() + margin),
    )


def central_roi(shape, fraction=0.8):
    """[DS] Table 4-14: depth quality is specified within 80% of the depth FOV."""
    h, w = shape
    mh, mw = int(h * (1 - fraction) / 2), int(w * (1 - fraction) / 2)
    roi = np.zeros(shape, dtype=bool)
    roi[mh : h - mh, mw : w - mw] = True
    return roi


def show_depth(ax, depth, vmin, vmax, title):
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("black")  # invalid pixels (0) shown in black, as holes
    ax.imshow(np.ma.masked_equal(depth, 0), cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=9)
    ax.axis("off")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=int, default=0)
    parser.add_argument("--view", type=int, default=0)
    args = parser.parse_args()

    frame = load_frame(args.scene, args.view)
    simulated = simulate_frame(frame, CAM, MODEL)
    causes = hole_causes(frame)
    crop = largest_bracket_crop(frame)

    roi = central_roi(frame.depth.shape)
    table, metal = (frame.category == 0) & roi, (frame.category > 0) & roi
    fill_table = (simulated[table] > 0).mean()
    fill_metal = (simulated[metal] > 0).mean()

    vmin, vmax = np.percentile(frame.depth[frame.depth > 0], [1, 99])
    crop_depth = frame.depth[crop]
    cmin, cmax = np.percentile(crop_depth[crop_depth > 0], [1, 99])
    cause_cmap = ListedColormap(["white", *HOLE_COLORS])

    fig, axes = plt.subplots(2, 3, figsize=(13, 6.2))
    axes[0, 0].imshow(frame.rgb)
    axes[0, 0].set_title(f"RGB (scene {args.scene}, view {args.view})", fontsize=9)
    axes[0, 0].axis("off")
    show_depth(axes[0, 1], frame.depth, vmin, vmax, "ideal depth (renderer)")
    show_depth(
        axes[0, 2],
        simulated,
        vmin,
        vmax,
        f"simulated D435 depth  |  fill in 80% ROI: table {fill_table:.1%}, metal {fill_metal:.1%}",
    )

    show_depth(axes[1, 0], crop_depth, cmin, cmax, "bracket close-up: ideal")
    show_depth(axes[1, 1], simulated[crop], cmin, cmax, "bracket close-up: simulated")
    axes[1, 2].imshow(
        causes[crop], cmap=cause_cmap, vmin=0, vmax=len(HOLE_EFFECTS), interpolation="nearest"
    )
    axes[1, 2].set_title("cause of each hole", fontsize=9)
    axes[1, 2].axis("off")
    axes[1, 2].legend(
        handles=[Patch(color=c, label=e) for e, c in zip(HOLE_EFFECTS, HOLE_COLORS, strict=True)],
        loc="center left",
        bbox_to_anchor=(1.0, 0.5),
        fontsize=8,
        frameon=False,
    )

    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"camera_model_s{args.scene:03d}_v{args.view}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"saved {out}  (fill in 80% ROI: table {fill_table:.1%}, metal {fill_metal:.1%})")


if __name__ == "__main__":
    main()
