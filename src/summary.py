"""Dataset statistics from the per-scene meta.json files."""

import json
from pathlib import Path

import numpy as np

RENDER_DIR = Path("data/renders")

MIN_VISIBLE_PX = 500


def main():
    metas = [json.loads(p.read_text()) for p in sorted(RENDER_DIR.glob("scene_*/meta.json"))]
    objects = [o for m in metas for o in m["objects"]]
    off_table = [o for o in objects if not o["on_table"]]

    bracket_px = np.array(
        [
            view[o["name"]]
            for m in metas
            for o in m["objects"]
            if o["mesh"] == "bracket" and o["on_table"]
            for view in m["visible_px"]
        ]
    )
    usable = bracket_px >= MIN_VISIBLE_PX
    roughness = np.array([m["alu_roughness"] for m in metas])

    summary = {
        "scenes": len(metas),
        "frames": sum(len(m["visible_px"]) for m in metas),
        "parts": len(objects),
        "brackets": sum(o["mesh"] == "bracket" for o in objects),
        "off_table": len(off_table),
        "off_table_rate": len(off_table) / len(objects),
        "bracket_views": int(bracket_px.size),
        "bracket_views_hidden": int((bracket_px == 0).sum()),
        "bracket_views_usable": int(usable.sum()),
        "min_visible_px": MIN_VISIBLE_PX,
        "bracket_px_percentiles": {
            str(q): int(np.percentile(bracket_px[bracket_px > 0], q)) for q in (10, 50, 90)
        },
        "roughness_range": [float(roughness.min()), float(roughness.max())],
    }

    for key, value in summary.items():
        print(f"{key:24s} {value}")

    (RENDER_DIR / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
