"""
Procedural meshes of aluminium T-slot profiles and L-brackets, built from the parts
section of the config. Writes .obj files and a manifest.json with the measured size of
each mesh. Units are metres.
"""

import json
from pathlib import Path

import trimesh
import yaml
from shapely.affinity import rotate
from shapely.geometry import Point, box
from shapely.ops import unary_union

CATEGORY_IDS = {"profile": 1, "bracket": 2}  # table has the id 0


def build(part, tslot):
    if part["type"] == "profile":
        return tslot_profile(part["length_m"], tslot)
    if part["type"] == "bracket":
        return l_bracket(part["leg_m"], part["thickness_m"], part["width_m"])

    raise ValueError(f"unknown part type: {part['type']}")


def tslot_cross_section(size, slot_open, slot_wall, slot_inner_w, slot_depth, bore_d):
    """Generate the cross-section of a T-slot profile"""

    h = size / 2
    outer = box(-h, -h, h, h)

    neck = box(-slot_open / 2, h - slot_wall, slot_open / 2, h)
    cavity = box(-slot_inner_w / 2, h - slot_depth, slot_inner_w / 2, h - slot_wall)
    slot = unary_union([neck, cavity])

    slots = unary_union([rotate(slot, angle, origin=(0, 0)) for angle in [0, 90, 180, 270]])
    bore = Point(0, 0).buffer(bore_d / 2)

    section = outer.difference(slots).difference(bore)
    assert section.geom_type == "Polygon", (
        f"cross-section split into {len(section.geoms)} pieces: slots cut through the walls"
    )

    return section


def tslot_profile(length, tslot):
    section = tslot_cross_section(
        size=tslot["size_m"],
        slot_open=tslot["slot_open_m"],
        slot_wall=tslot["slot_wall_m"],
        slot_inner_w=tslot["slot_inner_w_m"],
        slot_depth=tslot["slot_depth_m"],
        bore_d=tslot["bore_d_m"],
    )
    mesh = trimesh.creation.extrude_polygon(section, height=length)
    mesh.apply_translation([0, 0, -length / 2])

    return mesh


def l_bracket(leg, thickness, width):
    """Create an L-shaped angle bracket to join two parts"""

    l_shape = unary_union([box(0, 0, leg, thickness), box(0, 0, thickness, leg)])
    mesh = trimesh.creation.extrude_polygon(l_shape, height=width)
    mesh.apply_translation(-mesh.bounds.mean(axis=0))

    return mesh


def export_parts(config_path="configs/default.yaml", out_dir="data/meshes"):
    cfg = yaml.safe_load(Path(config_path).read_text())["parts"]
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    total_weight = sum(p["weight"] for p in cfg["catalogue"])
    manifest = {}
    for part in cfg["catalogue"]:
        mesh = build(part, cfg["tslot"])
        assert mesh.is_watertight, f"{part['name']} is not watertight"
        mesh.export(out / f"{part['name']}.obj")
        manifest[part["name"]] = {
            "file": f"{part['name']}.obj",
            "type": part["type"],
            "category_id": CATEGORY_IDS[part["type"]],
            "probability": part["weight"] / total_weight,
            "extents_mm": (mesh.extents * 1000).round(3).tolist(),  # measured, not typed
        }
        print(
            f"{part['name']:16s} size={(mesh.extents * 1000).round(1)} mm  "
            f"p={manifest[part['name']]['probability']:.2f}"
        )

    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    export_parts()
