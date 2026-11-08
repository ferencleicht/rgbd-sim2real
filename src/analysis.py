"""Paired comparisons of the experiment conditions, and result figures."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import binomtest  # noqa: E402

OUT_DIR = Path("output/experiment")
FIG_DIR = Path("output/figures")
FLOAT_KEYS = (
    "n_points",
    "add_s_mm",
    "t_err_mm",
    "r_err_deg",
    "t_err_z_mm",
    "distance_m",
    "roughness",
    "fitness",
    "init_add_s_mm",
)


def load(tag):
    with open(OUT_DIR / f"results_{tag}.csv") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for key in FLOAT_KEYS:
            r[key] = float(r[key])
        for key in ("success", "strict_success"):
            r[key] = r[key] == "True"
    return rows


def by_view(rows, cond):
    """Rows of one condition, keyed by bracket view, for paired comparisons."""
    return {(r["scene"], r["view"], r["object"]): r for r in rows if r["condition"] == cond}


def paired(rows, a, b):
    """Compare strict success of two conditions on the same views."""
    ra, rb = by_view(rows, a), by_view(rows, b)
    keys = sorted(ra)
    sa = np.array([ra[k]["strict_success"] for k in keys])
    sb = np.array([rb[k]["strict_success"] for k in keys])
    a_only, b_only = int((sa & ~sb).sum()), int((~sa & sb).sum())
    n_discordant = a_only + b_only
    return {
        "a": a,
        "b": b,
        "views": len(keys),
        "success_a": float(sa.mean()),
        "success_b": float(sb.mean()),
        "a_only": a_only,
        "b_only": b_only,
        "p_value": binomtest(a_only, n_discordant).pvalue if n_discordant else 1.0,
    }


def wilson(k, n, z=1.96):
    """95% confidence interval of a success rate (Wilson score interval)."""
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return centre - half, centre + half


def plot_success(rows, conds, out):
    rates, lows, highs = [], [], []
    for cond in conds:
        s = [r["strict_success"] for r in rows if r["condition"] == cond]
        lo, hi = wilson(sum(s), len(s))
        rates.append(np.mean(s))
        lows.append(rates[-1] - lo)
        highs.append(hi - rates[-1])
    colors = ["#1f77b4" if c == "ideal" else "#d62728" if c == "full" else "#7f7f7f" for c in conds]
    fig, ax = plt.subplots(figsize=(7, 6))
    y = np.arange(len(conds))[::-1]
    ax.barh(y, rates, xerr=[lows, highs], color=colors, capsize=3)
    ax.set_yticks(y, conds, fontsize=8)
    ax.set_xlabel("strict success (t < 5 mm and r < 5 deg), 95% CI")
    ax.set_xlim(0, 1)
    fig.tight_layout()
    fig.savefig(out, dpi=150)


def plot_vs_distance(rows, out):
    bins = np.arange(0.3, 0.85, 0.1)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for cond, color in (("ideal", "#1f77b4"), ("full", "#d62728")):
        r = [x for x in rows if x["condition"] == cond and not np.isnan(x["t_err_mm"])]
        dist = np.array([x["distance_m"] for x in r])
        for ax, key, label in (
            (axes[0], "t_err_mm", "translation error [mm]"),
            (axes[1], "r_err_deg", "rotation error [deg]"),
        ):
            values = np.array([x[key] for x in r])
            idx = np.digitize(dist, bins)
            centres = [(bins[i - 1] + bins[i]) / 2 for i in range(1, len(bins)) if (idx == i).any()]
            medians = [np.median(values[idx == i]) for i in range(1, len(bins)) if (idx == i).any()]
            ax.plot(centres, medians, "o-", color=color, label=cond)
            ax.set_xlabel("camera distance [m]")
            ax.set_ylabel(f"median {label}")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)


def plot_error_distribution(rows, out):
    fig, ax = plt.subplots(figsize=(6, 3.6))
    bins = np.logspace(-3, 3, 60)  # 0.001 to 1000 mm
    for cond, color in (("ideal", "#1f77b4"), ("full", "#d62728")):
        t = [r["t_err_mm"] for r in rows if r["condition"] == cond and not np.isnan(r["t_err_mm"])]
        # clip so no view is dropped outside the bins
        t = np.clip(t, bins[0], bins[-1])
        ax.hist(t, bins=bins, histtype="step", linewidth=1.5, color=color, label=cond)
    ax.set_xscale("log")
    ax.axvline(5, color="black", linestyle="--", linewidth=1, label="5 mm threshold")
    ax.set_xlabel("translation error [mm]")
    ax.set_ylabel("bracket views")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="main")
    args = parser.parse_args()

    rows = load(args.tag)
    conds = list(dict.fromkeys(r["condition"] for r in rows))
    effects = [c.removeprefix("only_") for c in conds if c.startswith("only_")]

    results = {"ideal_vs_full": paired(rows, "ideal", "full")}
    results["remove_one_from_full"] = [paired(rows, f"without_{e}", "full") for e in effects]
    results["only_one_vs_ideal"] = [paired(rows, f"only_{e}", "ideal") for e in effects]

    ideal = list(by_view(rows, "ideal").values())
    ok = [r for r in ideal if r["strict_success"]]
    bad = [r for r in ideal if not r["strict_success"]]
    results["ideal_failures"] = {
        "count": len(bad),
        "median_points_failed": float(np.median([r["n_points"] for r in bad])) if bad else None,
        "median_points_ok": float(np.median([r["n_points"] for r in ok])),
        "median_distance_failed": float(np.median([r["distance_m"] for r in bad])) if bad else None,
        "median_distance_ok": float(np.median([r["distance_m"] for r in ok])),
    }

    p = results["ideal_vs_full"]
    print(
        f"ideal vs full over {p['views']} bracket views: strict success "
        f"{p['success_a']:.1%} -> {p['success_b']:.1%}; only ideal succeeds: {p['a_only']}, "
        f"only full succeeds: {p['b_only']}, McNemar p = {p['p_value']:.2g}"
    )
    print(f"\n{'removing from full':24s} {'success':>8s} {'fixed':>6s} {'broken':>7s} {'p':>8s}")
    for q in results["remove_one_from_full"]:
        print(
            f"{q['a']:24s} {q['success_a']:8.1%} {q['a_only']:6d} {q['b_only']:7d} "
            f"{q['p_value']:8.2g}"
        )
    print(f"\n{'one effect alone':24s} {'success':>8s} {'broken':>7s} {'fixed':>6s} {'p':>8s}")
    for q in results["only_one_vs_ideal"]:
        print(
            f"{q['a']:24s} {q['success_a']:8.1%} {q['b_only']:7d} {q['a_only']:6d} "
            f"{q['p_value']:8.2g}"
        )
    print(f"\nfailures on ideal depth: {json.dumps(results['ideal_failures'])}")

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plot_success(rows, conds, FIG_DIR / f"success_{args.tag}.png")
    plot_vs_distance(rows, FIG_DIR / f"error_vs_distance_{args.tag}.png")
    plot_error_distribution(rows, FIG_DIR / f"error_distribution_{args.tag}.png")
    (OUT_DIR / f"analysis_{args.tag}.json").write_text(json.dumps(results, indent=2))
    print(f"\nfigures in {FIG_DIR}")


if __name__ == "__main__":
    main()
