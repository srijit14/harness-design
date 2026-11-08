"""The two report figures, from analysis/midway/summary.json (run analysis/midway_analysis.py --test first).

  uv run --no-project --python 3.11 --with matplotlib python analysis/midway_figures.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
S = json.loads((ROOT / "analysis" / "midway" / "summary.json").read_text())
OUT = ROOT / "analysis" / "midway" / "figures"

SURFACE, INK, INK2, GRID = "#ffffff", "#0b0b0b", "#52514e", "#e6e5e0"
V1, V2, V3, CEIL = "#8a8985", "#2a78d6", "#eb6834", "#52514e"
FOUND, OTHER, WASTED = "#1baf7a", "#c3c2b7", "#6250d6"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans", "font.size": 8, "text.color": INK, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": GRID, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": 0.6,
})


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def failure_modes():
    """Dev: for each failure mode among the 75 misses, % of tries in which each version finds the same detail."""
    F = S["failure_modes_latest"]
    names = {"NARROW": "Narrow search", "PICKED": "Picked among options", "FILLED": "Filled in a value",
             "WEAKENED": "Weakened the request"}
    groups = [("none", "v1: no checks", V1), ("unsourced", "v2: no-source values", V2),
              ("alternatives", "v3: + look-alikes", V3)]
    fig, ax = plt.subplots(figsize=(4.4, 1.75))
    h = 0.26
    modes = list(names)
    for i, m in enumerate(reversed(modes)):
        for j, (g, _, c) in enumerate(groups):
            v = F[g][m]["found_pct"]
            y = i + (1 - j) * h
            ax.barh(y, v, height=h - 0.04, color=c, edgecolor=SURFACE, linewidth=0.8)
            ax.text(v + 1, y, f"{v:.0f}", va="center", fontsize=6.5, color=INK2)
    ax.set_yticks(range(len(modes)), [f"{names[m]} ({F['none'][m]['n']}/{F['none'][m]['details']})" for m in reversed(modes)])
    ax.set_xlim(0, 108)
    ax.set_xlabel("% of tries that find the hidden detail")
    ax.grid(axis="y", visible=False)
    ax.legend([plt.Rectangle((0, 0), 1, 1, color=c) for _, _, c in groups], [lab for _, lab, _ in groups],
              loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=3, frameon=False, fontsize=6.5, handlelength=1, columnspacing=1)
    save(fig, "fig1_failure_modes")


def held_out():
    """Held-out: pass@1 per version, and what each version's questions found (per run)."""
    B = S["batches"]
    rows = [(n, lab, c) for n, lab, c in (("v1_test", "v1", V1), ("v2_test", "v2", V2), ("v3_test", "v3", V3)) if n in B]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(4.6, 1.4), gridspec_kw={"wspace": 0.4, "width_ratios": [1, 1.25]})
    bars = rows + [("full_info_test", "full info", CEIL)]
    for y, (n, lab, c) in enumerate(reversed(bars)):
        v = B[n]["pass_at_1"]
        a1.barh(y, v, height=0.6, color=c, edgecolor=SURFACE, linewidth=1)
        a1.text(v + 1.5, y, f"{v:.1f}", va="center", fontsize=7, color=INK)
    a1.set_yticks(range(len(bars)), [lab for _, lab, _ in reversed(bars)])
    a1.set_xlim(0, 72)
    a1.set_xlabel("pass@1 (%)")
    for y, (n, lab, c) in enumerate(reversed(rows)):
        b = B[n]
        found = b["details_found"] / b["runs"]
        wasted = b["wasted_per_run"]
        other = b["questions_per_run"] - found - wasted
        left = 0
        for v, col in ((found, FOUND), (other, OTHER), (wasted, WASTED)):
            a2.barh(y, v, left=left, height=0.6, color=col, edgecolor=SURFACE, linewidth=1)
            if v >= 0.6:
                a2.text(left + v / 2, y, f"{v:.1f}", ha="center", va="center", fontsize=6.5,
                        color=SURFACE if col == WASTED else INK)
            left += v
        a2.text(left + 0.12, y, f"{b['questions_per_run']:.1f}", va="center", fontsize=7, color=INK)
    a2.set_yticks(range(len(rows)), [lab for _, lab, _ in reversed(rows)])
    a2.set_xlim(0, 8.2)
    a2.set_xlabel("questions to the human per run")
    a2.legend([plt.Rectangle((0, 0), 1, 1, color=c) for c in (FOUND, OTHER, WASTED)],
              ["found a hidden detail", "repeat / judge error", "irrelevant"], loc="lower right",
              bbox_to_anchor=(1.04, 1.0), ncol=3, frameon=False, fontsize=6, handlelength=1, columnspacing=0.8)
    for a in (a1, a2):
        a.grid(axis="y", visible=False)
    save(fig, "fig2_held_out")


if __name__ == "__main__":
    failure_modes()
    held_out()
    print(f"Wrote {OUT}")
