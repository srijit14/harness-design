"""Figures for the midway report and demo, from analysis/midway/summary.json (run midway_analysis.py first).

  uv run --with matplotlib python analysis/midway_figures.py

Writes analysis/midway/figures/*.png (and .pdf for the report).
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
S = json.loads((ROOT / "analysis" / "midway" / "summary.json").read_text())
OUT = ROOT / "analysis" / "midway" / "figures"

SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8985", "#e6e5e0"
GROUP = {"none": "#8a8985", "unsourced": "#2a78d6", "alternatives": "#eb6834", "ceiling": INK2}
GROUP_NAME = {"none": "No checks", "unsourced": "Value with no source (v2)", "alternatives": "Similar alternatives (v3)"}
FOUND, WASTED, OTHER = "#1baf7a", "#6250d6", "#c3c2b7"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans", "font.size": 10, "text.color": INK, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": GRID, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": 0.8,
})


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def pass_by_batch():
    """Every batch's pass@1, with the no-check batches' range as the noise band."""
    order = ["v0", "v1", "v1_pinned", "v1b", "v2", "v2_pinned", "v2b", "v3", "full_info"]
    B = S["batches"]
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    lo, hi = min(S["noise"]["no_check_pass_at_1"]), max(S["noise"]["no_check_pass_at_1"])
    ax.axvspan(lo, hi, color=GRID, zorder=0)
    ax.text(lo - 0.5, -0.75, "no-check range", ha="right", va="center", color=INK2, fontsize=8)
    ax.set_ylim(-1.1, len(order) - 0.5)
    for y, n in enumerate(reversed(order)):
        b = B[n]
        c = GROUP[b["group"]]
        ax.scatter(b["pass_at_1"], y, s=70, color=c, edgecolor=SURFACE, linewidth=2, zorder=3,
                   marker="D" if b["group"] == "ceiling" else "o")
        ax.text(b["pass_at_1"] + 1.5, y, f"{b['pass_at_1']:.1f}%", va="center", color=INK, fontsize=9)
    ax.set_yticks(range(len(order)), [B[n]["label"] for n in reversed(order)])
    ax.set_xlim(20, 90)
    ax.set_xlabel("pass@1 (%), 20 practice tasks × 3 tries")
    ax.grid(axis="y", visible=False)
    ax.set_title("v2 sits near the noise; v3 is the first clear gain", loc="left", fontsize=11, color=INK)
    save(fig, "fig1_pass_by_batch")


def paired_diffs():
    """pass@1 difference against no checks, paired by task, two-level bootstrap 95% CI."""
    C = S["noise"]["comparisons"]
    rows = [("v2 family vs no checks", "unsourced"), ("v3 vs no checks", "alternatives"), ("v3 vs v2 family", "alternatives")]
    if "held-out: v3 vs v1" in C:
        rows.append(("held-out: v3 vs v1", "alternatives"))
    fig, ax = plt.subplots(figsize=(7.2, 0.6 + 0.55 * len(rows)))
    ax.axvline(0, color=INK2, linewidth=1)
    for y, (k, g) in enumerate(reversed(rows)):
        c = C[k]
        ax.plot(c["ci95"], [y, y], color=GROUP[g], linewidth=2, solid_capstyle="round")
        ax.scatter(c["diff"], y, s=60, color=GROUP[g], edgecolor=SURFACE, linewidth=2, zorder=3)
        ax.text(c["ci95"][1] + 1, y, f"{c['diff']:+.1f}  (p = {c['perm_p']:.3f})", va="center", fontsize=9, color=INK)
    ax.set_yticks(range(len(rows)), [k for k, _ in reversed(rows)])
    ax.set_xlim(-10, 50)
    ax.set_xlabel("pass@1 difference (points), 95% CI; p = within-task permutation test, one-sided")
    ax.grid(axis="y", visible=False)
    save(fig, "fig2_paired_diffs")


def failure_modes():
    """The 75 hand-labeled misses of v0: how often each version finds the same hidden detail."""
    names = {"NARROW": "Narrow search", "PICKED": "Picked among options", "FILLED": "Filled in a value",
             "WEAKENED": "Weakened the request", "DROPPED": "Dropped the request", "LITERAL": "Obeyed the impossible",
             "REJECTED": "Asked, judge rejected", "ALL": "All 75 misses"}
    F = S["failure_modes"]
    modes = list(names)
    groups = ["none", "unsourced", "alternatives"]
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    h = 0.26
    for i, m in enumerate(reversed(modes)):
        for j, g in enumerate(groups):
            v = F[g][m]["found_pct"]
            y = i + (1 - j) * h
            ax.barh(y, v, height=h - 0.03, color=GROUP[g], edgecolor=SURFACE, linewidth=1)
            ax.text(v + 1, y, f"{v:.0f}", va="center", fontsize=7.5, color=INK2)
    ax.set_yticks(range(len(modes)), [f"{names[m]} ({F['none'][m]['n']} / {F['none'][m]['details']})" for m in reversed(modes)])
    ax.set_xlim(0, 110)
    ax.set_xlabel("% of tries that find the same hidden detail   (labels: misses / distinct details)")
    ax.grid(axis="y", visible=False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=GROUP[g]) for g in groups]
    ax.legend(handles, ["No checks (v1, v1 pinned, v1b)", "Value with no source (v2, v2 pinned, v2b)", "Similar alternatives (v3)"],
              loc="upper center", bbox_to_anchor=(0.4, -0.13), frameon=False, fontsize=8, ncol=2)
    ax.set_title("Each check moves the failure mode it targets", loc="left", fontsize=11, color=INK)
    save(fig, "fig3_failure_modes")


def question_cost():
    """Questions per run, split by what they found (pooled per group)."""
    B = S["batches"]
    groups = {"No checks": ["v1", "v1_pinned", "v1b"], "v2 family": ["v2", "v2_pinned", "v2b"], "v3": ["v3"]}
    fig, ax = plt.subplots(figsize=(7.2, 2.4))
    for y, (g, ns) in enumerate(reversed(list(groups.items()))):
        runs = sum(B[n]["runs"] for n in ns)
        q = sum(B[n]["questions_per_run"] * B[n]["runs"] for n in ns) / runs
        found = sum(B[n]["details_found"] for n in ns) / runs
        wasted = sum(B[n]["wasted_per_run"] * B[n]["runs"] for n in ns) / runs
        other = q - found - wasted
        left = 0
        for v, c, lab in ((found, FOUND, "found a detail"), (other, OTHER, "repeat / judge error"), (wasted, WASTED, "irrelevant")):
            ax.barh(y, v, left=left, height=0.55, color=c, edgecolor=SURFACE, linewidth=2)
            if v > 0.45:
                ax.text(left + v / 2, y, f"{v:.1f}", ha="center", va="center", fontsize=8.5,
                        color=INK if c != WASTED else SURFACE)
            left += v
        ax.text(left + 0.1, y, f"{q:.1f} per run", va="center", fontsize=9, color=INK)
    ax.set_yticks(range(len(groups)), list(reversed(groups)))
    ax.set_xlim(0, 9)
    ax.set_xlabel("questions to the human per run")
    ax.grid(axis="y", visible=False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (FOUND, OTHER, WASTED)]
    ax.legend(handles, ["found a hidden detail", "repeat / judge error", "irrelevant (wasted)"], loc="lower right",
              frameon=False, fontsize=8, ncol=3, bbox_to_anchor=(1, 1.02))
    save(fig, "fig4_question_cost")


def per_task():
    """Passes out of 3, per task and batch."""
    order = [n for n in S["per_task_order"] if not n.endswith("_test")]
    tasks = list(S["per_task"])
    tasks = [t for t in tasks if all(n in S["per_task"][t] for n in order)]
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    ramp = ["#f3f2ee", "#b7d3f6", "#5598e7", "#1c5cab"]
    for i, t in enumerate(tasks):
        for j, n in enumerate(order):
            v = S["per_task"][t][n]
            ax.add_patch(plt.Rectangle((j, i), 0.94, 0.9, color=ramp[v]))
            ax.text(j + 0.47, i + 0.45, str(v), ha="center", va="center", fontsize=8, color=INK if v < 2 else SURFACE)
    ax.set_xlim(0, len(order))
    ax.set_ylim(len(tasks), 0)
    labels = {"v0": "v0\nScale", "v1": "v1", "v1_pinned": "v1\npinned", "v1b": "v1b", "v2": "v2", "v2_pinned": "v2\npinned",
              "v2b": "v2b", "v3": "v3", "full_info": "full\ninfo"}
    ax.set_xticks([j + 0.47 for j in range(len(order))], [labels.get(n, n) for n in order], fontsize=8)
    ax.set_yticks([i + 0.45 for i in range(len(tasks))], [t.replace("public_sql_", "task ") for t in tasks], fontsize=8)
    ax.xaxis.tick_top()
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    save(fig, "fig5_per_task")


def held_out():
    """Held-out tasks: v1 vs v3 vs full info, on pass@1 and on schema details found (two panels, two scales)."""
    B = S["batches"]
    if "v3_test" not in B:
        return
    names = [("v1_test", "v1, no checks", "none"), ("v3_test", "v3", "alternatives"), ("full_info_test", "Full info", "ceiling")]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 2.4), gridspec_kw={"wspace": 0.6})
    for y, (n, lab, g) in enumerate(reversed(names)):
        v = B[n]["pass_at_1"]
        a1.barh(y, v, height=0.55, color=GROUP[g], edgecolor=SURFACE, linewidth=2)
        a1.text(v + 1.5, y, f"{v:.1f}%", va="center", fontsize=9, color=INK)
    a1.set_yticks(range(len(names)), [lab for _, lab, _ in reversed(names)])
    a1.set_xlim(0, 75)
    a1.set_xlabel("pass@1 (%)")
    a1.set_title("Solved", loc="left", fontsize=10, color=INK)
    rows = names[:2]
    for y, (n, lab, g) in enumerate(reversed(rows)):
        f, t = B[n]["by_type"]["schema"]
        a2.barh(y, 100 * f / t, height=0.55, color=GROUP[g], edgecolor=SURFACE, linewidth=2)
        a2.text(100 * f / t + 1.5, y, f"{f}/{t}", va="center", fontsize=9, color=INK)
    a2.set_yticks(range(len(rows)), [lab for _, lab, _ in reversed(rows)])
    a2.set_xlim(0, 100)
    a2.set_xlabel("schema hidden details found (%)")
    a2.set_title("Schema details found", loc="left", fontsize=10, color=INK)
    for a in (a1, a2):
        a.grid(axis="y", visible=False)
    fig.suptitle("Held-out: 39 unseen tasks × 3 tries (full info: 1 try)", x=0.02, ha="left", y=1.08, fontsize=11, color=INK)
    save(fig, "fig6_held_out")


if __name__ == "__main__":
    pass_by_batch()
    paired_diffs()
    failure_modes()
    question_cost()
    per_task()
    held_out()
    print(f"Wrote figures to {OUT}")
