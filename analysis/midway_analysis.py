"""Midway analysis of every dev batch (20 practice tasks x 3 tries). Offline: reads run folders, no model calls.

  vendor/hil-bench/.venv/bin/python analysis/midway_analysis.py            # dev batches
  vendor/hil-bench/.venv/bin/python analysis/midway_analysis.py --test     # also the held-out batches, if run

Writes analysis/midway/summary.json and summary.md:
  1. per batch: pass@1, pass@3, pass^3, questions, details found by type, wasted questions, flags, cost
  2. noise: spread of the four no-check batches; paired task bootstrap and within-task permutation test
  3. after a hold: what the questions asked after a held submission found, by check
  4. the 75 hand-labeled misses of v0 (analysis/failure_labels_step4f.tsv): how often each version finds them
  5. per task: passes out of 3 in every batch
"""

import argparse
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hilx.run import list_cost, token_counts  # noqa: E402

OUT = ROOT / "analysis" / "midway"
TASKS = ROOT / "data" / "sql_tasks"
MODEL = "openrouter/qwen/qwen3-235b-a22b-2507"
CANT_ANSWER = "can't answer"  # the judge's own JSON failed to parse (FIRST_EXPERIMENT.md §0.8)
SEED, N_BOOT = 329, 10_000

# name, label, group, folder. Groups: no checks / value with no source / similar alternatives.
DEV = [
    ("v0", "Scale's harness", "none", "runs/step4f_qwen235b_fixed/both_20x3"),
    ("v1", "v1 (mixed routing)", "none", "runs/v1_native/dev_20x3"),
    ("v1_pinned", "v1 (pinned)", "none", "runs/v1_native/dev_20x3_parasail"),
    ("v1b", "v1b (+ open-question rule)", "none", "runs/v1b_open_questions/dev_20x3"),
    ("v2", "v2 (mixed routing)", "unsourced", "runs/v2_unsourced/dev_20x3"),
    ("v2_pinned", "v2 (pinned)", "unsourced", "runs/v2_unsourced/dev_20x3_parasail"),
    ("v2b", "v2b (+ open-question rule)", "unsourced", "runs/v2b_unsourced_open/dev_20x3"),
    ("v3", "v3 (similar alternatives)", "alternatives", "runs/v3_alternatives/dev_20x3"),
    ("full_info", "Full info (all answers given)", "ceiling", "runs/step4f_qwen235b_fixed/both_20x3"),
]
TEST = [
    ("v1_test", "v1, held-out", "none", "runs/v1_native/test_small_39x3"),
    ("v2_test", "v2, held-out", "unsourced", "runs/v2_unsourced/test_small_39x3"),
    ("v3_test", "v3, held-out", "alternatives", "runs/v3_alternatives/test_small_39x3"),
    ("full_info_test", "Full info, held-out (1 try)", "ceiling", "runs/step4g_fullinfo_test/fullinfo_39x1"),
]


def registry(task: str) -> dict[str, str]:
    """Hidden details (blockers) of a task: id -> type (question / schema / business info)."""
    return {b["id"]: b["type"] for b in json.loads((TASKS / task / "blocker_registry.json").read_text())["blockers"]}


def tool_positions(messages: list[dict], tool: str) -> list[int]:
    """Positions of the assistant messages that call `tool`. Checkpoint "call" numbers count model calls,
    including format-error retries that are dropped from the history, so they are not used to align."""
    calls = [m for m in messages if m["role"] == "assistant"]
    return [n for n, m in enumerate(calls) if any(c["function"]["name"] == tool for c in m.get("tool_calls") or [])]


def run_record(task, tr, resolved, questions, cost=None, checkpoints=(), messages=None) -> dict:
    present = registry(task)
    found = {q["blocker_name"] for q in questions if q["blocker_name"]}
    rec = {
        "task": task, "try": int(tr), "resolved": bool(resolved), "present": present, "found": found,
        "n_q": len(questions),
        "wasted": sum(1 for q in questions if not q["blocker_name"] and not q["response"].startswith(CANT_ANSWER)),
        "cant_answer": sum(1 for q in questions if q["response"].startswith(CANT_ANSWER)),
        "cost": cost, "flags": [f for c in checkpoints for f in c["flags"]],
        "held": any(c["held"] for c in checkpoints), "after_hold": None,
    }
    if rec["held"] and messages is not None:
        rec["after_hold"] = after_hold(questions, tool_positions(messages, "ask_human"),
                                       tool_positions(messages, "submit_sql"), checkpoints)
    return rec


def after_hold(questions, calls, submits, checkpoints) -> dict | None:
    """Split questions at the first held submission; attribute each later question to the checks whose
    flags held the submission just before it. The k-th checkpoint is the k-th submit_sql call."""
    if len(calls) != len(questions) or len(submits) < len(checkpoints):
        return None  # judge log and trajectory disagree; leave the run out rather than guess
    held = [c | {"call": submits[i]} for i, c in enumerate(checkpoints) if c["held"]]
    first = held[0]["call"]
    before = {q["blocker_name"] for q, n in zip(questions, calls) if n < first and q["blocker_name"]}
    seen, out = set(before), []
    for q, n in zip(questions, calls):
        if n < first:
            continue
        cause = [c for c in held if c["call"] < n][-1]
        checks = "+".join(sorted({f["check"] for f in cause["flags"]}))
        b = q["blocker_name"]
        kind = ("cant_answer" if q["response"].startswith(CANT_ANSWER) else "wasted" if not b
                else "repeat" if b in seen else "new")
        seen |= {b} if b else set()
        out.append({"kind": kind, "blocker": b, "checks": checks})
    return {"before": before, "questions": out}


def load_ours(folder: Path) -> list[dict]:
    logs = json.loads((folder / "ask_human_logs.json").read_text()) if (folder / "ask_human_logs.json").exists() else {}
    runs = []
    for f in sorted(folder.glob("pass_*/*/result.json")):
        r = json.loads(f.read_text())
        traj = json.loads((f.parent / "traj.json").read_text())
        cost = r.get("cost_list")
        if cost is None:  # runs made before cost_list existed: price the tokens at the same fixed list
            cost = list_cost(token_counts(traj["messages"]), MODEL)
        questions = logs.get(r["run_id"], {}).get("questions", [])
        runs.append(run_record(r["task"], r["pass"], r["resolved"], questions, cost, r["checkpoints"], traj["messages"]))
    return runs


def load_scale(folder: Path, mode: str) -> list[dict]:
    """Scale's harness (v0): native asking, or the full-info ceiling (all hidden details given up front)."""
    logs = json.loads((folder / "ask_human_logs.json").read_text()) if (folder / "ask_human_logs.json").exists() else {}
    runs = []
    for i in json.loads((folder / "consolidated_results.json").read_text())["instances"]:
        if i["mode"] != mode:
            continue
        qs = logs.get(i["instance_id"], {}).get("questions", []) if mode == "ask_human" else []
        runs.append(run_record(i["original_instance_id"], i["pass"], i["resolved"], qs))
    return runs


def load(name, folder) -> list[dict]:
    folder = ROOT / folder
    if name == "v0":
        return load_scale(folder, "ask_human")
    if name.startswith("full_info"):
        return load_scale(folder, "full_info")
    return load_ours(folder)


# ---------- 1. per batch ----------

def per_task(runs) -> dict[str, list[bool]]:
    t = defaultdict(list)
    for r in runs:
        t[r["task"]].append(r["resolved"])
    return t


def summarize(runs) -> dict:
    n, tasks = len(runs), per_task(runs)
    by_type = defaultdict(lambda: [0, 0])
    for r in runs:
        for b, typ in r["present"].items():
            by_type[typ][1] += 1
            by_type[typ][0] += b in r["found"]
    n_q = sum(r["n_q"] for r in runs)
    found = sum(len(r["found"]) for r in runs)
    passes = sum(r["resolved"] for r in runs)
    costs = [r["cost"] for r in runs if r["cost"] is not None]
    per_try = Counter(r["try"] for r in runs if r["resolved"])
    return {
        "runs": n, "passes": passes,
        "pass_at_1": 100 * passes / n,
        "pass_at_1_per_try": [per_try.get(p, 0) for p in sorted({r["try"] for r in runs})],
        "pass_at_3": 100 * sum(any(v) for v in tasks.values()) / len(tasks),
        "pass_pow_3": 100 * sum(all(v) for v in tasks.values()) / len(tasks),  # solved on every try
        "tasks_flipping": sum(0 < sum(v) < len(v) for v in tasks.values()),
        "questions_per_run": n_q / n,
        "details_found": found, "details_present": sum(len(r["present"]) for r in runs),
        "recall": 100 * found / max(1, sum(len(r["present"]) for r in runs)),
        "scale_precision": 100 * found / n_q if n_q else None,  # Scale's: distinct details found / questions
        "wasted_per_run": sum(r["wasted"] for r in runs) / n,
        "wasted_share": 100 * sum(r["wasted"] for r in runs) / n_q if n_q else None,
        "cant_answer": sum(r["cant_answer"] for r in runs),
        "questions_per_pass": n_q / passes if passes else None,
        "by_type": {k: v for k, v in sorted(by_type.items())},
        "flags": sum(len(r["flags"]) for r in runs),
        "flags_by_check": dict(Counter(f["check"] for r in runs for f in r["flags"])),
        "held_runs": sum(r["held"] for r in runs),
        "cost_list": round(sum(costs), 3) if costs else None,
        "cost_per_pass": round(sum(costs) / passes, 4) if costs and passes else None,
    }


# ---------- 2. noise ----------

def task_runs(batches) -> dict[str, list[bool]]:
    t = defaultdict(list)
    for runs in batches:
        for r in runs:
            t[r["task"]].append(r["resolved"])
    return t


def paired_bootstrap(a: dict, b: dict, rng) -> dict:
    """Difference in pass@1 (points). Two levels: tasks resampled with replacement, then each side's tries
    within the task, so both task mix and try-to-try noise count."""
    tasks = sorted(set(a) & set(b))
    rate = lambda v: sum(v) / len(v)
    diff = 100 * sum(rate(a[t]) - rate(b[t]) for t in tasks) / len(tasks)
    boots = []
    for _ in range(N_BOOT):
        ts = [rng.choice(tasks) for _ in tasks]
        boots.append(100 * sum(rate(rng.choices(a[t], k=len(a[t]))) - rate(rng.choices(b[t], k=len(b[t]))) for t in ts) / len(ts))
    boots.sort()
    return {"diff": diff, "ci95": [boots[int(0.025 * N_BOOT)], boots[int(0.975 * N_BOOT)]],
            "p_not_positive": sum(x <= 0 for x in boots) / N_BOOT}


def bootstrap_ci(t: dict, rng) -> list[float]:
    """95% CI of one batch's pass@1, same two-level resampling."""
    tasks = sorted(t)
    boots = sorted(100 * sum(sum(rng.choices(t[x], k=len(t[x]))) / len(t[x]) for x in (rng.choice(tasks) for _ in tasks)) / len(tasks)
                   for _ in range(N_BOOT))
    return [boots[int(0.025 * N_BOOT)], boots[int(0.975 * N_BOOT)]]


def permutation(a_batches, b_batches, rng, n=N_BOOT) -> float:
    """One-sided p-value that A beats B: within each task, shuffle which runs belong to A and B."""
    a, b = defaultdict(list), defaultdict(list)
    for runs in a_batches:
        for r in runs:
            a[r["task"]].append(r["resolved"])
    for runs in b_batches:
        for r in runs:
            b[r["task"]].append(r["resolved"])
    tasks = sorted(set(a) & set(b))

    def stat(pairs):
        return sum(sum(x) for x, _ in pairs) / sum(len(x) for x, _ in pairs) - sum(sum(y) for _, y in pairs) / sum(len(y) for _, y in pairs)

    observed = stat([(a[t], b[t]) for t in tasks])
    hits = 0
    for _ in range(n):
        pairs = []
        for t in tasks:
            pool = a[t] + b[t]
            rng.shuffle(pool)
            pairs.append((pool[:len(a[t])], pool[len(a[t]):]))
        hits += stat(pairs) >= observed - 1e-12
    return hits / n


# ---------- 3. after a hold ----------

def hold_funnel(runs, control_runs) -> dict:
    """What the questions after a hold found. 'new' details are also looked up in the no-check control
    batches: how often the same detail of the same task was found there anyway."""
    control = defaultdict(list)
    for r in control_runs:
        control[r["task"]].append(r["found"])
    held = [r for r in runs if r["held"]]
    aligned = [r for r in held if r["after_hold"] is not None]
    qs = [q | {"task": r["task"]} for r in aligned for q in r["after_hold"]["questions"]]
    kinds = Counter(q["kind"] for q in qs)
    new = [q for q in qs if q["kind"] == "new"]
    anyway = [sum(q["blocker"] in f for f in control[q["task"]]) / len(control[q["task"]]) for q in new if control[q["task"]]]
    by_check = defaultdict(Counter)
    for q in qs:
        by_check[q["checks"]][q["kind"]] += 1
    new_runs = [r for r in aligned if any(q["kind"] == "new" for q in r["after_hold"]["questions"])]
    return {
        "held_runs": len(held), "aligned_runs": len(aligned),
        "questions_after_hold": len(qs), "kinds": dict(kinds),
        "new_details": len(new),
        "new_found_anyway_in_control": round(100 * sum(anyway) / len(anyway), 1) if anyway else None,
        "runs_with_new_detail": len(new_runs),
        "runs_with_new_detail_passed": sum(r["resolved"] for r in new_runs),
        "held_runs_passed": sum(r["resolved"] for r in held),
        "by_check": {k: dict(v) for k, v in sorted(by_check.items())},
    }


# ---------- 4. the 75 hand-labeled misses ----------

def failure_modes(batches_by_group: dict[str, list[list[dict]]]) -> dict:
    """For each failure mode, the share of tries in which a version found the same detail of the same task.
    v0's own rate is 0 by construction (the labels are its misses), so the no-check batches v1/v1 pinned/v1b
    are the fair control."""
    rows = [r for r in csv.DictReader((l for l in (ROOT / "analysis" / "failure_labels_step4f.tsv").open()
                                       if not l.startswith("#")), delimiter="\t")]
    out = {}
    for group, batches in batches_by_group.items():
        found = defaultdict(list)
        for runs in batches:
            for r in runs:
                found[r["task"]].append(r["found"])
        per_label = defaultdict(list)
        for row in rows:
            tries = found[row["task"]]
            per_label[row["label"]].append(sum(row["blocker"] in f for f in tries) / len(tries))
        details = defaultdict(set)
        for row in rows:
            details[row["label"]].add((row["task"], row["blocker"]))
        out[group] = {k: {"n": len(v), "details": len(details[k]), "found_pct": round(100 * sum(v) / len(v), 1)}
                      for k, v in per_label.items()}
        allv = [x for v in per_label.values() for x in v]
        out[group]["ALL"] = {"n": len(allv), "details": len({(r["task"], r["blocker"]) for r in rows}),
                             "found_pct": round(100 * sum(allv) / len(allv), 1)}
    return out


# ---------- report ----------

def fmt(x, nd=1):
    return "—" if x is None else f"{x:.{nd}f}" if isinstance(x, float) else str(x)


def markdown(s) -> str:
    L = ["# Midway analysis (generated by analysis/midway_analysis.py; do not edit)", ""]
    L += ["## 1. Per batch (20 practice tasks × 3 tries unless noted)", "",
          "| Batch | pass@1 | per try | pass@3 | pass^3 | tasks flipping | q/run | wasted q/run | recall | schema | question | business | flags | held runs | cost (list) | q per pass |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, b in s["batches"].items():
        t = b["by_type"]
        cell = lambda k: f"{t[k][0]}/{t[k][1]}" if k in t else "—"
        L.append(f"| {b['label']} | {b['pass_at_1']:.1f} | {'/'.join(map(str, b['pass_at_1_per_try']))} | {b['pass_at_3']:.0f} | "
                 f"{b['pass_pow_3']:.0f} | {b['tasks_flipping']} | {b['questions_per_run']:.1f} | {b['wasted_per_run']:.1f} | "
                 f"{b['recall']:.0f}% | {cell('schema')} | {cell('question')} | {cell('business info')} | {b['flags']} | {b['held_runs']} | "
                 f"{'$' + fmt(b['cost_list'], 2) if b['cost_list'] is not None else '—'} | {fmt(b['questions_per_pass'])} |")
    n = s["noise"]
    L += ["", "## 2. Noise and significance (pass@1 points; tasks resampled, 10,000 draws)", "",
          f"- No-check batches (v0, v1, v1 pinned, v1b): pass@1 {', '.join(fmt(x) for x in n['no_check_pass_at_1'])}; "
          f"spread {fmt(n['no_check_spread'])} points. Largest gap between two no-check batches, bootstrap 95% CI: "
          f"{fmt(n['largest_no_check_gap']['diff'])} [{fmt(n['largest_no_check_gap']['ci95'][0])}, {fmt(n['largest_no_check_gap']['ci95'][1])}].", ""]
    L += ["| Comparison | diff | 95% CI | bootstrap P(diff ≤ 0) | permutation p (one-sided) |", "|---|---|---|---|---|"]
    for k, v in n["comparisons"].items():
        L.append(f"| {k} | {v['diff']:+.1f} | [{v['ci95'][0]:+.1f}, {v['ci95'][1]:+.1f}] | {v['p_not_positive']:.3f} | {fmt(v.get('perm_p'), 3)} |")
    L += ["", "## 3. After a hold (questions asked after the first held submission)", "",
          "| Batch | held runs | q after hold | new detail | repeat | wasted | can't answer | new detail found anyway in no-check runs | runs with new detail (passed) | held runs passed |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for name, h in s["after_hold"].items():
        k = h["kinds"]
        L.append(f"| {s['batches'][name]['label']} | {h['held_runs']} ({h['aligned_runs']} aligned) | {h['questions_after_hold']} | {k.get('new', 0)} | "
                 f"{k.get('repeat', 0)} | {k.get('wasted', 0)} | {k.get('cant_answer', 0)} | {fmt(h['new_found_anyway_in_control'])}% | "
                 f"{h['runs_with_new_detail']} ({h['runs_with_new_detail_passed']}) | {h['held_runs_passed']} |")
    L += ["", "By the checks that held the submission:", ""]
    for name, h in s["after_hold"].items():
        for chk, k in h["by_check"].items():
            L.append(f"- {s['batches'][name]['label']}, {chk}: " + ", ".join(f"{a} {b}" for a, b in sorted(k.items())))
    fm = s["failure_modes"]
    labels = sorted(fm["none"], key=lambda k: (k == "ALL", -fm["none"][k]["n"]))
    L += ["", "## 4. The 75 misses of v0, by failure mode: % of tries in which the same detail was found", "",
          "| Failure mode | misses (distinct details) | no checks (v1, v1 pinned, v1b) | value with no source (v2, v2 pinned, v2b) | similar alternatives (v3) |",
          "|---|---|---|---|---|"]
    for k in labels:
        L.append(f"| {k} | {fm['none'][k]['n']} ({fm['none'][k]['details']}) | {fm['none'][k]['found_pct']}% | {fm['unsourced'][k]['found_pct']}% | {fm['alternatives'][k]['found_pct']}% |")
    L += ["", "## 5. Passes out of 3, per task", "", "| Task | " + " | ".join(s["per_task_order"]) + " |",
          "|---|" + "---|" * len(s["per_task_order"])]
    for t, row in s["per_task"].items():
        L.append(f"| {t} | " + " | ".join(str(row.get(b, "")) for b in s["per_task_order"]) + " |")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true", help="also summarize the held-out batches")
    args = ap.parse_args()
    rng = random.Random(SEED)

    spec = DEV + ([t for t in TEST if (ROOT / t[3]).exists()] if args.test else [])
    runs = {name: load(name, folder) for name, _, _, folder in spec}
    label = {name: lab for name, lab, _, _ in spec}
    group = {name: g for name, _, g, _ in spec}

    batches = {name: summarize(r) | {"label": label[name], "group": group[name], "pass_at_1_ci95": bootstrap_ci(task_runs([r]), rng)}
               for name, r in runs.items()}
    dev = lambda g: [runs[n] for n, _, gg, _ in DEV if gg == g]

    no_check = [batches[n]["pass_at_1"] for n in ("v0", "v1", "v1_pinned", "v1b")]
    pairs = [(a, b) for a in ("v0", "v1", "v1_pinned", "v1b") for b in ("v0", "v1", "v1_pinned", "v1b") if a < b]
    gaps = {f"{a} vs {b}": paired_bootstrap(task_runs([runs[a]]), task_runs([runs[b]]), rng) for a, b in pairs}
    largest = max(gaps.values(), key=lambda v: abs(v["diff"]))
    comps = {
        "v2 family vs no checks": (dev("unsourced"), dev("none")),
        "v3 vs no checks": (dev("alternatives"), dev("none")),
        "v3 vs v2 family": (dev("alternatives"), dev("unsourced")),
        "v2 vs v1 (mixed)": ([runs["v2"]], [runs["v1"]]),
        "v2 pinned vs v1 pinned": ([runs["v2_pinned"]], [runs["v1_pinned"]]),
        "v2b vs v1b": ([runs["v2b"]], [runs["v1b"]]),
    }
    comps["v3 vs v1 pinned"] = ([runs["v3"]], [runs["v1_pinned"]])
    if "v3_test" in runs and "v1_test" in runs:
        comps["held-out: v3 vs v1"] = ([runs["v3_test"]], [runs["v1_test"]])
    if "v2_test" in runs:
        comps["held-out: v2 vs v1"] = ([runs["v2_test"]], [runs["v1_test"]])
        comps["held-out: v3 vs v2"] = ([runs["v3_test"]], [runs["v2_test"]])
    comparisons = {k: paired_bootstrap(task_runs(a), task_runs(b), rng) | {"perm_p": permutation(a, b, rng)}
                   for k, (a, b) in comps.items()}

    control = [r for b in dev("none")[1:] for r in b]  # v1, v1 pinned, v1b: our harness, no checks
    hold_batches = [n for n in ("v2", "v2_pinned", "v2b", "v3", "v2_test", "v3_test") if n in runs]
    held = {n: hold_funnel(runs[n], runs.get("v1_test", []) if n.endswith("_test") else control) for n in hold_batches}

    fm = failure_modes({"none": dev("none")[1:], "unsourced": dev("unsourced"), "alternatives": dev("alternatives")})
    # One batch per version, the latest of each: v1 pinned, v2 pinned, v3.
    fm_latest = failure_modes({"none": [runs["v1_pinned"]], "unsourced": [runs["v2_pinned"]], "alternatives": [runs["v3"]]})

    order = [n for n in runs]
    per = defaultdict(dict)
    for n in order:
        for t, v in per_task(runs[n]).items():
            per[t][n] = sum(v)
    per = dict(sorted(per.items(), key=lambda kv: int(kv[0].rsplit("_", 1)[1])))

    s = {"batches": batches,
         "noise": {"no_check_pass_at_1": no_check, "no_check_spread": max(no_check) - min(no_check),
                   "no_check_pairs": gaps, "largest_no_check_gap": largest, "comparisons": comparisons},
         "after_hold": held, "failure_modes": fm, "failure_modes_latest": fm_latest, "per_task_order": order, "per_task": per}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(s, indent=1, default=lambda o: sorted(o) if isinstance(o, set) else str(o)))
    (OUT / "summary.md").write_text(markdown(s))
    print(markdown(s))


if __name__ == "__main__":
    main()
