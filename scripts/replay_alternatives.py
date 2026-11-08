"""Replay the "similar alternatives" check offline on finished Scale-harness runs (no model calls).

For each submitted query, run the check and score each flag against the task's schema-type hidden
details (blocker registry): a flag is "relevant" if a table or column it names appears in one of
them. A missed schema detail (analysis/failure_labels_step4f.tsv) counts as caught if a relevant flag
fired on that run. Several threshold settings are compared.

  python scripts/replay_alternatives.py runs/step4f_qwen235b_fixed/both_20x3 --out runs/replay_alternatives
"""

import argparse
import csv
import glob
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hilx.adapters import ADAPTERS  # noqa: E402
from hilx.checks import similar_alternatives  # noqa: E402
from hilx.core.evidence import evidence_from_sweagent_traj  # noqa: E402

SQL = ADAPTERS["sql"]
TASKS = ROOT / "data" / "sql_tasks"
SETTINGS = {
    "v3": dict(),
    "v3b": dict(cross_word_sim=0.5, include_used=True, min_description_words=1),
}


def schema_blockers(task: str) -> dict[str, str]:
    reg = json.loads((TASKS / task / "blocker_registry.json").read_text())["blockers"]
    reg = reg.values() if isinstance(reg, dict) else reg
    return {b["id"]: b["description"] for b in reg if b["type"] == "schema"}


def relevant(flag_message: str, blockers: dict[str, str]) -> set[str]:
    """A flag is relevant to a hidden detail if the detail's description names the flagged entity AND at
    least one of the look-alikes listed (column or table names). Naming either alone is too loose."""
    labels = re.findall(r"`([^`]+)`", flag_message)
    first, alts = labels[0], labels[1:]
    def named(label, d):
        return any(re.search(r"(?<![a-z0-9_])" + re.escape(n.lower()) + r"(?![a-z0-9_])", d.lower()) for n in label.split(".")[-1:])
    return {bid for bid, d in blockers.items() if named(first, d) and any(named(a, d) for a in alts)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    results = {i["instance_id"]: i for i in json.loads((args.run_dir / "consolidated_results.json").read_text())["instances"]}
    with open(ROOT / "analysis" / "failure_labels_step4f.tsv") as f:
        labels = [r for r in csv.DictReader((l for l in f if not l.startswith("#")), delimiter="\t") if r["blocker_type"] == "schema"]
    catalogs, rows = {}, []
    for traj_path in sorted(glob.glob(str(args.run_dir / "*" / "*" / "pass_*" / "*" / "*.traj"))):
        r = results[Path(traj_path).stem]
        task = r["original_instance_id"]
        catalog = catalogs.setdefault(task, SQL.catalog(TASKS / task))
        evidence = evidence_from_sweagent_traj(json.loads(Path(traj_path).read_text()), SQL.tool_sources)
        blockers = schema_blockers(task)
        for name, opts in SETTINGS.items():
            flags, parsed = similar_alternatives.check(r["output"] or "", evidence, SQL, {"catalog": catalog}, **opts)
            rows.append({"task": task, "mode": r["mode"], "pass": r["pass"], "resolved": r["resolved"], "setting": name,
                         "parsed": parsed, "flags": [{"message": f.message, "found_in": f.found_in,
                                                      "relevant_to": sorted(relevant(f.message, blockers))} for f in flags]})
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "flags.jsonl").write_text("".join(json.dumps(x) + "\n" for x in rows))

    for name in SETTINGS:
        print(f"\n=== {name}: {SETTINGS[name]}")
        for mode in ("ask_human", "full_info"):
            rs = [x for x in rows if x["setting"] == name and x["mode"] == mode]
            fl = [f for x in rs for f in x["flags"]]
            rel = sum(bool(f["relevant_to"]) for f in fl)
            print(f"{mode:9s} runs={len(rs)} flags={len(fl)} ({len(fl) / len(rs):.1f}/run) relevant={rel} "
                  f"irrelevant={len(fl) - rel} flags_on_passing_runs={sum(len(x['flags']) for x in rs if x['resolved'])} "
                  f"unseen={sum(f['found_in'] == 'unseen' for f in fl)} unparsed={sum(not x['parsed'] for x in rs)}")
        by_run = {(x["task"], x["pass"]): x for x in rows if x["setting"] == name and x["mode"] == "ask_human"}
        caught = [l for l in labels if l["blocker"] in {b for f in by_run[(l["task"], int(l["try"]))]["flags"] for b in f["relevant_to"]}]
        print(f"missed schema details caught: {len(caught)}/{len(labels)}  by label: "
              + ", ".join(f"{k} {sum(l['label'] == k for l in caught)}/{sum(l['label'] == k for l in labels)}" for k in sorted({l['label'] for l in labels})))
        print("  not caught:", sorted({(l["task"][11:], l["blocker"]) for l in labels if l not in caught}))
    print(f"\nWrote {args.out / 'flags.jsonl'}")


if __name__ == "__main__":
    main()
