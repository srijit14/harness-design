"""Replay a check offline on finished runs of Scale's harness, before building it into our harness.

For each run: rebuild the evidence log from the .traj, run the check on the submitted SQL, and
score the flags two ways:
  - recall: of the values the agent made up (found by hand in the traces, MADE_UP below), how
    many were flagged;
  - usefulness: a flag on a value that the gold SQL also uses means the agent was right, so
    asking would have spent a question for nothing ("wasted"); otherwise "useful".
Full-info runs are a control: the task text carries every answer, so flags there should be rare.

  python scripts/replay_checks.py runs/step4f_qwen235b_fixed/both_20x3 --out runs/replay_unsourced
"""

import argparse
import collections
import glob
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hilx.adapters import ADAPTERS  # noqa: E402
from hilx.checks import unsourced_values  # noqa: E402
from hilx.core.values import number  # noqa: E402
from hilx.core.evidence import evidence_from_sweagent_traj  # noqa: E402

SQL = ADAPTERS["sql"]

# Values the agent made up in the asking runs of step4f, read by hand from each trace and final SQL
# (task, try) -> values. Only literals count; guesses made with logic and no literal can't be listed.
# Left out on purpose: '+' in public_sql_41 (the column description defines it; the mistake was the
# column, not the value).
MADE_UP = {
    ("public_sql_7", 1): ["0.9", "A", "4743"],
    ("public_sql_23", 3): ["10"],
    ("public_sql_38", 1): ["1998", "west Bohemia"],
    ("public_sql_38", 2): ["1998", "POPLATEK PO OBRATU", "west Bohemia"],
    ("public_sql_38", 3): ["1998"],
    ("public_sql_41", 1): ["250", "1993-12-31"],
    ("public_sql_41", 3): ["125", "220", "1995-01-01"],
    ("public_sql_45", 1): ["98684.7030254777"],
    ("public_sql_45", 3): ["1995-01-01", "1998-12-31"],
    ("public_sql_47", 2): ["35", "2012-01-01", "0.042"],
    ("public_sql_47", 3): ["500", "2007-01-01"],
    ("public_sql_68", 1): ["0.5", "2017"],
    ("public_sql_68", 2): ["2017"],
}


def norm(text: str) -> str:
    n = number(text)
    return repr(n) if n is not None else text.lower()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    results = {i["instance_id"]: i for i in json.loads((args.run_dir / "consolidated_results.json").read_text())["instances"]}
    gold = {i["problem_statement"]["id"]: i["problem_statement"]["ground_truth_query"] for i in json.loads((ROOT / "data/sql_tasks/instances.json").read_text())}

    rows = []
    for traj_path in sorted(glob.glob(str(args.run_dir / "*" / "*" / "pass_*" / "*" / "*.traj"))):
        iid = Path(traj_path).stem
        r = results[iid]
        traj = json.loads(Path(traj_path).read_text())
        evidence = evidence_from_sweagent_traj(traj, SQL.tool_sources)
        gold_values = {norm(v.text) for v in SQL.extract_values(gold[r["original_instance_id"]])[0]}
        made_up = {norm(v) for v in MADE_UP.get((r["original_instance_id"], r["pass"]), [])} if r["mode"] == "ask_human" else set()
        for count_data in (True, False):
            flags, parsed = unsourced_values.check(r["output"] or "", evidence, SQL, count_data=count_data)
            flagged = {norm(v.text) for f in flags for v in f.values}
            rows.append({
                "task": r["original_instance_id"], "mode": r["mode"], "pass": r["pass"], "resolved": r["resolved"],
                "setting": "data counts" if count_data else "data does not count", "parsed": parsed,
                "flags": [{"values": [v.text for v in f.values], "found_in": f.found_in, "context": f.values[0].context,
                           "in_gold": all(norm(v.text) in gold_values for v in f.values)} for f in flags],
                "made_up": sorted(made_up), "made_up_flagged": sorted(made_up & flagged),
            })

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "flags.jsonl").write_text("".join(json.dumps(x) + "\n" for x in rows))

    for setting in ("data counts", "data does not count"):
        print(f"\n=== setting: {setting}")
        for mode in ("ask_human", "full_info"):
            rs = [x for x in rows if x["setting"] == setting and x["mode"] == mode]
            flags = [f for x in rs for f in x["flags"]]
            useful = sum(not f["in_gold"] for f in flags)
            line = (f"{mode:9s} runs={len(rs)} flags={len(flags)} ({len(flags) / len(rs):.1f}/run) "
                    f"runs_flagged={sum(bool(x['flags']) for x in rs)} useful={useful} wasted={len(flags) - useful} "
                    f"flags_on_passing_runs={sum(len(x['flags']) for x in rs if x['resolved'])} unparsed={sum(not x['parsed'] for x in rs)}")
            if mode == "ask_human":
                mu = sum(len(x["made_up"]) for x in rs)
                line += f" | made-up values flagged {sum(len(x['made_up_flagged']) for x in rs)}/{mu}"
            print(line)
        missed = collections.Counter()
        for x in rows:
            if x["setting"] == setting:
                for v in set(x["made_up"]) - set(x["made_up_flagged"]):
                    missed[(x["task"], x["pass"], v)] += 1
        print("  made-up values NOT flagged:", sorted(missed))
    print(f"\nWrote {args.out / 'flags.jsonl'}")


if __name__ == "__main__":
    main()
