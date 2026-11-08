"""Download and prepare SQL tasks with HiL-Bench's own prepare_sql_task (unchanged).

Writes data/sql_tasks/<task_id>/ (DB, schema CSVs, blocker_registry.json, chroma_db/)
and data/sql_tasks/instances.json, the file `hil sql` reads.
"""

import argparse
import json
import sys
from pathlib import Path

from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor" / "hil-bench"))
from run_hil_bench import prepare_sql_task  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="dev", help="dev, test or all (from configs/split.json)")
    parser.add_argument("--tasks", nargs="*", help="Explicit task IDs; overrides --split")
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "sql_tasks")
    args = parser.parse_args()

    split = json.loads((ROOT / "configs" / "split.json").read_text())
    wanted = args.tasks or (split["dev"] + split["test"] if args.split == "all" else split[args.split])
    rows = {r["task_id"]: dict(r) for r in load_dataset(split["dataset"], split="train", revision=split["dataset_revision"])}

    args.out.mkdir(parents=True, exist_ok=True)
    instances_path = args.out / "instances.json"
    instances = {i["problem_statement"]["id"]: i for i in json.loads(instances_path.read_text())} if instances_path.exists() else {}
    for n, task_id in enumerate(wanted, 1):
        if task_id in instances and (args.out / task_id / "chroma_db").exists():
            print(f"[{n}/{len(wanted)}] {task_id}: already prepared")
            continue
        print(f"[{n}/{len(wanted)}] {task_id}: preparing", flush=True)
        _, instances[task_id] = prepare_sql_task(rows[task_id], args.out)

    # Drop the empty extra_fields that prepare_sql_task writes. Scale's runner expands each task into
    # one copy per model/mode/pass with a shallow copy, so every copy shares that one dict and ends up
    # with the last copy's agent config, model and TASK_INSTANCE_ID (hil_bench/scripts/swe.py
    # create_mixed_mode_instances, `ps.get("extra_fields", {})`). Without the key, each copy gets its own.
    for i in instances.values():
        if i["problem_statement"].get("extra_fields") == {}:
            del i["problem_statement"]["extra_fields"]
    ordered = sorted(instances.values(), key=lambda i: int(i["problem_statement"]["id"].rsplit("_", 1)[1]))
    instances_path.write_text(json.dumps(ordered, indent=2))
    print(f"Wrote {len(ordered)} instances to {instances_path}")


if __name__ == "__main__":
    main()
