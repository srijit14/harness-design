"""Pick the 20 practice tasks and write configs/split.json.

Method (our choice, FIRST_EXPERIMENT.md Step 1):
- Only tasks on the 5 small databases (each DB file < 100 MB).
- Quota per database is proportional to its task count (largest remainder,
  ties broken by the seeded shuffle).
- Within the quotas, tasks are picked greedily to cover as many blocker
  subtypes as possible; ties are broken by the seeded shuffle.
- Every other SQL task (small or large DB) goes to the test set.
"""

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor" / "hil-bench"))
from run_hil_bench import normalize_blocker_registry  # noqa: E402

HF_DATASET = "ScaleAI/hil-bench"
HF_REVISION = "045419da0f51f10efc621efdb481a6502df810d5"
SMALL_DBS = ["california_schools", "financial", "formula_1", "thrombosis_prediction", "toxicology"]


def subtypes(row):
    return {b["subtype"] for b in normalize_blocker_registry(row["blocker_registry"])["blockers"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-dev", type=int, default=20)
    parser.add_argument("--seed", type=int, default=329)
    parser.add_argument("--out", type=Path, default=ROOT / "configs" / "split.json")
    args = parser.parse_args()

    rows = [dict(r) for r in load_dataset(HF_DATASET, split="train", revision=HF_REVISION)]
    sql = [r for r in rows if r["task_type"].lower() == "sql"]
    rng = random.Random(args.seed)
    small = [r for r in sql if r["repo_or_db_name"] in SMALL_DBS]
    rng.shuffle(small)

    by_db = defaultdict(list)
    for r in small:
        by_db[r["repo_or_db_name"]].append(r)
    exact = {db: args.n_dev * len(by_db[db]) / len(small) for db in SMALL_DBS}
    quota = {db: int(exact[db]) for db in SMALL_DBS}
    order = sorted(SMALL_DBS, key=lambda db: (-(exact[db] - quota[db]), rng.random()))
    for db in order[: args.n_dev - sum(quota.values())]:
        quota[db] += 1

    covered, dev = set(), []
    remaining = {db: list(by_db[db]) for db in SMALL_DBS}
    while len(dev) < args.n_dev:
        for db in SMALL_DBS:
            if sum(r["repo_or_db_name"] == db for r in dev) >= quota[db]:
                continue
            best = max(remaining[db], key=lambda r: len(subtypes(r) - covered))
            remaining[db].remove(best)
            dev.append(best)
            covered |= subtypes(best)
            if len(dev) == args.n_dev:
                break

    dev_ids = sorted((r["task_id"] for r in dev), key=lambda t: int(t.rsplit("_", 1)[1]))
    test_ids = sorted((r["task_id"] for r in sql if r["task_id"] not in dev_ids), key=lambda t: int(t.rsplit("_", 1)[1]))
    all_subtypes = set().union(*(subtypes(r) for r in sql))
    dev_counts = Counter(s for r in dev for b in normalize_blocker_registry(r["blocker_registry"])["blockers"] for s in [b["subtype"]])

    out = {
        "dataset": HF_DATASET,
        "dataset_revision": HF_REVISION,
        "seed": args.seed,
        "method": __doc__.strip(),
        "dev": dev_ids,
        "test": test_ids,
        "dev_per_db": {db: quota[db] for db in SMALL_DBS},
        "dev_blockers": sum(dev_counts.values()),
        "dev_subtype_counts": dict(sorted(dev_counts.items())),
        "subtypes_missing_from_dev": sorted(all_subtypes - covered),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k not in ("method", "test")}, indent=2))
    print(f"test: {len(test_ids)} tasks")


if __name__ == "__main__":
    main()
