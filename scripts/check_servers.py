"""Step 1 check: both servers answer (FIRST_EXPERIMENT.md §8, Step 1).

Needs scripts/serve.py running. Sends 1 business-info query and 3 judge questions
(about $0.002 of judge calls):
  1. a blocker's own example question  -> expect the blocker's resolution
  2. an off-topic question             -> expect "irrelevant question"
  3. an "I'm assuming X, right?" form   -> expect rejection per the judge rules
Then prints the judge's log for this check's instance ID.
"""

import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
JUDGE = "http://localhost:9521"
TASK = "public_sql_0"


def post(url, payload):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=200) as r:
        return json.loads(r.read().decode())


def main():
    task_dir = ROOT / "data" / "sql_tasks" / TASK
    registry = json.loads((task_dir / "blocker_registry.json").read_text())["blockers"]
    b = registry[0]

    sys.path.insert(0, str(ROOT / "vendor" / "hil-bench" / "SWE-agent" / "tools" / "get_business_info" / "bin"))
    import os
    os.environ["CHROMA_PATH"] = str(task_dir / "chroma_db")
    info_text = urllib.request.urlopen(
        urllib.request.Request(
            "http://localhost:9531/query",
            data=json.dumps({"database_name": "financial", "search_string": "loan status", "chroma_path": os.environ["CHROMA_PATH"], "collection_name": "business_info", "n_results": 5}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        ),
        timeout=60,
    ).read().decode()
    print("business_info:", "OK" if '"documents"' in info_text else f"UNEXPECTED {info_text[:200]}")

    instance_id = f"{TASK}__check__none__pass_1"
    questions = [
        ("example question", b["example_questions"][0]),
        ("off-topic", "What is the capital of France?"),
        ("assumption form", f"I'm assuming {b['description'][:80].rstrip()} means the obvious thing. Is that correct?"),
    ]
    for label, q in questions:
        resp = post(f"{JUDGE}/ask", {"question": q, "instance_id": instance_id})["response"]
        print(f"\n[{label}] Q: {q}\n  A: {resp[:300]}")
    print(f"\nexpected resolution for {b['id'] if 'id' in b else 'blocker 0'}: {b['resolution'][:300]}")

    logs = post(f"{JUDGE}/get_logs", {})
    entry = logs.get(instance_id) or next((v for k, v in logs.items() if k.startswith(TASK)), None)
    print("\njudge log keys:", list(logs.keys()))
    if entry:
        for q in entry["questions"]:
            print(" -", {k: (str(v)[:80] if isinstance(v, str) else v) for k, v in q.items()})


if __name__ == "__main__":
    main()
