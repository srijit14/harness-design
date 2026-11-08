"""Run one harness version on a task set, N tries each, and score it.

Starts the business-info and judge servers (scripts/serve.py, Scale's servers unchanged), runs every
(task, try) in parallel, rescues the judge's question logs, and writes:
  <out>/version.yaml, run_info.json      the version file as run, git commit, config fingerprint
  <out>/pass_<n>/<task>/traj.json        messages, evidence log, checkpoints
  <out>/pass_<n>/<task>/result.json      submission, scores (ours and Scale's), cost, flags
  <out>/ask_human_logs.json              the judge's logs (which question found which hidden detail)
  <out>/metrics.json                     pass@1, pass@3, questions, precision/recall, flags
Finished runs (result.json present) are skipped, so rerunning the same command resumes.

  python scripts/spend_guard.py --budget 5 --watch-dir runs/v1 -- \
    vendor/hil-bench/.venv/bin/python scripts/run_version.py configs/versions/v1_native.yaml --out runs/v1
"""

import argparse
import hashlib
import json
import os
import socket
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hilx.scale import link_chroma_by_task  # noqa: E402
from hilx.run import TASKS_DIR, run_one  # noqa: E402
from hil_bench.utils.compute_hil_metrics import compute_hil_metrics  # noqa: E402


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def start_servers(log_path: Path) -> tuple[subprocess.Popen, int, int]:
    # Server output goes to a file, not a pipe: a pipe nobody reads fills up after ~64 KB of request
    # logs and every server thread then blocks on write (this hung our first v1/v2 runs at 20:33).
    judge_port, info_port = free_port(), free_port()
    log = open(log_path, "a")
    proc = subprocess.Popen(
        [sys.executable, str(ROOT / "scripts" / "serve.py"), "--judge-port", str(judge_port), "--info-port", str(info_port)],
        stdout=log, stderr=subprocess.STDOUT, text=True,
    )
    deadline = time.time() + 300
    while time.time() < deadline and proc.poll() is None:
        if "READY" in log_path.read_text():
            return proc, judge_port, info_port
        time.sleep(1)
    proc.terminate()
    raise RuntimeError(f"servers did not start; see {log_path}")


def servers_ok(judge_port: int, info_port: int) -> bool:
    try:
        urllib.request.urlopen(f"http://localhost:{info_port}/health", timeout=10)
        get_logs(judge_port)
        return True
    except Exception:
        return False


def get_logs(judge_port: int) -> dict:
    req = urllib.request.Request(f"http://localhost:{judge_port}/get_logs", data=b"{}", method="POST", headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=60))


def git_state() -> dict:
    def git(*a):
        return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    return {"commit": git("rev-parse", "HEAD"), "dirty_files": git("status", "--porcelain", "hilx", "configs", "scripts").splitlines()}


def metrics(results: list[dict], logs: dict, passes: int) -> dict:
    by_pass = {p: [r for r in results if r["pass"] == p] for p in range(1, passes + 1)}
    tasks = sorted({r["task"] for r in results})
    solved_any = sum(any(r["resolved"] for r in results if r["task"] == t) for t in tasks)
    own_logs = {k: v for k, v in logs.items() if k in {r["run_id"] for r in results}}
    hil = {k: v for k, v in compute_hil_metrics(json.loads(json.dumps(own_logs))).items() if k != "instances"} if own_logs else {}
    n_q = sum(len(v["questions"]) for v in own_logs.values())
    flags = [f for r in results for c in r["checkpoints"] for f in c["flags"]]
    return {
        "runs": len(results),
        "pass_at_1_per_try": {p: sum(r["resolved"] for r in rs) for p, rs in by_pass.items()},
        "pass_at_1": round(100 * sum(r["resolved"] for r in results) / len(results), 1),
        "pass_at_1_scale_scoring": round(100 * sum(r["resolved_scale_scoring"] for r in results) / len(results), 1),
        "pass_at_3": round(100 * solved_any / len(tasks), 1),
        "questions_per_run": round(n_q / len(results), 2),
        "hil_metrics": hil,
        "exit_statuses": {s: sum(r["exit_status"] == s for r in results) for s in sorted({r["exit_status"] for r in results})},
        "endofsql_stripped": sum(r["endofsql_stripped"] for r in results),
        "flags": len(flags), "runs_with_held_submission": sum(any(c["held"] for c in r["checkpoints"]) for r in results),
        "cost": round(sum(r["cost"] for r in results), 4),  # as billed; depends on provider and cache hits
        "cost_list": round(sum(r.get("cost_list", 0) for r in results), 4),  # tokens at the fixed price list: compare this
        "tokens": {k: sum(r.get(k, 0) for r in results) for k in ("tokens_in", "tokens_cached", "tokens_out")},
        "providers": {p: sum(r.get("providers", {}).get(p, 0) for r in results) for p in sorted({p for r in results for p in r.get("providers", {})})},
        "runs_with_server_errors": [r["run_id"] for r in results if r.get("server_errors")],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("version", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tasks", type=Path, default=TASKS_DIR / "instances.json")
    ap.add_argument("--passes", type=int, default=3)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--routing", type=Path, default=ROOT / "configs" / "routing" / "fp8_any.yaml",
                    help="OpenRouter provider pin and call timeout for the agent model")
    args = ap.parse_args()

    load_dotenv(ROOT / ".env")
    version_text = args.version.read_text()
    version = yaml.safe_load(version_text)
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out / "version.yaml").exists() and (args.out / "version.yaml").read_text() != version_text:
        sys.exit(f"{args.out} was run with a different version file; use a new --out")
    (args.out / "version.yaml").write_text(version_text)
    agent_cfg = (ROOT / version["agent_config"]).read_text()
    routing = yaml.safe_load(args.routing.read_text())
    # Runs made before routing was pinned have no "routing" here; never mix pinned and unpinned runs.
    old_info = json.loads((args.out / "run_info.json").read_text()) if (args.out / "run_info.json").exists() else None
    if old_info and old_info.get("routing") != routing:
        sys.exit(f"{args.out} was run with different model routing; use a new --out")
    info = {"version": version["name"], "git": git_state(), "tasks_file": str(args.tasks), "passes": args.passes,
            "routing": routing, "fingerprint": hashlib.sha256((version_text + agent_cfg).encode()).hexdigest()[:16]}
    (args.out / "run_info.json").write_text(json.dumps(info, indent=2))

    instances = json.loads(args.tasks.read_text())
    jobs = [(i, p) for p in range(1, args.passes + 1) for i in instances]
    todo = [(i, p) for i, p in jobs if not (args.out / f"pass_{p}" / i["problem_statement"]["id"] / "result.json").exists()]
    print(f"{version['name']}: {len(jobs)} runs, {len(jobs) - len(todo)} already done", flush=True)

    logs_path = args.out / "ask_human_logs.json"
    old_logs = json.loads(logs_path.read_text()) if logs_path.exists() else {}
    servers, judge_port, info_port = start_servers(args.out / "servers.log")
    try:
        link_chroma_by_task()
        os.environ.update({
            "ASK_HUMAN_SERVER_URL": f"http://localhost:{judge_port}/ask",
            "GET_BUSINESS_INFO_SERVER_URL": f"http://localhost:{info_port}",
            "DATABASE_DESCRIPTIONS_BASE_DIR": str(TASKS_DIR),
            "CHROMA_COLLECTION_NAME": "business_info",
        })
        with ThreadPoolExecutor(args.workers) as pool:
            futures = {pool.submit(run_one, i, version, p, args.out, routing): (i["problem_statement"]["id"], p) for i, p in todo}
            for n, fut in enumerate(as_completed(futures), 1):
                task, p = futures[fut]
                try:
                    r = fut.result()
                    print(f"[{n}/{len(todo)}] {task} try {p}: {'PASS' if r['resolved'] else 'fail'} ({r['exit_status']}, "
                          f"{r['model_calls']} calls, held {sum(c['held'] for c in r['checkpoints'])}x, server errors {r['server_errors']})", flush=True)
                except Exception as e:
                    print(f"[{n}/{len(todo)}] {task} try {p}: CRASHED {type(e).__name__}: {e}", flush=True)
                # A run whose tools failed is not a valid run: stop the batch rather than keep going blind.
                if not servers_ok(judge_port, info_port):
                    print("Servers stopped answering; cancelling the remaining runs. See servers.log.", flush=True)
                    for f in futures:
                        f.cancel()
                    break
                # Save the judge's logs as we go: they live only in the server's memory.
                logs_path.write_text(json.dumps(old_logs | get_logs(judge_port), indent=2))
    finally:
        servers.terminate()

    results = [json.loads(f.read_text()) for f in sorted(args.out.glob("pass_*/*/result.json"))]
    logs = json.loads(logs_path.read_text()) if logs_path.exists() else {}
    missing = [r["run_id"] for r in results if r["run_id"] not in logs]
    m = metrics(results, logs, args.passes)
    m["runs_missing_judge_log"] = missing  # runs that asked nothing have no log entry
    (args.out / "metrics.json").write_text(json.dumps(m, indent=2, default=str))
    print(json.dumps({k: v for k, v in m.items() if k != "hil_metrics"}, indent=1, default=str))


if __name__ == "__main__":
    main()
