"""Start HiL-Bench's business-info server and judge (ask_human) server, unchanged, for our own harness.

Reads .env at the project root and applies configs/judge_config.yaml the same way `hil sql` does
(vendor load_and_apply_judge_config). Runs until Ctrl-C. Scale's own `hil sql` starts its own
servers, so this script is only for our harness and for scripts/check_servers.py.

Note: the judge keeps its question logs in memory. Fetch POST /get_logs before stopping it.
"""

import argparse
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "vendor" / "hil-bench"
sys.path.insert(0, str(VENDOR))
from hil_bench.utils.config_mapping import load_and_apply_judge_config  # noqa: E402


def wait_for(url, timeout_s, proc):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"Server exited early (code {proc.returncode}) while waiting for {url}")
        try:
            urllib.request.urlopen(url, timeout=2)
            return
        except Exception:
            time.sleep(1)
    raise TimeoutError(url)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks-dir", type=Path, default=ROOT / "data" / "sql_tasks")
    parser.add_argument("--judge-port", type=int, default=9521)
    parser.add_argument("--info-port", type=int, default=9531)
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")
    for var in ("OPENROUTER_API_KEY", "LITELLM_USER"):
        if not os.getenv(var, "").strip():
            sys.exit(f"{var} is not set. Copy .env.example to .env and fill it in.")
    load_and_apply_judge_config(ROOT / "configs" / "judge_config.yaml")

    py = str(VENDOR / ".venv" / "bin" / "python")
    info = subprocess.Popen([py, "-m", "hil_bench.business_info_server", "--port", str(args.info_port)], cwd=VENDOR)
    judge = subprocess.Popen(
        [py, "-m", "hil_bench.ask_human_server", "--port", str(args.judge_port), "--tasks-dir", str(args.tasks_dir.resolve())],
        cwd=VENDOR,
    )
    try:
        wait_for(f"http://localhost:{args.info_port}/health", 180, info)
        # The judge has no health route; POST /get_logs answers once Flask is up.
        deadline = time.time() + 60
        while True:
            try:
                urllib.request.urlopen(urllib.request.Request(f"http://localhost:{args.judge_port}/get_logs", data=b"{}", method="POST"), timeout=2)
                break
            except Exception:
                if judge.poll() is not None or time.time() > deadline:
                    raise RuntimeError("Judge server did not start")
                time.sleep(1)
        print(f"READY business_info=http://localhost:{args.info_port} judge=http://localhost:{args.judge_port}/ask", flush=True)
        while info.poll() is None and judge.poll() is None:
            time.sleep(2)
        print("A server exited; shutting down.", flush=True)
    finally:
        for p in (info, judge):
            p.terminate()


if __name__ == "__main__":
    main()
