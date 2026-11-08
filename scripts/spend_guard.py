"""Run a command under a dollar budget, then scrub secrets from its outputs.

Spend is counted from two sources:
  - OpenRouter: its own usage counter for our key (GET /api/v1/key). Scale's runner turns its
    per-task cost cap off for models missing from LiteLLM's price table (the Qwen models), so this
    is our cap there. The key also has a hard limit on OpenRouter's side.
  - Vertex AI / other LiteLLM-priced models (--watch-dir): the sum of `instance_cost` in every
    .traj under that folder. Scale's runner fills it in for priced models such as Gemini 3.1 Pro
    and also stops any single task at $2.50. Google has no hard stop, so this sum is our cap.
The guard kills the command's whole process group once the total passes --budget. Both counters
can lag a little, so set the budget with some headroom.

After the command ends, every file under --watch-dir is scanned for the secret values in .env
and any hit is replaced with <REDACTED> (LiteLLM's Gemini-API route puts the key in request
URLs, which can surface in error text that the runner saves).

  python scripts/spend_guard.py --budget 1.00 [--watch-dir runs/x] -- <command> [args...]
  python scripts/spend_guard.py --status
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from dotenv import dotenv_values, load_dotenv

ROOT = Path(__file__).resolve().parents[1]


def key_info():
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/key",
        headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY'].strip()}"},
    )
    return json.loads(urllib.request.urlopen(req, timeout=30).read())["data"]


def traj_cost(watch_dir):
    """Sum of instance_cost over all trajectories under watch_dir (0 if none yet)."""
    total = 0.0
    if watch_dir and watch_dir.exists():
        for f in watch_dir.rglob("*.traj"):
            try:
                total += float(json.loads(f.read_text())["info"]["model_stats"].get("instance_cost") or 0)
            except Exception:  # a trajectory being written mid-read
                pass
    return total


def scrub(watch_dir):
    """Replace .env secret values in any file under watch_dir; return the files changed."""
    secrets = [v.strip().encode() for k, v in dotenv_values(ROOT / ".env").items() if "KEY" in k and v and len(v.strip()) >= 16]
    changed = []
    for f in watch_dir.rglob("*"):
        if not f.is_file():
            continue
        data = f.read_bytes()
        new = data
        for s in secrets:
            new = new.replace(s, b"<REDACTED>")
        if new != data:
            f.write_bytes(new)
            changed.append(str(f))
    return changed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget", type=float, help="Max dollars this command may spend")
    parser.add_argument("--poll", type=float, default=15.0, help="Seconds between usage checks")
    parser.add_argument("--watch-dir", type=Path, help="Run output folder: count its trajectory costs, scrub it afterwards")
    parser.add_argument("--status", action="store_true", help="Print key usage and exit")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    load_dotenv(ROOT / ".env")

    info = key_info()
    if args.status:
        print(f"used ${info['usage']:.4f} of ${info['limit']} (remaining ${info['limit_remaining']:.4f})")
        return
    cmd = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not cmd or args.budget is None:
        parser.error("need --budget and a command after --")

    start = info["usage"]
    start_traj = traj_cost(args.watch_dir)
    print(f"[spend_guard] start OpenRouter usage ${start:.4f}, budget ${args.budget:.2f}", flush=True)
    proc = subprocess.Popen(cmd, start_new_session=True)
    killed = False
    or_spent = 0.0
    while proc.poll() is None:
        time.sleep(args.poll)
        try:
            or_spent = key_info()["usage"] - start
        except Exception as e:  # a failed check shouldn't kill the run
            print(f"[spend_guard] usage check failed: {e}", flush=True)
        spent = or_spent + traj_cost(args.watch_dir) - start_traj
        if spent > args.budget:
            print(f"[spend_guard] spent ${spent:.4f} > budget ${args.budget:.2f}; stopping", flush=True)
            os.killpg(proc.pid, signal.SIGTERM)
            killed = True
            break
    proc.wait()
    time.sleep(5)  # let the last calls reach OpenRouter's counter
    or_spent = key_info()["usage"] - start
    other = traj_cost(args.watch_dir) - start_traj
    print(f"[spend_guard] done; spent ${or_spent + other:.4f} (OpenRouter ${or_spent:.4f}, trajectories ${other:.4f})"
          f"{' (stopped at budget)' if killed else ''}", flush=True)
    if args.watch_dir and args.watch_dir.exists():
        changed = scrub(args.watch_dir)
        print(f"[spend_guard] scrubbed secrets from {len(changed)} file(s)" + (f": {changed[:5]}" if changed else ""), flush=True)
    sys.exit(2 if killed else proc.returncode)


if __name__ == "__main__":
    main()
