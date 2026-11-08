"""Launch Scale's `hil sql` with two runner bugs worked around, and optional resume. Use it for every run.

Takes the same arguments as `hil sql`, plus --resume. No vendor file is changed; overrides happen at call time.

1. Business info per task. Scale's runner sets one CHROMA_PATH for the whole batch, taken from the
   first task (hil_bench/scripts/swe.py, "Using ChromaDB from"). Every task then searches task 1's
   business-info index, so tasks on other databases always get "No business information found",
   and same-database tasks get task 1's notes. If CHROMA_BASE_PATH is set, the runner keeps it, and
   the get_business_info tool resolves CHROMA_BASE_PATH/<task_id> per run. This script points it at
   data/chroma_by_task/<task_id> -> data/sql_tasks/<task_id>/chroma_db links (built here).
2. Resume (--resume). `hil sql` always passes redo_existing=True, and create_results_directory
   switches to `<run_name>_<timestamp>/` whenever `<run_name>/` already holds trajectories. With
   --resume, both are overridden, so SWE-agent skips finished runs and redoes only unfinished ones in
   `<run_name>/`. The judge keeps question logs in memory, so a resumed run's ask metrics cover only
   the redone runs. Rescue the old logs (POST /get_logs) before stopping a run, then merge them.

The shared extra_fields bug (every model/mode/pass copy getting the last copy's config) is fixed in
the task files by scripts/materialize.py.

  python scripts/hil_sql.py [--resume] <same args as hil sql>
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hilx.scale import link_chroma_by_task  # noqa: E402  (also puts vendor/hil-bench on the path)
import hil_bench.scripts.batch_runner as batch_runner  # noqa: E402
import hil_bench.scripts.sql as sql_script  # noqa: E402
from hil_bench.cli import main  # noqa: E402


def enable_resume():
    run_batch = sql_script._run_batch

    def run_batch_resume(*args, **kwargs):
        kwargs["redo_existing"] = False
        return run_batch(*args, **kwargs)

    def existing_results_directory(run_name, base_output_dir=None):
        results_dir = (base_output_dir or Path("results")) / run_name
        results_dir.mkdir(parents=True, exist_ok=True)
        print(f"[resume] reusing {results_dir}", flush=True)
        return results_dir

    sql_script._run_batch = run_batch_resume
    batch_runner.create_results_directory = existing_results_directory


if __name__ == "__main__":
    args = sys.argv[1:]
    if "--resume" in args:
        args.remove("--resume")
        enable_resume()
    link_chroma_by_task()
    sys.argv = ["hil", "sql", *args]
    main()
