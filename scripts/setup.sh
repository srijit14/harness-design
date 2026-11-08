#!/usr/bin/env bash
# One-time setup: pinned third-party code, one Python environment, and the 20 dev + 39 held-out tasks.
# Safe to rerun; finished steps are skipped. Needs git, uv and an OPENROUTER_API_KEY in .env.
set -euo pipefail
cd "$(dirname "$0")/.."

HIL_COMMIT=c5a656e41b5c50917686f9df8e4c72152bf27ba9
MINI_COMMIT=04d809ceab9df28f9adaed044884180159172930
PY=vendor/hil-bench/.venv/bin/python
TASKS=data/sql_tasks

[ -f .env ] || { cp .env.example .env; echo "Created .env: fill in OPENROUTER_API_KEY and LITELLM_USER, then rerun."; exit 1; }
for k in OPENROUTER_API_KEY LITELLM_USER; do
  grep -q "^$k=." .env || { echo "Set $k in .env, then rerun."; exit 1; }
done

echo "== 1/3 Pinned code: HiL-Bench and mini-swe-agent (neither is edited)"
clone() { [ -d "$2" ] || git clone -q "$1" "$2"; git -C "$2" checkout -q "$3"; }
clone https://github.com/hilbenchauthors/hil-bench vendor/hil-bench "$HIL_COMMIT"
clone https://github.com/SWE-agent/mini-swe-agent vendor/mini-swe-agent "$MINI_COMMIT"

echo "== 2/3 Python environment (HiL-Bench's, plus mini-swe-agent and the SQL parser)"
(cd vendor/hil-bench && uv sync -q --python 3.11)
uv pip install -q --python "$PY" ./vendor/mini-swe-agent sqlglot==26.6.0

echo "== 3/3 Tasks from HF ScaleAI/hil-bench at a pinned revision (about 15 s each)"
$PY scripts/materialize.py --split dev
if [ ! -f "$TASKS/instances_test_small.json" ]; then
  cp "$TASKS/instances.json" "$TASKS/instances_dev.json"
  $PY scripts/materialize.py --tasks $($PY -c "import json; print(' '.join(json.load(open('configs/test_small_tasks.json'))))")
  $PY - <<'EOF'
import json, shutil
want = set(json.load(open("configs/test_small_tasks.json")))
every = json.load(open("data/sql_tasks/instances.json"))
json.dump([i for i in every if i["problem_statement"]["id"] in want], open("data/sql_tasks/instances_test_small.json", "w"), indent=2)
shutil.copy("data/sql_tasks/instances_dev.json", "data/sql_tasks/instances.json")
print(f"dev: {len(json.load(open('data/sql_tasks/instances.json')))} tasks, held-out: {len(want)} tasks")
EOF
fi

echo "Done. Next: scripts/reproduce.sh"
