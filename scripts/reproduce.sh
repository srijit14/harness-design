#!/usr/bin/env bash
# Rerun every batch behind the results, then rebuild the tables and figures.
#
#   scripts/reproduce.sh            all 13 batches (about $20 and 7 hours on OpenRouter), then the analysis
#   scripts/reproduce.sh analysis   tables and figures only, from the run folders already in runs/ (seconds, free)
#
# Batches run one at a time, each under a dollar cap (spend_guard.py). A finished run is skipped, so rerunning
# resumes where it stopped. Run setup.sh first.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=vendor/hil-bench/.venv/bin/python
MODEL=openrouter/qwen/qwen3-235b-a22b-2507
DEV=data/sql_tasks/instances.json
HELD_OUT=data/sql_tasks/instances_test_small.json
PINNED=configs/routing/parasail_fp8.yaml
[ -x "$PY" ] || { echo "Run scripts/setup.sh first."; exit 1; }

guard() { $PY scripts/spend_guard.py --budget "$1" --watch-dir "$2" -- "${@:3}"; }
version() {  # version() <budget> <version> <out> [run_version.py options]
  echo "== $2 -> $3"
  guard "$1" "$3" $PY scripts/run_version.py "configs/versions/$2.yaml" --out "$3" "${@:4}"
}
scale() {  # scale() <budget> <out dir> <run name> <tasks> [hil_sql.py options]: Scale's own harness
  echo "== Scale's harness -> $2/$3"
  [ -f "$2/$3/consolidated_metrics.json" ] && { echo "already done"; return; }  # Scale's runner redoes finished runs
  guard "$1" "$2" $PY scripts/hil_sql.py "$4" -m "$MODEL" -w 6 --run-name "$3" --output-dir "$2" \
    --config-mapping configs/config_mappings.yaml --judge-config configs/judge_config.yaml "${@:5}"
}

if [ "${1:-all}" != analysis ]; then
  # Dev: 20 tasks x 3 tries. Scale's harness (asking + full info), then v1, v2, v3 and the controls.
  scale 5 runs/step4f_qwen235b_fixed both_20x3 "$DEV" -k 3 --all-modes
  version 5 v1_native runs/v1_native/dev_20x3_parasail --routing "$PINNED"
  version 5 v2_unsourced runs/v2_unsourced/dev_20x3_parasail --routing "$PINNED"
  version 5 v3_alternatives runs/v3_alternatives/dev_20x3
  version 5 v1_native runs/v1_native/dev_20x3              # same versions on mixed routing: noise estimate
  version 5 v2_unsourced runs/v2_unsourced/dev_20x3
  version 5 v1b_open_questions runs/v1b_open_questions/dev_20x3   # "ask better questions" prompt rule
  version 5 v2b_unsourced_open runs/v2b_unsourced_open/dev_20x3

  # Held-out: 39 tasks x 3 tries, run once the versions were frozen; full info x 1.
  version 12 v1_native runs/v1_native/test_small_39x3 --tasks "$HELD_OUT"
  version 12 v2_unsourced runs/v2_unsourced/test_small_39x3 --tasks "$HELD_OUT"
  version 12 v3_alternatives runs/v3_alternatives/test_small_39x3 --tasks "$HELD_OUT"
  scale 8 runs/step4g_fullinfo_test fullinfo_39x1 "$HELD_OUT" -k 1 --full-info
fi

echo "== Tables and statistics -> analysis/midway/summary.md"
$PY analysis/midway_analysis.py --test > /dev/null
echo "== Figures -> analysis/midway/figures/"
env -u PYTHONPATH uv run -q --no-project --python 3.11 --with matplotlib python -I analysis/midway_figures.py
