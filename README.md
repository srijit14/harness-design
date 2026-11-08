# Checks at submit: a harness that makes agents notice when to ask

CS 329Z project (Stanford, autumn 2026). Author: Srijit Ghosh.

When the task hides details, LLM agents ask too rarely. On HiL-Bench SQL, Qwen3-235B solves 80% of the practice
tasks when every hidden detail is given up front, and 38% when it has to ask for them. We hand-labeled all 75 details
the agent missed: when it asks, its questions are good, but it usually doesn't notice that it should ask.

This repo keeps the agent unchanged and adds a **checkpoint at submit**. When the agent submits, the harness runs
deterministic checks on the answer against its own log of what the model actually saw. It holds the submission while
any new flag is open, and the agent has to ask the human about it. Each version adds one check aimed at one failure
mode:

| Version | Check | Targets |
|---|---|---|
| v1 | none (must match Scale's harness) | — |
| v2 | **value with no source**: a literal in the query that no given or schema text contains | filled-in values |
| v3 | **similar alternatives**: the query uses one of several look-alike tables or columns | narrow search, picking among options |

**Results so far.** Qwen3-235B-A22B-2507 is the agent and Llama-3.3-70B the judge, with 3 tries per task. The checks were
tuned on the 20 dev tasks; the 39 held-out tasks were run only once the versions were frozen.

| Version | pass@1, dev | pass@1, held-out | vs v1, held-out | schema details found, held-out | questions per run, held-out |
|---|---|---|---|---|---|
| v1: no checks | 35.0% | 13.7% | — | 39/108 | 3.4 |
| v2: value with no source | 38.3% | 15.4% | +1.7 points (p = 0.42) | 54/108 | 4.2 |
| v3: v2 + similar alternatives | 53.3% | 21.4% | +7.7 points (p = 0.05) | 75/108 | 7.0 |
| full info (ceiling) | 80.0% | 59.0% (1 try) | | | |

- Each row is that version's latest batch.
- Batches without checks, rerun, score 35–38% on dev, so smaller dev differences are noise.
- p comes from a within-task permutation test.

All tables and statistics are in `analysis/midway/summary.md`, and the figures in `analysis/midway/figures/`.
Both are regenerated from the run folders by the commands under "Reproduce the results".

---

## Layout

```
hilx/                  our harness (imports mini-swe-agent and Scale's SQL pieces; never copies or edits them)
  agent.py             the loop (Scale's step rules) + the checkpoint at submit
  core/                domain-neutral: evidence log, entities, value matching
  checks/              one file per check: unsourced_values.py (v2), similar_alternatives.py (v3)
  adapters/            domain plug-ins; sql.py is the only one so far
  run.py, model.py     one run: tools, model calls, scoring (Scale's scorer), cost
  scale.py             the Scale (HiL-Bench) imports and runner-bug workarounds
configs/
  versions/*.yaml      one file per harness version; never edited after it has run
  sql/*.yaml           Scale's agent configs (only model/hosting changed)
  routing/*.yaml       OpenRouter provider settings for the agent model
  split.json           20 dev tasks / 80 test tasks (seeded); test_small_tasks.json = the 39 small-DB test tasks
  judge_config.yaml    the simulated human (Llama-3.3-70B on OpenRouter)
scripts/
  materialize.py       download and prepare tasks (Scale's prepare_sql_task)
  run_version.py       run one harness version on a task set, N tries, and score it
  hil_sql.py           run Scale's own harness (v0, full-info ceiling) with two runner bugs worked around
  spend_guard.py       run any command under a dollar budget; scrub secrets from outputs
  replay_checks.py, replay_alternatives.py   replay a check offline on finished runs (no model calls)
  serve.py, check_servers.py, make_split.py  servers, a server smoke test, the dev/test split
analysis/
  failure_labels_step4f.tsv   hand labels: each of the 75 missed details and what the agent did instead
  midway_analysis.py          all tables and statistics, offline, from run folders
  midway_figures.py           the figures, from analysis/midway/summary.json
  midway/                     their outputs: summary.md, summary.json, figures/
```

`vendor/`, `data/` and `runs/` are gitignored. You recreate `vendor/` and `data/` with the commands below, and
`runs/` is what the experiments write.

## Setup

Requires Python 3.11, [uv](https://docs.astral.sh/uv/), git, and an OpenRouter API key. Tested on macOS (Apple
silicon).

```bash
# 1. Pinned third-party code. Neither repo is edited. Data: HF ScaleAI/hil-bench at revision 045419da0f51f10efc621efdb481a6502df810d5.
git clone https://github.com/hilbenchauthors/hil-bench vendor/hil-bench
git -C vendor/hil-bench checkout c5a656e41b5c50917686f9df8e4c72152bf27ba9
git clone https://github.com/SWE-agent/mini-swe-agent vendor/mini-swe-agent
git -C vendor/mini-swe-agent checkout 04d809ceab9df28f9adaed044884180159172930

# 2. One environment for everything: HiL-Bench's, plus mini-swe-agent and the SQL parser.
#    `uv sync` removes packages it didn't install, so rerun the second line after any sync.
(cd vendor/hil-bench && uv sync --python 3.11)
uv pip install --python vendor/hil-bench/.venv/bin/python ./vendor/mini-swe-agent sqlglot==26.6.0

# 3. Keys
cp .env.example .env        # fill in OPENROUTER_API_KEY and LITELLM_USER

# 4. Tasks: downloads databases, schema files and blocker registries from HF (ScaleAI/hil-bench at a pinned
#    revision) and builds each task's business-info index. About 15 s per task.
PY=vendor/hil-bench/.venv/bin/python
$PY scripts/materialize.py --split dev                      # 20 dev tasks -> data/sql_tasks/instances.json
```

For the held-out tasks, prepare them and write their own instances file, so the dev file stays as it is:

```bash
cp data/sql_tasks/instances.json data/sql_tasks/instances_dev.json
$PY scripts/materialize.py --tasks $(python3 -c "import json;print(' '.join(json.load(open('configs/test_small_tasks.json'))))")
$PY - <<'EOF'
import json, shutil
want = set(json.load(open("configs/test_small_tasks.json")))
allx = json.load(open("data/sql_tasks/instances.json"))
json.dump([i for i in allx if i["problem_statement"]["id"] in want], open("data/sql_tasks/instances_test_small.json", "w"), indent=2)
shutil.copy("data/sql_tasks/instances_dev.json", "data/sql_tasks/instances.json")
EOF
```

To check that both servers work (about $0.002): run `$PY scripts/serve.py` in one shell and
`$PY scripts/check_servers.py` in another.

## Reproduce the results

Each command below runs one batch. Always wrap a live run in `spend_guard.py`. A finished run is skipped, so
rerunning the same command resumes a batch. Costs are what we paid on OpenRouter; each dev batch of 60 runs takes
about 30–35 minutes with 6 workers.

```bash
PY=vendor/hil-bench/.venv/bin/python
guard() { $PY scripts/spend_guard.py --budget "$1" --watch-dir "$2" -- "${@:3}"; }

# v0: Scale's harness, native asking + full-info ceiling (20 dev tasks x 3; about $1.90)
guard 5 runs/step4f_qwen235b_fixed $PY scripts/hil_sql.py data/sql_tasks/instances.json \
  -m openrouter/qwen/qwen3-235b-a22b-2507 -k 3 --all-modes -w 6 --run-name both_20x3 \
  --output-dir runs/step4f_qwen235b_fixed --config-mapping configs/config_mappings.yaml --judge-config configs/judge_config.yaml

# v1, v2, v3 and the v1b/v2b controls on the 20 dev tasks x 3 (about $1–2.50 each)
for v in v1_native v2_unsourced v3_alternatives v1b_open_questions v2b_unsourced_open; do
  guard 5 runs/$v/dev_20x3 $PY scripts/run_version.py configs/versions/$v.yaml --out runs/$v/dev_20x3
done
# The "pinned" v1/v2 batches add: --routing configs/routing/parasail_fp8.yaml --out runs/<v>/dev_20x3_parasail

# Held-out: v1, v2 and v3 on the 39 small-DB test tasks x 3 (about $2, $2 and $3.50), full info x 1 (about $1)
for v in v1_native v2_unsourced v3_alternatives; do
  guard 12 runs/$v/test_small_39x3 $PY scripts/run_version.py configs/versions/$v.yaml \
    --out runs/$v/test_small_39x3 --tasks data/sql_tasks/instances_test_small.json
done
guard 8 runs/step4g_fullinfo_test $PY scripts/hil_sql.py data/sql_tasks/instances_test_small.json \
  -m openrouter/qwen/qwen3-235b-a22b-2507 -k 1 --full-info -w 6 --run-name fullinfo_39x1 \
  --output-dir runs/step4g_fullinfo_test --config-mapping configs/config_mappings.yaml --judge-config configs/judge_config.yaml

# Tables, statistics and figures (offline, seconds). The figures need only matplotlib.
$PY analysis/midway_analysis.py --test          # -> analysis/midway/summary.md, summary.json
uv run --no-project --python 3.11 --with matplotlib python analysis/midway_figures.py   # -> analysis/midway/figures/

# Replay a check offline on the v0 runs before running it live (no model calls)
$PY scripts/replay_checks.py runs/step4f_qwen235b_fixed/both_20x3 --out runs/replay_unsourced
$PY scripts/replay_alternatives.py runs/step4f_qwen235b_fixed/both_20x3 --out runs/replay_alternatives
```

**What will differ on a rerun.** The agent runs at temperature 1.0 and OpenRouter routes calls to several fp8
providers, so pass@1 moves by 3–5 points between identical batches. Our four no-check batches span 35–38%.
Compare versions with the paired tests in `summary.md` §2, not with single numbers.

- The v1, v2, v1b and v2b `dev_20x3` batches ran before provider routing was set. A rerun uses the current default,
  `configs/routing/fp8_any.yaml`.
- Each batch records the commit it ran at (`run_info.json`; all ran on a clean tree). Check out that commit to rerun
  a batch exactly. Later commits only add versions and options, and the replay scripts confirm a refactor leaves
  flags unchanged.

## What each run records

- `run_info.json`: the version name, git commit and dirty files, routing, and a fingerprint of the version file
  plus the agent config.
- `pass_<n>/<task>/traj.json`: the full message list, the evidence log (what the model saw, tagged given /
  structure / data) and every checkpoint with its flags.
- `pass_<n>/<task>/result.json`: the submission, scores (ours and Scale's), tokens, cost at list price and flags.
- `ask_human_logs.json`: the judge's log, i.e. which question found which hidden detail.
- `metrics.json`: pass@1, pass@3, questions, precision/recall, flags and cost for the batch.

## Known issues kept for parity with Scale's harness

- Scale's judge sometimes can't parse its own JSON when a question quotes a term. It answers "can't answer", and
  the run continues.
- Scale's runner has two bugs. One shares a config dict across modes; the other uses one business-info index for
  every task. `scripts/materialize.py` and `scripts/hil_sql.py` work around both without editing vendor code.
- Scale's `submit_sql` tool doc makes models append `ENDOFSQL` to the query. Our harness strips it before scoring
  and also records Scale's unstripped score (`resolved_scale_scoring`).
