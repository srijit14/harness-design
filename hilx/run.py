"""Run one task once under one version, and score it with Scale's scorer."""

import json
import re
import shutil
from pathlib import Path

from hilx.adapters import ADAPTERS
from hilx.agent import HilAgent, SQLRunEnv
from hilx.model import HilModel
from hilx.scale import ROOT, SQLProblemStatement, SQLToolHandler, evaluate_single_sql_instance, load_agent_config

TASKS_DIR = ROOT / "data" / "sql_tasks"
PRICE_LIST = json.loads((ROOT / "configs" / "model_prices.json").read_text())
STRAY_END = re.compile(r"\s*;?\s*\bENDOFSQL\b\s*$")
# What Scale's tools return when their server fails or times out (bin/get_business_info, bin/ask_human).
SERVER_ERRORS = ("Error querying knowledge base", "can't answer (perhaps transient hiccup)")


def strip_stray_end(sql: str) -> str:
    """Our harness's one scoring change (Srijit, 2026-11-06): drop a trailing ENDOFSQL that the model
    typed inside its query. Scale's harness would score that query as an error."""
    return STRAY_END.sub("", sql).strip()


def run_id(task_id: str, version: str, pass_n: int) -> str:
    # The judge and the business-info tool take the task from the part before "__".
    return f"{task_id}__{version}__pass_{pass_n}"


def token_counts(messages: list[dict]) -> dict:
    """Tokens and the provider for each model call, from OpenRouter's usage report. Tokens, not dollars,
    are what compare across runs: the bill also depends on the provider and on prompt-cache hits."""
    t = {"tokens_in": 0, "tokens_cached": 0, "tokens_out": 0, "providers": {}}
    for m in messages:
        r = m.get("extra", {}).get("response") if m.get("role") == "assistant" else None
        if not r:
            continue
        u = r.get("usage") or {}
        t["tokens_in"] += u.get("prompt_tokens") or 0
        t["tokens_cached"] += (u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
        t["tokens_out"] += u.get("completion_tokens") or 0
        p = r.get("provider") or "unknown"
        t["providers"][p] = t["providers"].get(p, 0) + 1
    return t


def list_cost(t: dict, model_name: str) -> float:
    """The run's tokens priced at one fixed list (configs/model_prices.json), so cost compares across runs
    whichever provider served the calls and whatever it billed."""
    p = PRICE_LIST[model_name]
    uncached = t["tokens_in"] - t["tokens_cached"]
    return uncached * p["input_cost_per_token"] + t["tokens_cached"] * p["cache_read_input_token_cost"] + t["tokens_out"] * p["output_cost_per_token"]


def run_one(instance: dict, version: dict, pass_n: int, out_dir: Path, routing: dict | None = None) -> dict:
    ps = SQLProblemStatement(**instance["problem_statement"])
    rid = run_id(ps.id, version["name"], pass_n)
    run_dir = out_dir / f"pass_{pass_n}" / ps.id
    run_dir.mkdir(parents=True, exist_ok=True)

    scale_cfg = load_agent_config(ROOT / version["agent_config"])
    handler = SQLToolHandler(scale_cfg.tools)
    handler._task_instance_id = rid  # Scale sets this in DefaultSQLAgent.setup; the tools read it
    working_db = run_dir / "working.sqlite"
    shutil.copy(instance["env"]["base_db_path"], working_db)

    model = HilModel(scale_cfg.model, scale_cfg.tools.tools, routing)
    adapter = ADAPTERS[version.get("adapter", "sql")]
    needs_catalog = any(c["name"] == "similar_alternatives" for c in version.get("checks", []))
    context = {"catalog": adapter.catalog(TASKS_DIR / ps.id)} if needs_catalog else {}
    agent = HilAgent(
        model, SQLRunEnv(handler, str(working_db)), scale_cfg, adapter, context=context,
        checks=version.get("checks", []), on_flag=version.get("on_flag", "ask"),
        max_checkpoint_rounds=version.get("max_checkpoint_rounds", 2),
        system_prompt_addition=version.get("system_prompt_addition", ""), output_path=run_dir / "traj.json",
    )
    try:
        exit_info = agent.run(database_name=ps.database_name, problem_statement=ps.get_problem_statement())
    finally:
        working_db.unlink(missing_ok=True)

    submission = exit_info.get("submission") or ""
    stripped = strip_stray_end(submission)
    scores = {}
    for name, sql in (("scale", submission), ("ours", stripped)):
        _, res, ok = evaluate_single_sql_instance(rid, {"model_patch": sql}, instance, TASKS_DIR)
        scores[name] = {"resolved": ok, "status": res.get("status"), "error": res.get("error")}
    result = {
        "run_id": rid, "task": ps.id, "version": version["name"], "pass": pass_n,
        "exit_status": exit_info.get("exit_status"), "error": exit_info.get("error"),
        "submission": submission, "endofsql_stripped": stripped != submission,
        "resolved": scores["ours"]["resolved"], "resolved_scale_scoring": scores["scale"]["resolved"], "scores": scores,
        "model_calls": agent.n_calls, "cost": agent.cost, "format_errors": len(agent.format_errors),
        "checkpoints": agent.checkpoints,
        **(tokens := token_counts(agent.messages)), "cost_list": list_cost(tokens, scale_cfg.model.name),
        "server_errors": sum(any(e in str(m.get("content", "")) for e in SERVER_ERRORS) for m in agent.messages if m.get("role") != "assistant"),
    }
    (run_dir / "result.json").write_text(json.dumps(result, indent=2))
    return result
