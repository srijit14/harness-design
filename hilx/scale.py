"""Scale's SQL pieces, imported unchanged from vendor/hil-bench so v1 matches Scale's harness.

From Scale we take: the agent config (prompts, tool list, error text), the tool definitions sent to
the model, the parser that turns a tool call into an action, the tool runner, the task class and
the scorer. Our own code is only the loop (hilx/agent.py) and the model wrapper (hilx/model.py).
"""

import os
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "vendor" / "hil-bench"
SWE_AGENT = VENDOR / "SWE-agent"
for p in (VENDOR, SWE_AGENT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from hil_bench.utils.calculate_sql_pass_at_1 import evaluate_single_sql_instance  # noqa: E402
from sweagent.agent.agents import DefaultSQLAgentConfig  # noqa: E402
from sweagent.agent.problem_statement import SQLProblemStatement  # noqa: E402
from sweagent.tools.sql_tools import SQLToolHandler  # noqa: E402

__all__ = ["load_agent_config", "link_chroma_by_task", "SQLToolHandler", "SQLProblemStatement", "evaluate_single_sql_instance"]

CHROMA_BY_TASK = ROOT / "data" / "chroma_by_task"


def link_chroma_by_task():
    """Business info per task. Scale's runner sets one CHROMA_PATH for the whole batch, from the first
    task, so other tasks search the wrong index (FIRST_EXPERIMENT.md, runner bug 2). With
    CHROMA_BASE_PATH set, the get_business_info tool resolves CHROMA_BASE_PATH/<task_id> per run.
    This points it at data/chroma_by_task/<task_id> -> data/sql_tasks/<task_id>/chroma_db links."""
    CHROMA_BY_TASK.mkdir(parents=True, exist_ok=True)
    for chroma in (ROOT / "data" / "sql_tasks").glob("*/chroma_db"):
        link = CHROMA_BY_TASK / chroma.parent.name
        if not link.exists():
            link.symlink_to(chroma.resolve())
    os.environ.pop("CHROMA_PATH", None)
    os.environ["CHROMA_BASE_PATH"] = str(CHROMA_BY_TASK)


def load_agent_config(path: Path) -> DefaultSQLAgentConfig:
    """Load a Scale-format SQL agent YAML (e.g. configs/sql/qwen3_235b_2507_ask.yaml) the way Scale's
    runner does, with tool paths made absolute so it works from any working directory."""
    agent = yaml.safe_load(Path(path).read_text())["agent"]
    tools = agent["tools"]
    tool_dir = (VENDOR / tools["tool_dir"]).resolve()
    tools["tool_dir"] = str(tool_dir)
    tools["bundles"] = [{**b, "path": str((tool_dir.parent / b["path"]).resolve())} for b in tools["bundles"]]
    return DefaultSQLAgentConfig.model_validate(agent)
