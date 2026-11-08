"""The evidence log: every piece of text the model saw during a run, tagged by source type.

Source types, from strongest to weakest:
  given      what someone told the agent (the task, notes, the human's answers)
  structure  descriptions of the environment (a schema, a file tree, API docs)
  data       raw contents the agent looked at (query results, file contents, logs)
Which tool's output is which type is up to the domain adapter (hilx/adapters). The agent's own text
is never evidence, and neither are tool outputs that echo it back (a submit tool).
"""

from dataclasses import dataclass

SOURCE_RANK = {"given": 0, "structure": 1, "data": 2}

# Replies that carry no information, so they are not evidence (Scale's judge replies).
EMPTY_REPLIES = ("irrelevant question", "can't answer")


@dataclass
class Evidence:
    step: int  # 0 = the task message
    tool: str  # "task" for the task message
    source: str  # given | structure | data
    text: str  # exactly as the model saw it


def is_informative(observation: str) -> bool:
    return not observation.strip().lower().startswith(EMPTY_REPLIES)


def evidence_from_sweagent_traj(traj: dict, tool_sources: dict[str, str]) -> list[Evidence]:
    """Rebuild the evidence log from a SWE-agent .traj (Scale's harness), for offline replay."""
    task = next(h for h in traj["history"] if h["role"] == "user")
    log = [Evidence(0, "task", "given", task["content"] if isinstance(task["content"], str) else str(task["content"]))]
    for i, step in enumerate(traj["trajectory"], 1):
        tool = (step["action"] or "").split(maxsplit=1)[0] if step["action"] else ""
        source = tool_sources.get(tool)
        text = str(step["observation"] or "")
        if source and is_informative(text):
            log.append(Evidence(i, tool, source, text))
    return log
