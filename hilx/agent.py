"""The agent loop: mini-swe-agent's DefaultAgent with Scale's SQL step rules, plus the checkpoint.

Scale's rules kept (SWE-agent DefaultAgent.step / forward_with_handling and DefaultSQLAgent):
  - exactly one tool call per turn, parsed by Scale's FunctionCallingParser;
  - a bad tool call is retried up to max_requeries times with Scale's error text, and the failed
    tries are not kept in the history the model sees; after that the run exits (exit_format);
  - observations are wrapped in Scale's templates ("OBSERVATION:\n...");
  - submit_sql ends the run with "Submission received:\n<sql>";
  - at max_steps, a run that has not submitted ends with an empty submission (max_steps_reached).

Ours:
  - the domain adapter (hilx/adapters): which tools are evidence, how to read the submission, and the
    value extractor. The loop, evidence log and checkpoint below don't depend on the domain;
  - the evidence log (hilx/core/evidence.py): what the model saw, tagged by source;
  - the checkpoint: when the agent submits, the version's checks run on the query. If any check
    flags something new, the submission is held and the flags go back to the agent instead of
    "Submission received". After max_checkpoint_rounds, submissions are accepted as they are.
    With no checks (v1), the checkpoint does nothing.
"""

import logging
import time
from pathlib import Path

from jinja2 import Template
from minisweagent.agents.default import DefaultAgent

import hilx.scale  # noqa: F401  (puts Scale's code on the import path)
from sweagent.exceptions import FormatError

from hilx.adapters.base import DomainAdapter
from hilx.checks import CHECKS
from hilx.core.evidence import Evidence, is_informative
from hilx.model import ContextLimitExceeded

ON_FLAG_TEXT = {
    # The agent writes its own questions; this text only says what to do with the flags.
    "ask": "Before your query is accepted, ask the human analyst about each point above with ask_human, one question per point. Then submit again.",
    # v2b: same, plus the rule that Scale's judge applies but Scale's prompt doesn't state (no guesses in questions).
    "ask_open": "Before your query is accepted, ask the human analyst about each point above with ask_human, one question per point. "
                "Ask an open question about what was meant; do not include your own value or guess in the question. Then submit again.",
    "assume": "Before your query is accepted, decide on each point above and record your assumption. Then submit again.",
}


class SQLRunEnv:
    """mini-swe-agent environment that runs actions with Scale's SQLToolHandler on this run's DB copy."""

    def __init__(self, handler, working_db_path: str):
        self.handler = handler
        self.working_db_path = working_db_path

    def execute(self, action: str) -> str:
        return self.handler.execute_action(action, self.working_db_path)

    def get_template_vars(self, **kwargs) -> dict:
        return {}

    def serialize(self) -> dict:
        return {"info": {"env": {"working_db_path": self.working_db_path}}}


class HilAgent(DefaultAgent):
    def __init__(self, model, env: SQLRunEnv, scale_config, adapter: DomainAdapter, *, checks: list[dict] | None = None, on_flag: str = "ask",
                 max_checkpoint_rounds: int = 2, system_prompt_addition: str = "", context: dict | None = None,
                 output_path: Path | None = None):
        super().__init__(
            model, env,
            system_template=scale_config.templates.system_template,
            instance_template=scale_config.templates.instance_template,
            step_limit=scale_config.max_steps,
            cost_limit=scale_config.model.per_instance_cost_limit,
            max_consecutive_format_errors=scale_config.max_requeries,
            output_path=output_path,
        )
        self.scale = scale_config
        self.adapter = adapter
        self.context = context or {}  # per-task inputs for checks (the adapter's catalog)
        self.logger = logging.getLogger("hilx.agent")
        self.checks = checks or []  # [{"name": ..., "options": {...}}]
        self.on_flag = on_flag
        self.max_checkpoint_rounds = max_checkpoint_rounds
        self.system_prompt_addition = system_prompt_addition.strip()  # appended after Scale's system prompt
        self.evidence: list[Evidence] = []
        self.checkpoints: list[dict] = []  # one entry per checkpoint round
        self.format_errors: list[dict] = []
        self._raised: set[str] = set()

    # --- loop -------------------------------------------------------------------------------

    def run(self, task: str = "", **kwargs) -> dict:
        self.extra_template_vars |= kwargs
        self.messages = []
        self.add_messages(
            {"role": "system", "content": "\n\n".join(filter(None, [self._render_template(self.config.system_template), self.system_prompt_addition]))},
            {"role": "user", "content": self._render_template(self.config.instance_template)},
        )
        self.evidence = [Evidence(0, "task", "given", self.messages[1]["content"])]
        while self.messages[-1].get("role") != "exit":
            try:
                self.step()
            except Exception as e:  # API failures etc. end the run, not the batch
                self.logger.exception("run failed")
                self._exit("exit_error", "", error=f"{type(e).__name__}: {e}")
            finally:
                self.save(self.config.output_path)
        return self.messages[-1]["extra"]

    def _exit(self, status: str, submission: str, **extra):
        self.add_messages({"role": "exit", "content": status, "extra": {"exit_status": status, "submission": submission, **extra}})

    def step(self):
        if 0 < self.config.cost_limit <= self.cost:
            return self._exit("exit_cost", "")
        history = self.messages
        for n_fail in range(self.scale.max_requeries + 1):
            if n_fail == self.scale.max_requeries:
                return self._exit("exit_format", "")
            self.n_calls += 1
            try:
                message = self.model.query(history)
            except ContextLimitExceeded:
                return self._exit("exit_context", "")
            self.cost += message["extra"]["cost"]
            try:
                _, action = self.scale.tools.parse_function(
                    {"message": message["content"], "tool_calls": message["tool_calls"]}, self.scale.tools.commands
                )
                break
            except FormatError as e:
                info = getattr(e, "extra_info", {})
                error = Template(self.scale.tools.format_error_template).render(**info, exception_message=getattr(e, "message", str(e)))
                self.format_errors.append({"call": self.n_calls, "error_code": info.get("error_code"), "content": message["content"]})
                history = self.messages + [{"role": "assistant", "content": message["content"]}, {"role": "user", "content": error}]
        self.add_messages(message)
        observation, submission = self._execute(action)
        self.add_messages(self._observation_message(observation, message["tool_calls"][0]["id"]))
        if submission is not None:
            return self._exit("submitted", submission)
        if 0 < self.config.step_limit <= self.n_calls:
            return self._exit("max_steps_reached", "")

    def _execute(self, action: str) -> tuple[str, str | None]:
        """Returns (observation, submission or None)."""
        tool = action.split(maxsplit=1)[0] if action.strip() else ""
        if tool == self.scale.tools.submit_command:
            answer = self.adapter.submission(action)
            if not answer:
                return "Submission command used with no query.", ""
            held = self._checkpoint(answer)
            if held:
                return held, None
            return f"Submission received:\n{answer}", answer
        t0 = time.perf_counter()
        observation = self.env.execute(action)
        source = self.adapter.tool_sources.get(tool)
        if source and is_informative(observation):
            self.evidence.append(Evidence(self.n_calls, tool, source, observation))
        self.messages[-1]["extra"]["execution_time"] = time.perf_counter() - t0
        return observation, None

    def _observation_message(self, observation: str, tool_call_id: str) -> dict:
        t = self.scale.templates
        if observation.strip() == "":
            template, elided = t.next_step_no_output_template, 0
        elif len(observation) > t.max_observation_length:
            template, elided = t.next_step_truncated_observation_template, len(observation) - t.max_observation_length
        else:
            template, elided = t.next_step_template, 0
        content = Template(template).render(observation=observation, elided_chars=elided, max_observation_length=t.max_observation_length, **self.extra_template_vars)
        return {"role": "tool", "content": content, "tool_call_id": tool_call_id, "extra": {"raw_output": observation}}

    # --- checkpoint ---------------------------------------------------------------------------

    def _checkpoint(self, answer: str) -> str | None:
        """Run the checks on a submitted answer. Returns the message that holds the submission, or None."""
        if not self.checks or len(self.checkpoints) >= self.max_checkpoint_rounds:
            return None
        new = []
        for c in self.checks:
            flags, parsed = CHECKS[c["name"]](answer, self.evidence, self.adapter, self.context, **c.get("options", {}))
            for f in flags:
                key = f"{f.check}:" + (f.key or "|".join(sorted(v.text.lower() for v in f.values)))
                if key not in self._raised:
                    self._raised.add(key)
                    new.append(f)
        self.checkpoints.append({
            "round": len(self.checkpoints) + 1, "call": self.n_calls, "sql": answer, "held": bool(new),
            "flags": [{"check": f.check, "values": [v.text for v in f.values], "key": f.key, "found_in": f.found_in, "message": f.message} for f in new],
        })
        if not new:
            return None
        points = "\n".join(f"{i}. {f.message}" for i, f in enumerate(new, 1))
        return f"Your query was not submitted. A check found these points:\n{points}\n\n{ON_FLAG_TEXT[self.on_flag]}"

    # --- saving -------------------------------------------------------------------------------

    def serialize(self, *extra_dicts) -> dict:
        return super().serialize(
            {"info": {"checkpoints": self.checkpoints, "format_errors": self.format_errors}},
            {"evidence": [e.__dict__ for e in self.evidence]},
            *extra_dicts,
        )
