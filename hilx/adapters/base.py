"""What a domain must supply to run the harness. Everything else (evidence log, checkpoint, checks,
what to do with a flag) is shared.

To take the harness to a new kind of task, write one adapter:
  tool_sources   which tool outputs are evidence, and of which source type (given/structure/data)
  submission     how to read the final answer out of the agent's submit action
  extract_values the values the agent committed to in its answer
  catalog        every named entity in the task's environment, with descriptions (tables/columns)
  entities_used  the entities the answer uses, each as "content" or "join" (plumbing)
  source_words   how flag messages name each source type, in the domain's terms
  plumbing_words name words too generic to make two entities look alike (id, name, date in SQL)
"""

from typing import Protocol

from pathlib import Path

from hilx.core.entities import Entity
from hilx.core.values import Value


class DomainAdapter(Protocol):
    name: str
    tool_sources: dict[str, str]
    source_words: dict[str, str]  # keys: given, structure, data, data_any
    plumbing_words: set[str]

    def submission(self, action: str) -> str:
        """The answer inside a submit action ('' if none)."""
        ...

    def extract_values(self, answer: str) -> tuple[list[Value], bool]:
        """Values committed to in the answer. Returns (values, parsed_ok)."""
        ...

    def catalog(self, task_dir: Path) -> list[Entity]:
        """Every table/column (file/function, ...) in the task's environment, with descriptions."""
        ...

    def entities_used(self, answer: str, catalog: list[Entity]) -> tuple[list[tuple[Entity, str]], bool]:
        """Entities the answer uses, each with role "content" or "join". Returns (uses, parsed_ok)."""
        ...
