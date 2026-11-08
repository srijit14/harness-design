"""Check: values in the answer that no source gave the agent ("filled in" failures).

1. The domain adapter extracts every value the agent committed to in its answer (for SQL: numbers,
   strings, dates; items of a list count one by one). Trivial values are skipped.
2. Look each one up, exactly, in the evidence log (what the model saw). No semantic search: the
   question is "was this exact value given?", and similarity would match "65" to "senior".
3. Flag a value by the best source found. With count_data=True, only values found nowhere are
   flagged. With count_data=False, values found only in raw data are flagged too, because seeing a
   value in the data says nothing about what it means.

Nothing here is domain-specific: the extractor and the wording come from the adapter.
"""

from hilx.adapters.base import DomainAdapter
from hilx.core.evidence import SOURCE_RANK, Evidence
from hilx.core.values import EvidenceIndex, Flag

NAME = "unsourced_values"


def check(answer: str, evidence: list[Evidence], adapter: DomainAdapter, context: dict | None = None,
          count_data: bool = True) -> tuple[list[Flag], bool]:
    """Flags for values in `answer` with no acceptable source. Returns (flags, parsed_ok).
    The unsourced items of one list make one flag, since one list is one question."""
    values, parsed = adapter.extract_values(answer)
    index = EvidenceIndex(evidence)
    unsourced, seen = {}, set()
    for v in values:
        if v.text.lower() in seen:
            continue
        seen.add(v.text.lower())
        best = min((h.source for h in index.sources(v)), key=SOURCE_RANK.get, default=None)
        if best is None or (best == "data" and not count_data):
            key = ("list", v.group) if v.group is not None else ("value", v.text.lower())
            unsourced.setdefault(key, []).append((v, best))
    w = adapter.source_words
    flags = []
    for items in unsourced.values():
        vs = [v for v, _ in items]
        best = min((b for _, b in items), key=lambda b: SOURCE_RANK.get(b, 9), default=None)
        shown = ", ".join(repr(v.text) for v in vs)
        where = f"(in `{vs[0].context}`)"
        if all(b is None for _, b in items):
            msg = f"{shown} {where}: not in {w['given']}, {w['structure']} or {w['data_any']}."
        else:
            msg = f"{shown} {where}: found only in {w['data']}, not in {w['given']} or {w['structure']}."
        flags.append(Flag(NAME, vs, best, msg))
    return flags, parsed
