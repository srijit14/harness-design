"""Check: the answer uses one of several look-alike entities ("narrow search" and "picked" failures).

For each entity the answer uses for its content (not join keys), find look-alikes in the task's
catalog (hilx/core/entities.py for the signals). Flag the entity with its look-alikes, and say whether
the agent ever saw them (narrow search) or saw them and chose alone (picked). The human is asked
which one is meant; seeing the options doesn't settle that.

Thresholds were set on the 20 practice tasks' traces (scripts/replay_alternatives.py).
"""

import re

from hilx.adapters.base import DomainAdapter
from hilx.core.entities import Entity, name_words, similarity
from hilx.core.evidence import Evidence
from hilx.core.values import Flag

NAME = "similar_alternatives"


def informative(e: Entity, min_words: int) -> bool:
    """A description that says something beyond the name ('A4' for column A4 says nothing)."""
    words = set(re.findall(r"[a-z]+", e.description.lower())) - name_words(e.name or e.container)
    return len(words) >= min_words


def look_alikes(u: Entity, catalog: list[Entity], used: set[Entity], same_sim: float, cross_sim: float,
                cross_word_sim: float, table_sim: float, confusable_sim: float, include_used: bool,
                min_description_words: int, plumbing_words: set[str]) -> list[Entity]:
    """Names (or descriptions) nominate candidates; descriptions decide. A candidate is a look-alike only
    if the documentation doesn't tell the two apart: similar descriptions, or one has no real description."""
    out = []
    for c in catalog:
        if c == u or (c in used and not include_used) or (c.name is None) != (u.name is None):
            continue
        if u.name is not None and c.name.lower() == u.name.lower():  # the same column in another table: a join copy
            continue
        s = similarity(u, c)
        confusable = (s["description_sim"] >= confusable_sim or not informative(u, min_description_words)
                      or not informative(c, min_description_words))
        if u.name is None:
            wu, wc = name_words(u.container), name_words(c.container)
            nominated = wu <= wc or wc <= wu or s["description_sim"] >= table_sim
        elif c.container == u.container:
            nominated = bool(s["shared_words"]) or s["same_up_to_digits"] or s["description_sim"] >= same_sim
        else:
            content_words = set(s["shared_words"]) - plumbing_words
            nominated = s["description_sim"] >= cross_sim or bool(content_words and s["description_sim"] >= cross_word_sim)
        if nominated and confusable:
            out.append(c)
    return out


def _seen(e: Entity, evidence: list[Evidence]) -> bool:
    pat = re.compile(r"(?<![A-Za-z0-9_])" + re.escape(e.name or e.container) + r"(?![A-Za-z0-9_])")
    return any(pat.search(ev.text) for ev in evidence if ev.source in ("structure", "data"))


def check(answer: str, evidence: list[Evidence], adapter: DomainAdapter, context: dict | None = None,
          same_sim: float = 0.6, cross_sim: float = 0.8, cross_word_sim: float = 0.4, table_sim: float = 0.8,
          confusable_sim: float = 0.7, include_used: bool = False, min_description_words: int = 2,
          max_shown: int = 6) -> tuple[list[Flag], bool]:
    catalog = (context or {}).get("catalog")
    if catalog is None:
        return [], True
    uses, parsed = adapter.entities_used(answer, catalog)
    used = {e for e, _ in uses}
    flags, covered = [], set()
    for u, role in uses:
        if role != "content" or u in covered:
            continue
        # include_used (v3b): look-alikes the query also uses stay in the list, since using several of them
        # is a choice too ("picked"); v3 dropped them and so listed unrelated look-alikes instead.
        alts = look_alikes(u, catalog, used, same_sim, cross_sim, cross_word_sim, table_sim, confusable_sim,
                           include_used, min_description_words, adapter.plumbing_words)
        if not alts:
            continue
        if include_used:
            covered |= set(alts)  # one flag per group of look-alikes the query uses
        unseen = [a for a in alts if not _seen(a, evidence)]
        shown = "; ".join(f"`{a.label}`" + (" (also in your query)" if a in used else f" ({a.description[:80]})" if a.description else "")
                          for a in alts[:max_shown])
        more = f"; and {len(alts) - max_shown} more" if len(alts) > max_shown else ""
        kind = "column" if u.name else "table"
        if unseen:
            how = f"You never looked at {', '.join(f'`{a.label}`' for a in unseen[:max_shown])}."
        else:
            how = "You saw them and chose one yourself."
        msg = f"`{u.label}` is one of several similar {kind}s: {shown}{more}. {how} Which one is meant is not settled by the data."
        flags.append(Flag(NAME, [], "unseen" if unseen else "seen", msg, key=u.label.lower()))
    return flags, parsed
