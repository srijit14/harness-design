"""Values an agent committed to, and exact lookup of each value in the evidence log.

An adapter's extractor turns an answer (a SQL query, a code diff, an action's arguments) into Values.
Everything after that, the lookup and the flag, is the same for every domain.
"""

import re
from dataclasses import dataclass

from hilx.core.evidence import Evidence

TRIVIAL_NUMBERS = {0.0, 1.0, -1.0, 100.0}
FORMAT_STRING = re.compile(r"^%[A-Za-z%\-:/ ]*$")  # strftime formats like '%Y'
DATE = re.compile(r"^(\d{4})-\d{2}-\d{2}(?:[ T][\d:.]+)?$")
NUMBER_IN_TEXT = re.compile(r"(?<![\w.])\d+(?:\.\d+)?(?![\w]|\.\d)")
THOUSANDS = re.compile(r"(?<=\d),(?=\d{3}\b)")  # "50,000" -> "50000"
DECADE = re.compile(r"(?<=\d0)'?s\b")  # "1990s" -> "1990"
SHORT = 2  # strings this short ('A', 'UK') match only with the same capitals, so 'A' is not the word "a"


@dataclass
class Value:
    text: str  # as written in the answer (an adapter may strip wildcards and the like)
    kind: str  # number | string | date
    context: str  # the expression it sits in, for the flag message
    group: int | None = None  # values in the same list share a group, and are one flag


@dataclass
class Flag:
    check: str
    values: list[Value]  # one value, or the unsourced items of one list
    found_in: str | None  # best source type across the values, or None
    message: str
    key: str = ""  # what makes two flags the same, for checks without values (default: the values)


def number(text: str) -> float | None:
    try:
        return float(text)
    except ValueError:
        return None


def is_trivial(v: Value) -> bool:
    if v.kind == "number":
        return number(v.text) in TRIVIAL_NUMBERS
    return v.text.strip() == "" or bool(FORMAT_STRING.match(v.text))


def _forms(v: Value) -> tuple[list[float], list[str]]:
    """The number and text forms to look up. 0.78 also matches 78 (a percent); a date also matches its year."""
    numbers, texts = [], []
    if v.kind == "number" or (v.kind == "string" and v.text.isdigit()):
        n = abs(number(v.text))
        numbers.append(n)
        if 0 < n < 1:
            numbers.append(round(n * 100, 6))
    if v.kind == "date":
        texts.append(v.text)
        numbers.append(float(DATE.match(v.text).group(1)))
    if v.kind == "string" and not v.text.isdigit():
        texts.append(v.text)
    return numbers, texts


def _text_pattern(s: str, exact_case: bool) -> re.Pattern:
    """Whole-token match; a trailing plural is allowed ('OWNER' matches 'owners'). Case-insensitive,
    except exact_case=True for short values, which would otherwise match ordinary words."""
    if exact_case:
        return re.compile(r"(?<![A-Za-z0-9])" + re.escape(s) + r"(?![A-Za-z0-9])")
    return re.compile(r"(?<![a-z0-9])" + re.escape(s) + r"(?:s|es)?(?![a-z0-9])", re.IGNORECASE)


class EvidenceIndex:
    def __init__(self, evidence: list[Evidence]):
        self.entries = []
        for e in evidence:
            numeric = DECADE.sub("", THOUSANDS.sub("", e.text))
            self.entries.append((e, {round(float(m.group(0)), 6) for m in NUMBER_IN_TEXT.finditer(numeric)}))

    def sources(self, v: Value) -> list[Evidence]:
        numbers, texts = _forms(v)
        hits = []
        for e, nums in self.entries:
            if any(round(n, 6) in nums for n in numbers) or any(
                _text_pattern(t, len(t) <= SHORT).search(e.text) for t in texts
            ):
                hits.append(e)
        return hits
