"""Named things in an environment (tables and columns, files and functions) and how alike two are.

Used by the "similar alternatives" check: an answer that uses one of several look-alike entities
made a choice, whether or not the agent saw the others. Similarity combines three domain-neutral
signals, because no single one catches every confusing case in the practice traces:
  - shared name words  (fastestLapSpeed ~ avg_speed: "speed"; F1_ID ~ DB_ID: "id")
  - names equal up to digits  (Date1 ~ Date2, A10 ~ A11)
  - description similarity (sentence embeddings; NumTstTakr ~ TestTakerCount, 0.77)
"""

import re
import threading
from dataclasses import dataclass

EMBED_MODEL = "all-MiniLM-L6-v2"  # the model Scale's business-info server uses
_model, _lock, _cache = None, threading.Lock(), {}


@dataclass(frozen=True)
class Entity:
    container: str  # table (or file)
    name: str | None  # column (or function); None = the container itself
    description: str = ""

    @property
    def label(self) -> str:
        return f"{self.container}.{self.name}" if self.name else self.container


def name_words(name: str) -> set[str]:
    """'fastestLapSpeed' -> {fastest, lap, speed}; 'test_taker_audited' -> {test, taker, audited}."""
    parts = re.findall(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+", name)
    return {p.lower() for p in parts if len(p) > 1 or p.isalpha()}


def digit_skeleton(name: str) -> str | None:
    """'Date3' -> 'date#'; names with no digits -> None."""
    return re.sub(r"\d+", "#", name.lower()) if re.search(r"\d", name) else None


def embed(texts: list[str]):
    global _model
    with _lock:
        if _model is None:
            from sentence_transformers import SentenceTransformer
            _model = SentenceTransformer(EMBED_MODEL)
        missing = [t for t in texts if t not in _cache]
        if missing:
            for t, v in zip(missing, _model.encode(missing, normalize_embeddings=True)):
                _cache[t] = v
        return [_cache[t] for t in texts]


def similarity(a: Entity, b: Entity) -> dict:
    """The three signals for a pair of entities of the same kind (two columns, or two tables)."""
    na, nb = a.name or a.container, b.name or b.container
    va, vb = embed([f"{na}: {a.description}", f"{nb}: {b.description}"])
    return {
        "shared_words": sorted(name_words(na) & name_words(nb)),
        "same_up_to_digits": digit_skeleton(na) is not None and digit_skeleton(na) == digit_skeleton(nb),
        "description_sim": float(va @ vb),
    }
