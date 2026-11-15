"""BM25 index, the lexical half of V2.

The implementation is the one already proven in `mcp_servers/portfolio/search.py`, lifted
here so the variants own their indexing and the tool layer keeps working unchanged.

Why BM25 matters on this corpus specifically: it rewards rare terms. "Celery", "dbt" and
"pgvector" each appear in two or three items out of 592, so their idf is high and a match
dominates the score. A dense embedding maps them into a neighbourhood of related concepts,
which is useful for paraphrase and actively harmful when the client asked for that exact
library. V2 exists to have both.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field

from mcp_servers.portfolio.corpus import EvidenceItem

_WORD = re.compile(r"[a-z0-9+#.]+")

_STOP = frozenset(
    """
    a an the and or but if then than that this these those of in on at to for with by from
    as is are was were be been being do does did doing have has had having i you he she it
    we they my your our their me him her them so not no yes can will would should could may
    might must about into over under again further once here there when where why how all
    any both each few more most other some such only own same too very just also like need
    needs needed looking experience work working project projects client clients please want
    wanted using use used will able team help new good great best
    """.split()
)

K1 = 1.5
B = 0.75


def tokenize(text: str) -> list[str]:
    return [t for t in _WORD.findall(text.lower()) if t not in _STOP and len(t) > 1]


@dataclass
class BM25Index:
    items: list[EvidenceItem]
    docs: list[list[str]] = field(default_factory=list)
    freqs: list[Counter] = field(default_factory=list)
    lengths: list[int] = field(default_factory=list)
    idf: dict[str, float] = field(default_factory=dict)
    avg_len: float = 1.0

    @classmethod
    def build(cls, items: Sequence[EvidenceItem]) -> BM25Index:
        rows = list(items)
        docs = [tokenize(f"{i.project or ''} {' '.join(i.skills)} {i.text}") for i in rows]
        lengths = [len(d) or 1 for d in docs]
        freqs = [Counter(d) for d in docs]

        df: Counter[str] = Counter()
        for doc in docs:
            df.update(set(doc))
        n = len(docs) or 1
        idf = {
            term: max(1e-6, math.log(1 + (n - count + 0.5) / (count + 0.5)))
            for term, count in df.items()
        }
        return cls(
            items=rows,
            docs=docs,
            freqs=freqs,
            lengths=lengths,
            idf=idf,
            avg_len=sum(lengths) / max(1, len(lengths)),
        )

    def scores(self, query: str) -> list[float]:
        terms = tokenize(query)
        out = [0.0] * len(self.items)
        if not terms:
            return out
        for index in range(len(self.items)):
            freq = self.freqs[index]
            length = self.lengths[index]
            total = 0.0
            for term in terms:
                tf = freq.get(term, 0)
                if not tf:
                    continue
                total += (
                    self.idf.get(term, 0.0)
                    * (tf * (K1 + 1))
                    / (tf + K1 * (1 - B + B * length / self.avg_len))
                )
            out[index] = total
        return out

    def rank(self, query: str, k: int = 5) -> list[tuple[EvidenceItem, float]]:
        scored = list(zip(self.items, self.scores(query), strict=True))
        scored.sort(key=lambda p: (-p[1], p[0].id))
        return [(item, score) for item, score in scored[:k] if score > 0]


_cache: dict[str, BM25Index] = {}


def get_bm25_index(items: Sequence[EvidenceItem]) -> BM25Index:
    rows = list(items)
    key = f"{len(rows)}:" + "|".join(i.id for i in rows[:200])
    index = _cache.get(key)
    if index is None:
        index = BM25Index.build(rows)
        _cache[key] = index
    return index


def clear_cache() -> None:
    _cache.clear()
