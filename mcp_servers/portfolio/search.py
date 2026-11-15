"""Ranking behind `search_evidence`.

The Portfolio server is the tool layer, not the RAG system. Part B's job is to choose a
retrieval design by measurement; this module's job is to be a defensible default until it
does, and then to get out of the way.

So the contract is a one-function protocol:

    rank(requirement, items, k, **opts) -> list[EvidenceItem]  # score set, best first

`get_ranker()` returns whatever `rag.retriever.get_ranker` provides when that module
exists, and the built-in otherwise. Part B implements V0 to V9 there and the tool
signature never changes, which is what the schema snapshot test in tests/contract pins.

The built-in is BM25 over the item text plus an exact-match bonus on the indexed skill
list. BM25 rather than cosine because this corpus turns on exact library names: a job
asking for Celery should surface the project that used Celery, and a dense embedding
blurs "Celery" into "task queue". That is V2's whole argument, and it is worth having
before V2 exists so the baseline is honest about where the difficulty is.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence
from typing import Any

from mcp_servers.portfolio.corpus import OUTCOME_RANK, EvidenceItem

_WORD = re.compile(r"[a-z0-9+#.]+")

# Words that match everything in a freelance corpus and so discriminate nothing.
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

# BM25 constants. k1 controls how fast term frequency saturates, b how much document
# length is penalised. These are the standard defaults and we have no measurement yet
# that would justify moving them; Part B's ablation is where that argument belongs.
_K1 = 1.5
_B = 0.75


def tokenize(text: str) -> list[str]:
    return [t for t in _WORD.findall(text.lower()) if t not in _STOP and len(t) > 1]


class BM25Ranker:
    """Okapi BM25 with a skill-overlap bonus.

    Built per call rather than cached: the corpus is 592 items and tokenising it takes a
    few milliseconds, which is not worth the cache-invalidation bug that incremental
    re-indexing (B1) would otherwise introduce.
    """

    def __init__(self, items: Sequence[EvidenceItem]) -> None:
        self.items = list(items)
        self.docs = [tokenize(f"{i.project or ''} {' '.join(i.skills)} {i.text}") for i in items]
        self.lengths = [len(d) or 1 for d in self.docs]
        self.avg_len = sum(self.lengths) / max(1, len(self.lengths))
        self.freqs = [Counter(d) for d in self.docs]

        df: Counter[str] = Counter()
        for doc in self.docs:
            df.update(set(doc))
        n = len(self.docs)
        # Standard BM25 idf, floored at a small positive number so a term appearing in
        # every document contributes nothing rather than going negative and actively
        # penalising a match.
        self.idf = {
            term: max(1e-6, math.log(1 + (n - count + 0.5) / (count + 0.5)))
            for term, count in df.items()
        }

    def _bm25(self, query: list[str], doc_index: int) -> float:
        freq = self.freqs[doc_index]
        length = self.lengths[doc_index]
        score = 0.0
        for term in query:
            tf = freq.get(term, 0)
            if not tf:
                continue
            idf = self.idf.get(term, 0.0)
            score += idf * (tf * (_K1 + 1)) / (tf + _K1 * (1 - _B + _B * length / self.avg_len))
        return score

    def rank(
        self,
        requirement: str,
        k: int = 5,
        *,
        outcome_aware: bool = False,
        skill_bonus: float = 1.4,
    ) -> list[EvidenceItem]:
        query = tokenize(requirement)
        if not query:
            return []
        query_set = set(query)
        lowered = requirement.lower()

        scored: list[tuple[float, EvidenceItem]] = []
        for index, item in enumerate(self.items):
            score = self._bm25(query, index)

            # An indexed skill naming the requirement is stronger evidence than the same
            # words appearing in prose, so it is scored separately rather than being left
            # to term frequency.
            for skill in item.skills:
                sl = skill.lower()
                if sl in lowered or query_set & set(tokenize(skill)):
                    score += skill_bonus

            if outcome_aware and item.outcome in OUTCOME_RANK:
                # V4's premise, available as an option so the ablation can switch it off:
                # phrasing from a proposal that won is worth more than phrasing that was
                # ignored. Multiplicative so it reorders near-ties without inventing
                # relevance where BM25 found none.
                score *= 1.0 + 0.18 * OUTCOME_RANK[item.outcome]

            if score > 0:
                scored.append((score, item))

        scored.sort(key=lambda pair: (-pair[0], pair[1].id))
        out: list[EvidenceItem] = []
        for score, item in scored[:k]:
            # Copy so a cached corpus item never carries one query's score into the next.
            clone = EvidenceItem(**{**item.__dict__, "score": round(score, 4)})
            out.append(clone)
        return out


def _builtin_rank(
    requirement: str, items: Sequence[EvidenceItem], k: int = 5, **opts: Any
) -> list[EvidenceItem]:
    return BM25Ranker(items).rank(requirement, k, **opts)


def get_ranker():
    """The ranking function to use: Part B's measured retriever, else the built-in.

    Resolved per call, not at import, so enabling a variant in config does not need a
    server restart and this module has no import-time dependency on rag/.
    """
    try:
        from rag.retriever import get_ranker as rag_ranker
    except Exception:
        return _builtin_rank
    try:
        return rag_ranker() or _builtin_rank
    except Exception:
        # A broken variant must not take the tool layer down with it. The Fit Matcher
        # getting baseline evidence beats the Fit Matcher getting an exception.
        return _builtin_rank
