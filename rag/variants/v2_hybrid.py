"""V2 - hybrid BM25 + dense with reciprocal rank fusion. Retrieval group.

The hypothesis this variant tests, stated plainly: **this corpus rewards exact term
matching more than semantic similarity, because the thing clients name is usually a
library.** "Celery", "dbt", "pgvector" and "Snowflake" each appear two or three times in
592 items, so BM25 gives them high idf and a match dominates. A dense embedding places
them near "task queue", "data pipeline" and "vector database", which is exactly right for
paraphrase and exactly wrong when the client asked for Celery specifically.

Fusion is by reciprocal rank rather than score blending, deliberately. BM25 scores are
unbounded sums of idf terms; cosine similarities sit in [-1, 1]. Normalising them onto a
common scale requires knowing each distribution, and that normalisation becomes a hidden
tuning knob that the ablation cannot see. RRF only reads *rank*, so it needs no
calibration and cannot be quietly tuned.

    RRF(item) = sum over retrievers of 1 / (k_rrf + rank)

k_rrf = 60 is the value from the original Cormack et al. work and is left alone; moving it
would need its own ablation row to be worth claiming.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from mcp_servers.portfolio.corpus import EvidenceItem
from rag.embeddings import cosine_scores
from rag.index.bm25 import get_bm25_index
from rag.index.dense import DenseIndex, get_dense_index

K_RRF = 60


def reciprocal_rank_fusion(
    rankings: list[list[str]], k_rrf: int = K_RRF, weights: list[float] | None = None
) -> dict[str, float]:
    """Fuse several ranked id lists into one score per id."""
    weights = weights or [1.0] * len(rankings)
    fused: dict[str, float] = {}
    for ranking, weight in zip(rankings, weights, strict=True):
        for rank, item_id in enumerate(ranking, start=1):
            fused[item_id] = fused.get(item_id, 0.0) + weight / (k_rrf + rank)
    return fused


def retrieve(
    requirement: str,
    items: Sequence[EvidenceItem],
    k: int = 5,
    *,
    index: DenseIndex | None = None,
    config: dict[str, Any] | None = None,
    candidate_depth: int = 30,
    bm25_weight: float = 1.0,
    dense_weight: float = 1.0,
    **_: Any,
) -> list[EvidenceItem]:
    """Rank by RRF over a BM25 ranking and a dense ranking."""
    if not items:
        return []
    rows = list(items)
    dense = index or get_dense_index(rows, (config or {}).get("embeddings"))
    lexical = get_bm25_index(rows)

    # Each retriever proposes candidates deeper than k, so fusion has something to
    # disagree about. Fusing two top-5 lists mostly reproduces whichever ranked first.
    depth = max(candidate_depth, k * 3)

    bm25_ranking = [item.id for item, _ in lexical.rank(requirement, depth)]

    query_vector = dense.encode_query(requirement)
    scores = cosine_scores(query_vector, dense.matrix)
    dense_ranking = [
        item.id
        for item, _ in sorted(
            zip(dense.items, scores, strict=True), key=lambda p: -float(p[1])
        )[:depth]
    ]

    fused = reciprocal_rank_fusion(
        [bm25_ranking, dense_ranking], weights=[bm25_weight, dense_weight]
    )

    by_id = {item.id: item for item in rows}
    ordered = sorted(fused.items(), key=lambda pair: (-pair[1], pair[0]))

    out: list[EvidenceItem] = []
    for item_id, score in ordered[:k]:
        item = by_id.get(item_id)
        if item is None:
            continue
        # found_by drives the ablation's error analysis: a hit only BM25 surfaced is
        # direct evidence for the exact-term hypothesis above.
        found_by = [
            name
            for name, ranking in (("bm25", bm25_ranking), ("dense", dense_ranking))
            if item_id in ranking
        ]
        out.append(item.clone(score=round(float(score), 5)).annotate(found_by=found_by))
    return out


VARIANT = {
    "id": "V2",
    "name": "hybrid_rrf",
    "group": "retrieval",
    "description": "BM25 + dense fused by reciprocal rank fusion.",
    "retrieve": retrieve,
}
