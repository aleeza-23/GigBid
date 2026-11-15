"""V0 - naive dense: fixed chunks, top-k cosine over the whole job post. Mandatory.

The number everything else has to beat. Two choices make it the honest baseline rather
than a strawman:

* the **whole job post** is the query, un-decomposed, which is what a first attempt at
  this always does and is exactly what V1 is designed to improve on;
* the embedding is a **single vector per item**, so a project described four ways
  competes with itself, which is what V5 would address.

Keeping V0 weak in those specific ways is the point. Making it weak in *other* ways, by
picking a bad k or skipping normalisation, would flatter everything downstream.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from mcp_servers.portfolio.corpus import EvidenceItem
from rag.embeddings import cosine_scores
from rag.index.dense import DenseIndex, get_dense_index


def retrieve(
    requirement: str,
    items: Sequence[EvidenceItem],
    k: int = 5,
    *,
    index: DenseIndex | None = None,
    config: dict[str, Any] | None = None,
    **_: Any,
) -> list[EvidenceItem]:
    """Top-k by cosine similarity on a single embedding per item."""
    if not items:
        return []
    # Pass the config through. Omitting it silently fell back to the LSA development
    # backend while V1 and V2 used the configured model, which would have made the
    # baseline comparison meaningless: V0 would have been losing to a different
    # embedder, not to a better retrieval design.
    dense = index or get_dense_index(items, (config or {}).get("embeddings"))
    query_vector = dense.encode_query(requirement)
    scores = cosine_scores(query_vector, dense.matrix)

    ranked = sorted(zip(dense.items, scores, strict=True), key=lambda p: -float(p[1]))
    return [item.clone(score=round(float(score), 4)) for item, score in ranked[:k]]


VARIANT = {
    "id": "V0",
    "name": "naive_dense",
    "group": "baseline",
    "description": "Fixed chunks, top-k cosine on the whole job post.",
    "retrieve": retrieve,
}
