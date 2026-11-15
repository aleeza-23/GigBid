"""V12 - requirement decomposition over hybrid retrieval. The configuration that ships.

Not a new idea: it is V1's query handling with V2's retriever underneath, and it exists as
its own variant because the measurements said that combination is the best one and neither
V1 nor V2 is it.

What Part B found, on 142 dev rows with BGE embeddings:

| variant | Recall@5 | nDCG@10 | ms/query |
|---|---|---|---|
| V0 naive dense | 0.582 | 0.750 | 754 |
| V1 decompose (dense) | 0.651 | 0.774 | 56 |
| V2 hybrid RRF | 0.581 | 0.768 | 4 |
| V3 cross-encoder rerank | 0.592 | 0.744 | 11,737 |
| V8 corrective | 0.571 | 0.747 | 65 |

And from the ablation, where the arm "full minus rerank" is exactly this variant: 0.661
Recall@5 and 0.825 nDCG@10, the best of any arm, against 0.563 and 0.670 with the reranker
in place. Removing reranking *raised* Recall@5 by 9.8 points.

So reranking is not merely failing to earn its latency here, it is actively reordering good
candidates downwards. Both rerankers do it: the real cross-encoder takes 11.7 seconds a
query to finish 1 point above the naive baseline, and the lexical stand-in costs 9.8
points. The honest reading is that neither reranker has anything to add to a 197-item
corpus where the hybrid retriever has already seen every candidate -- reranking is a
precision tool for deep candidate pools, and this pool is shallow.

Decomposition is what carries the pipeline, which is the one place the original hypothesis
held: retrieving once per requirement beats one blended query by 6.9 points for about 50 ms.

`candidate_depth` is irrelevant here because nothing reranks, and that is the point: the
shipped configuration has no model in the retrieval path at all, so a job's retrieval cost
does not grow with its requirement count beyond a few milliseconds each.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from mcp_servers.portfolio.corpus import EvidenceItem
from rag.variants import v1_decompose, v2_hybrid


def retrieve(
    requirement: str | dict[str, Any],
    items: Sequence[EvidenceItem],
    k: int = 5,
    *,
    config: dict[str, Any] | None = None,
    llm: Any = None,
    requirements: list[str] | None = None,
    **kwargs: Any,
) -> list[EvidenceItem]:
    """Decompose the post, retrieve hybrid per requirement, merge by best score."""
    if not items:
        return []

    parts = requirements or v1_decompose.decompose(requirement, llm=llm)
    if not parts:
        parts = [str(requirement)]

    # Headroom per requirement, for the same reason V1 takes it: merging k results per
    # requirement means a requirement whose best evidence sits fourth never contributes.
    per_requirement = max(3, k)

    best: dict[str, tuple[float, EvidenceItem, list[str]]] = {}
    for part in parts:
        for item in v2_hybrid.retrieve(part, items, per_requirement, config=config, **kwargs):
            score = float(item.score or 0.0)
            current = best.get(item.id)
            if current is None:
                best[item.id] = (score, item, [part])
            else:
                # Max, not sum: one requirement proved completely beats five proved vaguely.
                best[item.id] = (max(current[0], score), current[1], [*current[2], part])

    ordered = sorted(best.values(), key=lambda row: (-row[0], row[1].id))
    out: list[EvidenceItem] = []
    for score, item, covered in ordered[:k]:
        out.append(
            item.clone(score=round(score, 4)).annotate(
                retriever="v12_decompose_hybrid",
                covers_requirements=sorted(set(covered)),
                requirements_total=len(parts),
            )
        )
    return out


VARIANT = {
    "id": "V12",
    "name": "decompose_hybrid",
    "group": "retrieval",
    "description": (
        "V1's per-requirement queries over V2's hybrid retriever, with no reranking. "
        "The best measured configuration in Part B."
    ),
    "retrieve": retrieve,
}
