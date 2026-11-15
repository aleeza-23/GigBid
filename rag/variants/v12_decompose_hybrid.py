"""V12 - requirement decomposition over hybrid retrieval. Measured, and rejected.

V1's query handling with V2's retriever underneath. It exists because the ablation
suggested this combination would be the best one, and then a direct measurement of it said
otherwise. Keeping the variant and its number is more useful than deleting it: it is the
one configuration a reader would otherwise assume nobody tried.

Part B, 142 dev rows with gold evidence, `BAAI/bge-small-en-v1.5` embeddings:

| variant | Recall@5 | nDCG@10 | ms/query |
|---|---|---|---|
| V0 naive dense, one blended query | 0.582 | 0.750 | 1,325 |
| **V1 decomposition over dense** | **0.651** | **0.774** | **59** |
| V12 decomposition over hybrid (this) | 0.550 | 0.666 | 110 |
| V2 hybrid RRF, one query | 0.581 | 0.768 | 6 |
| V3 cross-encoder rerank | 0.592 | 0.744 | 11,273 |
| V8 corrective loop | 0.571 | 0.747 | 50 |

So V1 ships and this does not. Three readings, in the order they matter:

**Decomposition is the component that works.** One query per requirement beats one blended
query by 6.9 Recall@5 points for about 60 ms.

**Hybrid fusion adds nothing here.** V2 scores 0.581 against the baseline's 0.582. Adding
it under decomposition makes things worse rather than neutral, which is this variant's
result: 0.550 against V1's 0.651.

The reason is the merge. Reciprocal rank fusion scores depend on the ranking of the query
that produced them, so a score from one requirement's ranking is not comparable with a
score from another's. V1 merges cosine similarities, which are on a shared scale; this
variant merges RRF scores by taking the max, and the maximum of six incomparable numbers is
not a ranking. The ablation's equivalent arm scored 0.661 because it fuses all
requirements' rankings globally rather than merging per-requirement scores, which is a
different algorithm from the one here.

**Reranking hurts.** The cross-encoder spends 11.3 seconds a query to finish 1.0 point
above the baseline, and in the ablation removing the reranker raised Recall@5 by 9.8
points. Reranking is a precision tool for deep candidate pools; a 197-item corpus the
retriever has already scored in full is not one, so the reranker can only reorder, and
here it reorders worse.

To run it anyway: `RAG_VARIANT=V12 make rag-eval`.
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
