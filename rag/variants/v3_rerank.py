"""V3 - cross-encoder reranking. Retrieval group.

What it tests: precision at small k. The Writer is shown four or five items, so what
matters is not whether the right evidence is in the top 30 but whether it is in the top 4.

A bi-encoder embeds query and document separately and compares vectors, which is fast and
necessarily lossy: the document was encoded without knowing the query. A cross-encoder
reads both together and scores the pair, which is far more accurate and far too slow to
run over 592 items. So the shape is retrieve-then-rerank: a cheap retriever proposes 30
candidates, the cross-encoder reorders them, the top k ship.

The honest caveat for the report: reranking cannot repair recall. If the right evidence is
not in the candidate set, no reranker will find it, which is why V3 is layered on V2's
candidates rather than V0's. That dependency is also why the ablation has to remove V3 and
V2 separately, or their contributions are impossible to attribute.

With no cross-encoder installed this falls back to a deterministic feature-based reranker
and says so, so the pipeline runs offline. That fallback is not a cross-encoder and its
numbers are labelled accordingly.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any

from mcp_servers.portfolio.corpus import OUTCOME_RANK, EvidenceItem
from rag.index.bm25 import tokenize
from rag.variants import v2_hybrid

_model: Any = None
_model_name = ""
_model_tried = False


def get_cross_encoder() -> Any:
    """Load the reranker once, or return None if unavailable or disabled.

    `RERANK_BACKEND=lexical` forces the deterministic fallback. Tests and CI set it: a
    cross-encoder scoring 30 candidates per query on CPU turns a two-second test file into
    a five-minute one, and what those tests check is the pipeline's wiring, not the
    model's judgement. Production leaves it unset and gets the real reranker.
    """
    global _model, _model_name, _model_tried
    if os.getenv("RERANK_BACKEND", "").lower() == "lexical":
        return None
    if _model_tried:
        return _model
    _model_tried = True
    name = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-base")
    try:
        from sentence_transformers import CrossEncoder

        _model = CrossEncoder(name)
        _model_name = name
    except Exception:
        _model = None
    return _model


def _lexical_rerank_score(query: str, item: EvidenceItem) -> float:
    """Deterministic stand-in when no cross-encoder is installed.

    Not a cross-encoder: it cannot read the pair jointly. It approximates precision by
    weighting an exact skill match above prose overlap, and prefers a case study over a
    code file at equal relevance, because a case study states outcomes the Writer can
    quote. Useful for exercising the pipeline; not a result to report as reranking.
    """
    query_tokens = set(tokenize(query))
    if not query_tokens:
        return 0.0
    lowered = query.lower()

    skill_hits = sum(
        1 for s in item.skills if s.lower() in lowered or set(tokenize(s)) & query_tokens
    )
    text_tokens = set(tokenize(item.text))
    overlap = len(query_tokens & text_tokens) / len(query_tokens)

    type_bonus = {"case_study": 0.30, "review": 0.15, "readme": 0.10, "proposal": 0.05}.get(
        item.type, 0.0
    )
    outcome_bonus = 0.08 * OUTCOME_RANK.get(item.outcome or "", 0)
    return skill_hits * 0.9 + overlap * 1.4 + type_bonus + outcome_bonus


def retrieve(
    requirement: str,
    items: Sequence[EvidenceItem],
    k: int = 5,
    *,
    candidate_depth: int | None = None,
    config: dict[str, Any] | None = None,
    base_retrieve: Any = None,
    **kwargs: Any,
) -> list[EvidenceItem]:
    """Retrieve candidates, then reorder them with a cross-encoder."""
    if not items:
        return []

    # Depth from config, because it is the main cost lever: a cross-encoder runs one
    # forward pass per candidate.
    if candidate_depth is None:
        candidate_depth = int((config or {}).get("candidate_depth") or 12)

    # Candidates from V2 by default: reranking cannot recover evidence the retriever
    # never surfaced, so the better candidate set is the right base.
    propose = base_retrieve or v2_hybrid.retrieve
    candidates = propose(
        requirement, items, candidate_depth, config=config, candidate_depth=candidate_depth, **kwargs
    )
    if not candidates:
        return []

    model = get_cross_encoder()
    if model is not None:
        pairs = [
            (requirement, f"{c.project or ''} {' '.join(c.skills)} {c.text}"[:2000])
            for c in candidates
        ]
        scores = model.predict(pairs)
        reranker = _model_name
    else:
        scores = [_lexical_rerank_score(requirement, c) for c in candidates]
        reranker = "lexical-fallback (NOT a cross-encoder)"

    ranked = sorted(
        zip(candidates, scores, strict=True), key=lambda pair: (-float(pair[1]), pair[0].id)
    )

    return [
        item.clone(score=round(float(score), 4)).annotate(reranker=reranker)
        for item, score in ranked[:k]
    ]


VARIANT = {
    "id": "V3",
    "name": "cross_encoder_rerank",
    "group": "retrieval",
    "description": "V2 candidates reordered by a cross-encoder for precision at small k.",
    "retrieve": retrieve,
}
