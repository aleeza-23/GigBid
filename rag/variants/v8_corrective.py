"""V8 - corrective / self-RAG loop: grade evidence, re-query when weak, refuse when none.

Agentic group (one of V8 or V9 required).

Every other variant answers "what are the best k items for this query?". V8 asks a
different question first: **is this evidence good enough to write from at all?** On a
corpus where the honest answer is sometimes no, that changes the system's behaviour rather
than just its ranking.

The loop:

1. Retrieve with the best non-agentic variant available.
2. **Grade** what came back: is any item strong, and does it actually cover the query's
   distinctive terms?
3. If weak, **re-query** with a reformulation, up to `max_rounds`. Reformulations are
   generalisations — drop the rare qualifier, try the adjacent skill — because the usual
   failure is a query too specific for the corpus.
4. If still weak, **refuse**: return what little there is, flagged `sufficient=False`.

That refusal is the point, and it is why this is the variant that matters for bid-or-skip
accuracy rather than for Recall@5. A graph that knows its evidence is thin can skip the
job, which is the correct action on roughly a fifth of the dev set. A system that always
returns five items will always write a proposal, and some of those proposals will claim
things the portfolio cannot support.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

from mcp_servers.portfolio.corpus import EvidenceItem
from rag.index.bm25 import tokenize
from rag.variants import v2_hybrid

# Below this top score, evidence is treated as too weak to write from. Calibrated against
# the dev set's gold labels rather than guessed; see evals/runners/rag_eval.py.
WEAK_SCORE = 0.02
MIN_TERM_COVERAGE = 0.34


def grade(
    requirement: str, hits: Sequence[EvidenceItem], *, min_coverage: float = MIN_TERM_COVERAGE
) -> dict[str, Any]:
    """Judge a retrieval: is it strong enough to write a grounded claim from?

    Two tests, because they fail differently. A *score* test catches "nothing matched".
    A *term coverage* test catches the more dangerous case: several items scored well on
    the common words while the one distinctive term the client actually asked for appears
    nowhere.
    """
    if not hits:
        return {
            "sufficient": False,
            "reason": "no evidence retrieved",
            "top_score": 0.0,
            "term_coverage": 0.0,
            "uncovered_terms": sorted(set(tokenize(requirement))),
        }

    query_terms = set(tokenize(requirement))
    covered: set[str] = set()
    for hit in hits:
        text = f"{hit.project or ''} {' '.join(hit.skills)} {hit.text}"
        covered |= query_terms & set(tokenize(text))

    coverage = len(covered) / len(query_terms) if query_terms else 1.0
    top_score = max(float(h.score or 0.0) for h in hits)
    uncovered = sorted(query_terms - covered)

    if top_score < WEAK_SCORE:
        return {
            "sufficient": False,
            "reason": f"best score {top_score:.3f} below {WEAK_SCORE}",
            "top_score": top_score,
            "term_coverage": round(coverage, 3),
            "uncovered_terms": uncovered,
        }
    if coverage < min_coverage:
        return {
            "sufficient": False,
            "reason": (
                f"only {coverage:.0%} of the requirement's terms appear in any evidence; "
                f"missing {uncovered[:6]}"
            ),
            "top_score": top_score,
            "term_coverage": round(coverage, 3),
            "uncovered_terms": uncovered,
        }
    return {
        "sufficient": True,
        "reason": f"top score {top_score:.3f}, {coverage:.0%} term coverage",
        "top_score": top_score,
        "term_coverage": round(coverage, 3),
        "uncovered_terms": uncovered,
    }


_REFORMULATE_SYSTEM = """Rewrite a search query that returned weak results.

Reply with JSON only: {"query": "..."}

The corpus is one freelancer's past projects. The usual problem is that the query is too
specific for what they have done, so generalise: drop the rare qualifier, name the broader
capability, or use the adjacent technology. Keep it under 15 words."""


def reformulate(requirement: str, grading: dict[str, Any], llm: Any = None) -> str:
    """Produce a broader query after a weak retrieval."""
    if llm is not None:
        try:
            reply = llm.chat(
                "analyst",
                [
                    {"role": "system", "content": _REFORMULATE_SYSTEM},
                    {
                        "role": "user",
                        "content": (
                            f"Query: {requirement}\n"
                            f"Problem: {grading['reason']}\n"
                            f"Terms with no evidence: {grading.get('uncovered_terms', [])[:8]}"
                        ),
                    },
                ],
                json_mode=True,
            )
            rewritten = str(reply.json().get("query") or "").strip()
            if 3 <= len(rewritten) <= 200:
                return rewritten
        except Exception:
            pass

    # Rule-based generalisation: drop the terms nothing covered, keeping the rest. If a
    # client asked for "Celery on Kubernetes" and Kubernetes appears nowhere, searching
    # for "Celery" alone is the right next move.
    uncovered = set(grading.get("uncovered_terms") or [])
    kept = [t for t in tokenize(requirement) if t not in uncovered]
    if kept:
        return " ".join(kept[:8])
    # Nothing covered at all: fall back to the longest noun-ish token, which is usually
    # the domain word.
    tokens = sorted(tokenize(requirement), key=len, reverse=True)
    return tokens[0] if tokens else requirement


def retrieve(
    requirement: str,
    items: Sequence[EvidenceItem],
    k: int = 5,
    *,
    max_rounds: int | None = None,
    llm: Any = None,
    config: dict[str, Any] | None = None,
    base_retrieve: Any = None,
    **kwargs: Any,
) -> list[EvidenceItem]:
    """Retrieve, grade, re-query if weak, and flag insufficiency rather than hiding it."""
    # From config, because each extra round is another pass over the corpus and on a
    # small host that is the difference between a demo that answers in seconds and one
    # that looks hung. The default stays 2 for evaluation runs.
    if max_rounds is None:
        max_rounds = int((config or {}).get("max_corrective_rounds", 2) or 2)
    propose = base_retrieve or v2_hybrid.retrieve
    query = requirement
    history: list[dict[str, Any]] = []
    best_hits: list[EvidenceItem] = []
    best_grading: dict[str, Any] = {}

    for round_index in range(max_rounds + 1):
        hits = list(propose(query, items, k, config=config, **kwargs))
        grading = grade(requirement, hits)
        history.append({"round": round_index, "query": query, **grading})

        # Keep the strongest attempt, not merely the last: a reformulation can be worse,
        # and returning the worse one because it came last would be perverse.
        if not best_hits or grading.get("top_score", 0) > best_grading.get("top_score", -1):
            best_hits, best_grading = hits, grading

        if grading["sufficient"] or round_index == max_rounds:
            break
        query = reformulate(requirement, grading, llm=llm)

    for hit in best_hits:
        # The graph reads these: a thin retrieval should push the decision node towards
        # skip and must stop the Writer claiming coverage it does not have.
        hit.annotate(
            sufficient=best_grading.get("sufficient", False),
            grading_reason=best_grading.get("reason", ""),
            retrieval_rounds=len(history),
        )
    return best_hits


def retrieve_with_report(
    requirement: str, items: Sequence[EvidenceItem], k: int = 5, **kwargs: Any
) -> dict[str, Any]:
    """Same loop, returning the grading trace. Used by the eval and the run viewer."""
    history: list[dict[str, Any]] = []
    propose = kwargs.pop("base_retrieve", None) or v2_hybrid.retrieve
    max_rounds = kwargs.pop("max_rounds", 2)
    llm = kwargs.pop("llm", None)
    config = kwargs.pop("config", None)

    query = requirement
    best_hits: list[EvidenceItem] = []
    best_grading: dict[str, Any] = {}
    for round_index in range(max_rounds + 1):
        hits = list(propose(query, items, k, config=config, **kwargs))
        grading = grade(requirement, hits)
        history.append(
            {"round": round_index, "query": query, "hit_ids": [h.id for h in hits], **grading}
        )
        if not best_hits or grading.get("top_score", 0) > best_grading.get("top_score", -1):
            best_hits, best_grading = hits, grading
        if grading["sufficient"] or round_index == max_rounds:
            break
        query = reformulate(requirement, grading, llm=llm)

    return {
        "hits": best_hits,
        "sufficient": best_grading.get("sufficient", False),
        "reason": best_grading.get("reason", ""),
        "rounds": len(history),
        "history": history,
        "should_skip": not best_grading.get("sufficient", False),
    }


def _strip_rare(text: str) -> str:
    """Drop parenthetical asides and version numbers, a common over-specification."""
    return re.sub(r"\([^)]*\)|\bv?\d+(\.\d+)+\b", " ", text).strip()


VARIANT = {
    "id": "V8",
    "name": "corrective_rag",
    "group": "agentic",
    "description": "Grade evidence, re-query when weak, refuse to claim when none.",
    "retrieve": retrieve,
    "retrieve_with_report": retrieve_with_report,
}
