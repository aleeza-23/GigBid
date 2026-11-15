"""Claim verification: split the draft, check each claim, remove what is unsupported (B3).

Runs between the Writer and the Critic, which is the order B3 specifies: unsupported
claims are removed *before* the Critic sees the draft, so the Critic spends its attention
on whether a grounded proposal is any good rather than on catching fabrications.

Deterministic first and always. The lexical and numeric checks in `rag/verify.py` need no
model: a claim stating "accuracy rose to 89%" either has those figures in the cited
evidence or it does not. A model pass refines only the `partly_supported` middle, where
wording differs but meaning may hold, and a verifier that cannot run must never approve by
default.

Three bugs found while building this are worth remembering, because each let something
through or blocked something true:

* a citation handle contributed a phantom number (`ai-p01-case` yielded `01`), which
  failed a true claim;
* a fabricated credential read as conversational filler and was never checked;
* lexical overlap marked "I hold a PhD from Stanford" *supported* at 40% overlap, because
  the evidence mentions machine learning.
"""

from __future__ import annotations

from typing import Any

from langchain_core.runnables import RunnableConfig

from agent.nodes._common import node_trace, step
from agent.state import GigBidState, dedupe_evidence


async def verifier(state: GigBidState, config: RunnableConfig | None = None) -> dict[str, Any]:
    from rag.verify import strip_unsupported, verify_proposal

    draft = state.get("proposal_draft")
    if draft is None:
        return {
            "steps_used": step(state),
            "execution_trace": node_trace(
                "verifier", "decision", result_summary={"skipped": "no draft"}
            ),
        }

    evidence = dedupe_evidence(state.get("evidence") or [])

    llm = None
    try:
        from agent.llm import get_llm

        candidate = get_llm()
        # Offline the model pass would return canned structures, which would look like
        # verification and be worth nothing. The deterministic pass stands alone.
        llm = None if candidate.offline else candidate
    except Exception:
        llm = None

    report = verify_proposal(draft.cover_letter, evidence, llm=llm)
    before = report["unsupported"]

    cleaned, removed = strip_unsupported(draft.cover_letter, report)

    updated = draft.model_copy(
        update={
            "cover_letter": cleaned or draft.cover_letter,
            "unsupported_claims_removed": removed,
            "status": "verified",
        }
    )

    # Re-verify the cleaned text so the reported after-rate describes what the Critic will
    # actually read, not what the Writer produced. B4 asks for the rate before and after.
    after_report = verify_proposal(cleaned or draft.cover_letter, evidence) if removed else report

    return {
        "proposal_draft": updated,
        "claim_verification": {
            **report,
            "unsupported_before": before,
            "unsupported_after": after_report["unsupported"],
            "unsupported_rate_before": report["unsupported_rate"],
            "unsupported_rate_after": after_report["unsupported_rate"],
            "removed_count": len(removed),
            "model_refined": llm is not None,
        },
        "tokens_used": 0,
        "steps_used": step(state),
        "execution_trace": node_trace(
            "verifier",
            "guardrail",
            result_summary={
                "claims": report["checkable_claims"],
                "supported": report["supported"],
                "unsupported_before": before,
                "unsupported_after": after_report["unsupported"],
                "removed": removed[:2],
                "verified_by": "rule+model" if llm else "rule",
            },
        ),
    }
