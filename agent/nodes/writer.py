"""Proposal Writer: draft a grounded proposal. Portfolio read only (A6).

The Writer holds no marketplace tools and cannot submit anything. That is deliberate and
it is the single most important allowlist entry in the system: the Writer reads a job post
written by a stranger, so it is the node most exposed to an injected instruction, and a
Writer that *could* call `submit_proposal` is one clever post away from doing so. The
capability to spend Connects belongs to the approval node, downstream of a human decision.

Everything the model may assert is in the prompt's EVIDENCE block, and the draft comes
back with `claim_citations` mapping each sentence to the evidence id behind it. Unsupported
claims are removed by the verifier node next, before the Critic ever sees the text.

On revision, the Critic's reasons and the removed claims are both fed back, so the second
attempt knows what was wrong rather than rerolling the dice.
"""

from __future__ import annotations

from typing import Any

from langchain_core.runnables import RunnableConfig

from agent.handoffs import ProposalDraft
from agent.nodes._common import call_model, evidence_block, node_trace, step
from agent.state import GigBidState, dedupe_evidence


async def writer(state: GigBidState, config: RunnableConfig | None = None) -> dict[str, Any]:
    from mcp_servers.portfolio import tools as portfolio

    spec = state.get("job_spec")
    persona = state.get("persona") or "ai_ml_engineer"
    evidence = dedupe_evidence(state.get("evidence") or [])
    quote = state.get("price_quote")
    revision = int(state.get("revision_count") or 0)

    # What the portfolio cannot support, named explicitly. R5 is easier to satisfy by
    # telling the model what it may not claim than by hoping it infers the boundary.
    unbacked: list[str] = []
    try:
        unbacked = portfolio.list_skills(persona).get("unbacked_skills") or []
    except Exception:
        pass

    # Style and outcome memory, but only promoted rules (C2, R7). An observation that has
    # not met the promotion threshold has no effect on drafting.
    style_rules: list[str] = []
    outcome_note = "(nothing learned for this category yet)"
    try:
        from agent.memory.store import get_memory

        memory = get_memory(user_id=state.get("user_id") or "default")
        style_rules = memory.style_instructions()
        note = memory.category_note(str((state.get("job") or {}).get("category") or ""))
        if note:
            outcome_note = note
    except Exception:
        pass

    revision_note = ""
    if revision:
        critique = state.get("critique")
        removed = (state.get("claim_verification") or {}).get("unsupported_texts") or []
        reasons = getattr(critique, "reasons", []) or []
        fixes = getattr(critique, "suggested_fixes", []) or []
        revision_note = (
            f"THIS IS REVISION {revision}. The previous draft was sent back.\n"
            f"Problems found: {'; '.join(str(r) for r in reasons[:4])}\n"
            f"Specific fixes requested: {'; '.join(str(f) for f in fixes[:4])}\n"
            + (
                f"Claims removed as unsupported, do not reinstate them: {removed[:3]}\n"
                if removed
                else ""
            )
            + "Fix these specifically. Do not rewrite the parts that were not criticised."
        )

    parsed, meta = call_model(
        "writer",
        prompt_values={
            "job_title": (state.get("job") or {}).get("title") or "",
            "job_text": (state.get("job_text_sanitised") or "")[:2500],
            "screening_questions": getattr(spec, "screening_questions", []) or [],
            "evidence": evidence_block(evidence, limit=6),
            "unbacked_skills": unbacked or "(none)",
            "bid_rate": f"${getattr(quote, 'amount_usd', 0):.0f}",
            "style_rules": style_rules or "(no style rules promoted yet)",
            "outcome_note": outcome_note,
            "revision_note": revision_note,
        },
        trace_id=state.get("trace_id"),
        job_id=state.get("job_id"),
    )

    if not isinstance(parsed, dict) or not str(parsed.get("cover_letter") or "").strip():
        # No draft. Not recoverable by rule: a template proposal is worse than none, and
        # the user is better served by a thread marked for attention.
        return {
            "status": "needs_attention",
            "steps_used": step(state),
            "tokens_used": meta.get("tokens") or 0,
            "errors": [
                {
                    "node": "writer",
                    "error": meta.get("error") or "model returned no cover letter",
                    "fatal": False,
                }
            ],
            "execution_trace": node_trace(
                "writer", "error", meta, error=meta.get("error") or "no cover letter"
            ),
        }

    letter = str(parsed["cover_letter"]).strip()
    citations = parsed.get("claim_citations") or {}
    if not isinstance(citations, dict):
        citations = {}

    draft = ProposalDraft(
        job_id=state.get("job_id") or "",
        cover_letter=letter,
        screening_answers=[str(a) for a in (parsed.get("screening_answers") or [])][:10],
        evidence_ids=[str(e.get("id")) for e in evidence[:8] if e.get("id")],
        claim_citations={str(k): [str(v) for v in (vals or [])] for k, vals in citations.items()},
        revision=revision,
        status="drafted",
    )

    return {
        "proposal_draft": draft,
        # Every revision kept: the report needs to show what changed between attempts.
        "draft_history": [
            {"revision": revision, "cover_letter": letter, "model": meta.get("model")}
        ],
        "steps_used": step(state),
        "tokens_used": meta.get("tokens") or 0,
        "execution_trace": node_trace(
            "writer",
            "model",
            meta,
            result_summary={
                "revision": revision,
                "chars": len(letter),
                "citations": len(draft.claim_citations),
                "evidence_offered": len(evidence),
            },
        ),
    }
