"""Critic: send a weak draft back, at most twice (C4). Portfolio read only (A6).

The bound lives in the graph's routing, not in this node and not in the prompt. A prompt
that says "you may only ask twice" is a request; a router that counts revisions is a
guarantee. `route_after_critique` is where the ceiling is enforced, and when it is reached
the draft proceeds to Compliance and then to a human, flagged as having failed critique
rather than silently accepted.

The Critic checks four things in order: grounding, personalisation, the opening, and
length. Grounding first because it is the only one that can lose the contract outright,
and because the verifier has already removed what it could prove was unsupported, so
anything the Critic still finds is a citation problem rather than a fabrication.
"""

from __future__ import annotations

from typing import Any

from langchain_core.runnables import RunnableConfig

from agent.handoffs import Critique
from agent.nodes._common import call_model, compact, evidence_block, node_trace, step
from agent.state import GigBidState, dedupe_evidence

# C4: the Critic may send a draft back at most this many times. Read from config so the
# change request can move it without touching code.
DEFAULT_MAX_REVISIONS = 2


def max_revisions(runnable_config: Any = None) -> int:
    from agent.nodes._common import budgets

    return int(budgets(runnable_config).get("max_revisions") or DEFAULT_MAX_REVISIONS)


async def critic(state: GigBidState, config: RunnableConfig | None = None) -> dict[str, Any]:
    draft = state.get("proposal_draft")
    if draft is None:
        return {
            "steps_used": step(state),
            "execution_trace": node_trace(
                "critic", "decision", result_summary={"skipped": "no draft"}
            ),
        }

    revision = int(state.get("revision_count") or 0)
    evidence = dedupe_evidence(state.get("evidence") or [])
    verification = state.get("claim_verification") or {}

    parsed, meta = call_model(
        "critic",
        prompt_values={
            "job_title": (state.get("job") or {}).get("title") or "",
            "job_text": (state.get("job_text_sanitised") or "")[:2000],
            "evidence": evidence_block(evidence, limit=5, chars=300),
            "verification": compact(
                {
                    "checkable_claims": verification.get("checkable_claims"),
                    "unsupported_after": verification.get("unsupported_after"),
                    "removed": verification.get("unsupported_texts", [])[:3],
                },
                500,
            ),
            "draft": draft.cover_letter,
            "revision": revision,
        },
        trace_id=state.get("trace_id"),
        job_id=state.get("job_id"),
    )

    if not isinstance(parsed, dict):
        # A Critic that cannot run must not block a verified draft. Approving here is
        # safe because the human approval gate is still ahead, and the trace records that
        # no critique happened.
        verdict = Critique(
            verdict="approve",
            reasons=[f"critic unavailable ({meta.get('error', 'unknown')}); draft not reviewed"],
            revision_requested=revision,
        )
        return {
            "critique": verdict,
            "steps_used": step(state),
            "tokens_used": meta.get("tokens") or 0,
            "execution_trace": node_trace("critic", "error", meta, error=meta.get("error")),
        }

    verdict = Critique(
        verdict="revise" if str(parsed.get("verdict")) == "revise" else "approve",
        reasons=[str(r) for r in (parsed.get("reasons") or [])][:6],
        grounding_ok=bool(parsed.get("grounding_ok", True)),
        personalised=bool(parsed.get("personalised", True)),
        length_ok=bool(parsed.get("length_ok", True)),
        opening_is_generic=bool(parsed.get("opening_is_generic", False)),
        suggested_fixes=[str(f) for f in (parsed.get("suggested_fixes") or [])][:6],
        revision_requested=revision,
    )

    # An ungrounded draft is sent back regardless of the model's own verdict. The verifier
    # already removed what it could prove; a remaining grounding failure is not something
    # to wave through because the Critic was feeling generous.
    if not verdict.grounding_ok and verdict.verdict == "approve":
        verdict.verdict = "revise"
        verdict.reasons.insert(0, "grounding check failed, overriding approve")

    updates: dict[str, Any] = {
        "critique": verdict,
        "steps_used": step(state),
        "tokens_used": meta.get("tokens") or 0,
    }
    if verdict.verdict == "revise":
        updates["revision_count"] = revision + 1

    return {
        **updates,
        "execution_trace": node_trace(
            "critic",
            "model",
            meta,
            result_summary={
                "verdict": verdict.verdict,
                "revision": revision,
                "grounding_ok": verdict.grounding_ok,
                "personalised": verdict.personalised,
                "reasons": verdict.reasons[:3],
            },
        ),
    }


def route_after_critique(state: GigBidState, config: RunnableConfig | None = None) -> str:
    """Conditional edge: back to the Writer, or on to Compliance.

    This function *is* the loop bound (C4). The Critic can ask for a revision as often as
    it likes; after `max_revisions` the answer is no, and the draft goes forward carrying
    the unresolved critique so the human sees what the Critic objected to.
    """
    critique = state.get("critique")
    if getattr(critique, "verdict", "approve") != "revise":
        return "compliance"
    if int(state.get("revision_count") or 0) > max_revisions(config):
        return "compliance"
    return "rewrite"
