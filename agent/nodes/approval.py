"""Human approval (C5) and the one place that spends Connects (R4).

Two nodes:

**`approval`** builds the payload and calls `interrupt()`. Everything C5 requires is in
that payload: the draft, the evidence behind every claim, the risk report, the price
rationale and the dry-run preview of exactly what would be sent and what it costs. The
graph stops here. It does not time out, poll or proceed on a default; a thread with nobody
to approve it waits forever, which is the correct behaviour for spending someone's money.

**`submit`** runs only after the interrupt returned an approval, and is the only node in
the system holding `submit_proposal`.

### Exactly once

The idempotency key is minted **here, once**, when the human approves, and stored in state
under a `keep_first` reducer so a resumed thread cannot mint a new one. That matters more
than it sounds: a fresh key per attempt makes idempotency decorative, and the usual way
that happens is a retry helpfully generating a new id.

The flow a crash has to survive is:

1. human approves -> key minted, written to the checkpoint;
2. `submit` claims the key in SQLite, *then* calls the marketplace;
3. result recorded against the key.

Kill the process between 2 and 3 and the key is left in flight. The retry is refused
rather than allowed through, because at that moment nobody knows whether the write landed,
and refusing costs a human looking at a pipeline board while proceeding costs real
Connects. That is the demonstration C6 asks for.

### Edits

The user may approve, edit or reject. An edit is diffed against what the agent wrote, the
diff is recorded as a style-memory observation (never promoted on its own, C2/R7), and the
edited text goes back through Compliance before it is sent, because a human edit can
reintroduce a phone number as easily as a model can.
"""

from __future__ import annotations

from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.types import interrupt

from agent.nodes._common import error_entry, node_trace, step
from agent.state import GigBidState, dedupe_evidence


def build_approval_request(state: GigBidState, preview: dict[str, Any] | None = None) -> dict[str, Any]:
    """The payload a human decides from. C5 lists what must be in it."""
    draft = state.get("proposal_draft")
    fit = state.get("fit_report")
    risk = state.get("risk_report")
    quote = state.get("price_quote")
    critique = state.get("critique")
    verdict = state.get("compliance_verdict")
    verification = state.get("claim_verification") or {}
    evidence = dedupe_evidence(state.get("evidence") or [])
    by_id = {str(e.get("id")): e for e in evidence}

    # Each claim with the evidence behind it, resolved. This is what the proposal studio
    # renders on hover, and what makes "show me the evidence for that claim" a click
    # rather than a search.
    claims: list[dict[str, Any]] = []
    for sentence, ids in (getattr(draft, "claim_citations", {}) or {}).items():
        claims.append(
            {
                "claim": sentence,
                "evidence": [
                    {
                        "id": eid,
                        "citation": (by_id.get(eid) or {}).get("citation"),
                        "project": (by_id.get(eid) or {}).get("project"),
                        "type": (by_id.get(eid) or {}).get("type"),
                        "text": str((by_id.get(eid) or {}).get("text") or "")[:400],
                        "resolved": eid in by_id,
                    }
                    for eid in ids
                ],
            }
        )

    return {
        "kind": "proposal",
        "job": {
            "job_id": state.get("job_id"),
            "title": (state.get("job") or {}).get("title"),
            "client_id": (state.get("job") or {}).get("client_id"),
        },
        "draft": {
            "cover_letter": getattr(draft, "cover_letter", ""),
            "screening_answers": getattr(draft, "screening_answers", []),
            "revision": getattr(draft, "revision", 0),
            "unsupported_claims_removed": getattr(draft, "unsupported_claims_removed", []),
        },
        "claims_with_evidence": claims,
        "risk": {
            "score": getattr(risk, "score", None),
            "flags": getattr(risk, "flags", []),
            "explanation": getattr(risk, "explanation", ""),
            "is_likely_scam": getattr(risk, "is_likely_scam", False),
        },
        "fit": {
            "coverage": getattr(fit, "coverage", None),
            "missing": getattr(fit, "missing_skills", []),
            "requirement_matrix": getattr(fit, "requirement_matrix", []),
        },
        "price": {
            "amount_usd": getattr(quote, "amount_usd", None),
            "type": getattr(quote, "type", None),
            "rationale": getattr(quote, "rationale", ""),
            "above_floor": getattr(quote, "above_floor", None),
            "profile_floor_usd": getattr(quote, "profile_floor_usd", None),
        },
        "critique": {
            "verdict": getattr(critique, "verdict", None),
            "reasons": getattr(critique, "reasons", []),
            "unresolved": getattr(critique, "verdict", None) == "revise",
        },
        "compliance": {
            "status": getattr(verdict, "status", None),
            "redactions": getattr(verdict, "redactions", []),
            "violations": getattr(verdict, "violations", []),
            "rules_cited": getattr(verdict, "policy_rules_cited", []),
        },
        "verification": {
            "checkable_claims": verification.get("checkable_claims"),
            "unsupported_before": verification.get("unsupported_before"),
            "unsupported_after": verification.get("unsupported_after"),
        },
        "dry_run_preview": preview,
        "trace_id": state.get("trace_id"),
        # Spelled out so no UI has to infer it.
        "actions": ["approve", "edit", "reject"],
        "warning": "Approving spends Connects and cannot be undone.",
    }


async def approval(state: GigBidState, config: RunnableConfig | None = None) -> dict[str, Any]:
    """Preview the send, then stop for a human (R4, C5)."""
    draft = state.get("proposal_draft")
    if draft is None or not getattr(draft, "cover_letter", "").strip():
        return {
            "status": "needs_attention",
            "steps_used": step(state),
            "errors": [error_entry("approval", "no draft to approve")],
            "execution_trace": node_trace(
                "approval", "decision", result_summary={"halted": "no draft"}
            ),
        }

    from mcp_servers.marketplace import server as market

    quote = state.get("price_quote")
    proposal = {
        "freelancer_id": state.get("freelancer_id") or state.get("persona") or "",
        "job_id": state.get("job_id") or "",
        "cover_letter": draft.cover_letter,
        "bid_rate": float(getattr(quote, "amount_usd", 0) or 0) or 1.0,
        "screening_answers": [],
    }

    preview: dict[str, Any] | None = None
    degraded: dict[str, Any] = {}
    try:
        # A dry run spends nothing and is the thing the user actually approves.
        preview = await market.create_proposal_draft(proposal)
    except Exception as exc:
        degraded["marketplace"] = str(exc)[:200]

    request = build_approval_request(state, preview)

    # The graph stops here. On resume, `interrupt` returns whatever the API passed to
    # Command(resume=...), which is the user's decision.
    decision = interrupt(request)

    if not isinstance(decision, dict):
        decision = {"action": str(decision or "reject")}

    action = str(decision.get("action") or "reject").lower()
    updates: dict[str, Any] = {
        "approval_request": request,
        "approval_decision": decision,
        "degraded": degraded,
        "steps_used": step(state),
    }

    if action == "reject":
        return {
            **updates,
            "status": "rejected",
            "execution_trace": node_trace(
                "approval",
                "approval",
                result_summary={"action": "reject", "reason": decision.get("reason")},
            ),
        }

    if action == "edit":
        edited = str(decision.get("cover_letter") or "").strip()
        if not edited:
            return {
                **updates,
                "status": "needs_attention",
                "errors": [error_entry("approval", "edit action with no text")],
                "execution_trace": node_trace(
                    "approval", "approval", result_summary={"action": "edit", "error": "no text"}
                ),
            }

        original = draft.cover_letter
        # Record the diff as a style observation. Never promoted on its own: the rule in
        # agent/memory/promotion_rules.py needs the same signal in three separate
        # approvals before it becomes a preference (C2, R7).
        observation: dict[str, Any] = {}
        try:
            from agent.memory.store import get_memory

            observation = get_memory(user_id=state.get("user_id") or "default").observe_edit(
                state.get("job_id") or "unknown",
                original,
                edited,
                reason=str(decision.get("reason") or "") or None,
            )
        except Exception:
            pass

        updates["proposal_draft"] = draft.model_copy(
            update={"cover_letter": edited, "status": "edited_by_user"}
        )
        updates["user_edits"] = [
            {
                "job_id": state.get("job_id"),
                "signals": observation.get("signals") or [],
                "chars_before": len(original),
                "chars_after": len(edited),
                "reason": decision.get("reason"),
            }
        ]

    # Mint the key once, here, at the moment of approval. `keep_first` on the state field
    # means a resumed thread keeps this one rather than generating another.
    from agent.marketplace_port import new_idempotency_key

    key = state.get("idempotency_key") or new_idempotency_key(
        "proposal", str(state.get("job_id") or "")
    )

    return {
        **updates,
        "status": "approved",
        "idempotency_key": key,
        "execution_trace": node_trace(
            "approval",
            "approval",
            result_summary={
                "action": action,
                "approved_by": state.get("user_id"),
                "idempotency_key": key,
                "edited": action == "edit",
            },
        ),
    }


async def submit(state: GigBidState, config: RunnableConfig | None = None) -> dict[str, Any]:
    """Submit the approved proposal. The only node that spends Connects."""
    if state.get("status") != "approved":
        return {
            "steps_used": step(state),
            "execution_trace": node_trace(
                "submit", "decision", result_summary={"skipped": state.get("status")}
            ),
        }

    draft = state.get("proposal_draft")
    quote = state.get("price_quote")
    key = state.get("idempotency_key")

    if not key:
        # Without a key there is no exactly-once guarantee, so refuse rather than send.
        return {
            "status": "needs_attention",
            "steps_used": step(state),
            "errors": [error_entry("submit", "no idempotency key; refusing to send", fatal=True)],
            "execution_trace": node_trace("submit", "error", error="no idempotency key"),
        }

    # A human edit can reintroduce a phone number as easily as a model can, so edited text
    # goes back through the guard before it leaves.
    text = draft.cover_letter
    if (state.get("approval_decision") or {}).get("action") == "edit":
        from agent.nodes.compliance import apply_redactions

        text, redactions, _ = apply_redactions(text)
        if redactions:
            return {
                "status": "needs_attention",
                "steps_used": step(state),
                "errors": [
                    error_entry(
                        "submit",
                        f"the edited text breaks policy and was not sent: {redactions[:2]}",
                    )
                ],
                "execution_trace": node_trace(
                    "submit",
                    "guardrail",
                    result_summary={"blocked_user_edit": redactions[:2]},
                ),
            }

    from mcp_servers.marketplace import server as market

    proposal = {
        "freelancer_id": state.get("freelancer_id") or state.get("persona") or "",
        "job_id": state.get("job_id") or "",
        "cover_letter": text,
        "bid_rate": float(getattr(quote, "amount_usd", 0) or 0) or 1.0,
        "screening_answers": [],
    }

    try:
        result = await market.submit_proposal(
            proposal, key, approved_by=state.get("user_id") or "unknown"
        )
    except Exception as exc:
        message = str(exc)
        # An unreachable marketplace leaves the key in flight on purpose: we do not know
        # whether the write landed, so the thread goes to a human rather than retrying.
        unavailable = "MARKETPLACE_UNAVAILABLE" in message or "in flight" in message
        return {
            "status": "needs_attention" if unavailable else "approved",
            "steps_used": step(state),
            "errors": [error_entry("submit", exc, fatal=unavailable)],
            "degraded": {"marketplace": message[:200]} if unavailable else {},
            "execution_trace": node_trace(
                "submit", "error", error=message[:300], tool="submit_proposal"
            ),
        }

    # Record the outcome for outcome memory. Only promoted once a category has enough
    # samples; one submission teaches nothing.
    try:
        from agent.memory.store import get_memory

        get_memory(user_id=state.get("user_id") or "default").observe_outcome(
            state.get("job_id") or "unknown",
            str((state.get("job") or {}).get("category") or "uncategorised"),
            "submitted",
            proposal_id=result.get("proposal_id"),
        )
    except Exception:
        pass

    return {
        "status": "submitted",
        "submission": result,
        "steps_used": step(state),
        "execution_trace": node_trace(
            "submit",
            "tool_call",
            tool="submit_proposal",
            result_summary={
                "proposal_id": result.get("proposal_id"),
                "connects_spent": result.get("connects_spent"),
                "idempotent_replay": result.get("idempotent_replay"),
                "idempotency_key": key,
            },
        ),
    }
