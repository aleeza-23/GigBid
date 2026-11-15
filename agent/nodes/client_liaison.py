"""Client Liaison: read client replies, classify intent, draft a response (C1).

Marketplace read, marketplace write *after approval*, and Workspace (A6). The only agent
in the system holding a send tool, and even then the send happens in a separate node
downstream of an interrupt.

Three nodes here:

* `read_messages` - pull the thread, sanitise the latest client message;
* `classify_intent` - question, negotiation, interview request or scam signal;
* `draft_reply` - write the reply and, for an interview request, offer real calendar slots.

Client messages get exactly the same treatment as job posts: untrusted data, stripped of
instruction-shaped text before any prompt sees them. A message is arguably the more
dangerous of the two, because a conversation is where a client would plausibly ask to move
off-platform and where a compliant-sounding agent might agree.
"""

from __future__ import annotations

from typing import Any

from langchain_core.runnables import RunnableConfig

from agent.decide import detect_injection, strip_injection
from agent.nodes._common import call_model, compact, evidence_block, node_trace, step
from agent.state import ConversationState


async def read_messages(
    state: ConversationState, config: RunnableConfig | None = None
) -> dict[str, Any]:
    """Load the thread and sanitise the newest client message."""
    from mcp_servers.marketplace import server as market

    thread_id = state.get("thread_id") or ""
    messages = list(state.get("messages") or [])
    degraded: dict[str, Any] = {}

    if not messages and thread_id:
        try:
            payload = await market.list_messages(thread_id)
            messages = list(payload.get("messages") or [])
        except Exception as exc:
            degraded["marketplace"] = str(exc)[:200]

    latest = ""
    for message in reversed(messages):
        sender = str(message.get("from") or message.get("sender") or "")
        if sender == "client":
            latest = str(message.get("text") or "")
            break

    injected, spans = detect_injection(latest)

    return {
        "messages": messages,
        "latest_message": latest,
        "latest_message_sanitised": strip_injection(latest),
        "risk_flags": ["injection_in_client_message"] if injected else [],
        "degraded": degraded,
        "steps_used": step(state),
        "execution_trace": node_trace(
            "read_messages",
            "guardrail" if injected else "tool_call",
            tool="list_messages",
            result_summary={
                "messages": len(messages),
                "injection_detected": injected,
                "spans_removed": len(spans),
            },
        ),
    }


async def classify_intent(
    state: ConversationState, config: RunnableConfig | None = None
) -> dict[str, Any]:
    """What does the client want, and is any of it a policy problem?"""
    message = state.get("latest_message_sanitised") or ""
    if not message.strip():
        return {
            "intent": "other",
            "intent_confidence": 0.0,
            "steps_used": step(state),
            "execution_trace": node_trace(
                "classify_intent", "decision", result_summary={"skipped": "no client message"}
            ),
        }

    # The rule runs first and independently: an off-platform request is findable by
    # pattern, and a model that classified it as a friendly "question" would route it
    # straight past the guard.
    from agent.decide import assess_risk

    rule_risk = assess_risk({"description": message}, {})
    rule_flags = list(rule_risk.flags)

    parsed, meta = call_model(
        "intent",
        prompt_values={"message": message[:2500]},
        trace_id=state.get("trace_id"),
        job_id=state.get("job_id"),
    )
    intent = "other"
    confidence = 0.3
    model_flags: list[str] = []
    wants_slots = False

    if isinstance(parsed, dict):
        candidate = str(parsed.get("intent") or "other")
        if candidate in ("question", "negotiation", "interview_request", "scam_signal", "other"):
            intent = candidate
        confidence = float(parsed.get("confidence") or 0.5)
        model_flags = [str(f) for f in (parsed.get("risk_flags") or [])][:4]
        wants_slots = bool(parsed.get("wants_slots"))

    # The rule can escalate to scam_signal but the model cannot de-escalate away from it.
    if rule_risk.off_platform_contact_requested or rule_risk.free_work_requested:
        intent = "scam_signal"
        confidence = max(confidence, 0.9)

    return {
        "intent": intent,
        "intent_confidence": round(confidence, 3),
        "risk_flags": rule_flags + model_flags,
        "steps_used": step(state),
        "tokens_used": meta.get("tokens") or 0,
        "execution_trace": node_trace(
            "classify_intent",
            "model",
            meta,
            result_summary={
                "intent": intent,
                "confidence": confidence,
                "wants_slots": wants_slots,
                "rule_flags": rule_flags[:3],
            },
        ),
    }


async def draft_reply(
    state: ConversationState, config: RunnableConfig | None = None
) -> dict[str, Any]:
    """Draft the response, offering real calendar slots when a call is wanted."""
    from mcp_servers.portfolio import tools as portfolio

    intent = state.get("intent") or "other"
    message = state.get("latest_message_sanitised") or ""
    persona = state.get("persona") or "ai_ml_engineer"

    slots: list[dict[str, Any]] = []
    degraded: dict[str, Any] = {}
    if intent == "interview_request":
        # Offer only slots the calendar actually has. A model inventing a Tuesday is a
        # meeting the freelancer cannot honour.
        try:
            from mcp_servers.workspace import server as workspace

            payload = workspace.free_interview_slots(days_ahead=7, limit=3)
            slots = list(payload.get("slots") or [])
        except Exception as exc:
            degraded["workspace"] = str(exc)[:200]

    evidence: list[dict[str, Any]] = []
    floor: Any = None
    try:
        skills = portfolio.list_skills(persona)
        floor = skills.get("hourly_rate_usd")
        if intent in ("question", "negotiation"):
            found = portfolio.search_evidence(message[:400], persona, k=3)
            evidence = found.get("hits") or []
    except Exception as exc:
        degraded["portfolio"] = str(exc)[:200]

    style_rules: list[str] = []
    try:
        from agent.memory.store import get_memory

        style_rules = get_memory(user_id=state.get("user_id") or "default").style_instructions()
    except Exception:
        pass

    history = [
        {"from": m.get("from") or m.get("sender"), "text": str(m.get("text") or "")[:300]}
        for m in (state.get("messages") or [])[-6:]
    ]

    parsed, meta = call_model(
        "client_liaison",
        prompt_values={
            "history": compact(history, 1800),
            "message": message[:2000],
            "intent": intent,
            "slots": compact(slots, 600) if slots else "(none available)",
            "evidence": evidence_block(evidence, limit=3, chars=350),
            "floor": f"${floor}/hour" if floor else "(unknown)",
            "style_rules": style_rules or "(no style rules promoted yet)",
        },
        trace_id=state.get("trace_id"),
        job_id=state.get("job_id"),
    )

    reply = ""
    notes = ""
    if isinstance(parsed, dict):
        reply = str(parsed.get("reply") or "").strip()
        notes = str(parsed.get("notes_for_user") or "")[:400]

    if not reply:
        return {
            "status": "needs_attention",
            "degraded": degraded,
            "steps_used": step(state),
            "tokens_used": meta.get("tokens") or 0,
            "errors": [
                {"node": "draft_reply", "error": meta.get("error") or "no reply drafted"}
            ],
            "execution_trace": node_trace(
                "draft_reply", "error", meta, error=meta.get("error") or "no reply"
            ),
        }

    return {
        "draft_reply": reply,
        "proposed_slots": slots,
        "degraded": degraded,
        "steps_used": step(state),
        "tokens_used": meta.get("tokens") or 0,
        "execution_trace": node_trace(
            "draft_reply",
            "model",
            meta,
            result_summary={
                "intent": intent,
                "chars": len(reply),
                "slots_offered": len(slots),
                "notes_for_user": notes[:120],
            },
        ),
    }
