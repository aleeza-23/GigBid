"""The conversation graph: one thread per client conversation (C1).

    START -> read_messages -> classify_intent -> route
                                                  |
                      scam_signal ---------------> flag_for_user -> END
                      question / negotiation /
                      interview_request --------> draft_reply
                                                      v
                                                  compliance
                                                      v
                                        approval  <-- interrupt() stops here
                                                      v
                                                send_message -> END

Shorter than the bid graph and deliberately so. A conversation needs one judgement (what
does the client want) and one artefact (a reply), not five analyses.

Two routing decisions carry the weight:

**A scam signal never reaches the Writer.** If the client asked to move off-platform, pay
in crypto, or get free work, the graph stops and tells the user. Drafting a polite reply
to a scam is worse than useless: it is a reply a hurried user might approve.

**An interrupt before every send, as in the bid graph.** The same approval discipline, the
same idempotency key minted once at approval, because a message cannot be unsent either.
"""

from __future__ import annotations

import time
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from agent.handoffs import ComplianceVerdict
from agent.nodes._common import call_model_async, error_entry, node_trace, step
from agent.nodes.client_liaison import classify_intent, draft_reply, read_messages
from agent.nodes.compliance import apply_redactions, load_policy
from agent.state import ConversationState, trace
from mcp_servers._common.trace import new_trace_id

DRAFTING_INTENTS = ("question", "negotiation", "interview_request", "other")


def route_after_intent(state: ConversationState) -> str:
    """A scam signal stops here; everything else gets a draft."""
    if state.get("status") in ("needs_attention", "dead_letter"):
        return "flag"
    if (state.get("intent") or "other") == "scam_signal":
        return "flag"
    if not (state.get("latest_message_sanitised") or "").strip():
        return "flag"
    return "draft"


async def flag_for_user(
    state: ConversationState, config: RunnableConfig | None = None
) -> dict[str, Any]:
    """Terminal node: record why the thread stopped.

    Every non-send path ends here, including a user's rejection, so it must not overwrite
    a status that already says what happened. Writing `needs_attention` unconditionally
    turned "the user rejected this reply" into "something went wrong", which is both
    wrong and worse: a rejection needs no attention, it has already had it.
    """
    intent = state.get("intent") or "other"
    status = state.get("status") or "needs_attention"

    if status in ("rejected", "submitted", "approved"):
        reason = f"thread ended as {status}"
    elif intent == "scam_signal":
        reason = "the client's message contains scam or policy signals, so no reply was drafted"
        status = "needs_attention"
    elif not (state.get("latest_message_sanitised") or "").strip():
        reason = "no new client message to answer"
        status = "needs_attention"
    else:
        reason = "the reply could not be completed"
        status = "needs_attention"

    return {
        "status": status,
        "execution_trace": trace(
            "flag_for_user",
            "decision",
            result_summary={
                "intent": intent,
                "status": status,
                "reason": reason,
                "risk_flags": (state.get("risk_flags") or [])[:4],
            },
        ),
    }


async def compliance_check(
    state: ConversationState, config: RunnableConfig | None = None
) -> dict[str, Any]:
    """The policy pack, applied to a message rather than a proposal.

    Shares the redaction patterns with the bid graph's Compliance Guard, because the rules
    are the same rules: no contact details, nothing off-platform, no guarantees.
    """
    draft = state.get("draft_reply") or ""
    if not draft.strip():
        return {
            "steps_used": step(state),
            "execution_trace": node_trace(
                "compliance_check", "decision", result_summary={"skipped": "no draft"}
            ),
        }

    cleaned, redactions, rules = apply_redactions(draft)

    violations: list[str] = []
    parsed, meta = await call_model_async(
        "compliance", prompt_values={"policy": load_policy()[:3000], "text": cleaned[:3000]},
        trace_id=state.get("trace_id"),
        job_id=state.get("job_id"),
    )
    if isinstance(parsed, dict):
        violations = [str(v) for v in (parsed.get("violations") or [])][:5]
        for rule in parsed.get("policy_rules_cited") or []:
            if str(rule) not in rules:
                rules.append(str(rule))

    blocked = isinstance(parsed, dict) and str(parsed.get("status")) == "blocked"
    verdict = ComplianceVerdict(
        status="blocked" if blocked else ("fixed" if redactions else "pass"),
        violations=violations,
        redactions=redactions,
        policy_rules_cited=rules,
        explanation=str((parsed or {}).get("explanation") or "")[:300],
        cleaned_text=cleaned if cleaned != draft else None,
    )

    updates: dict[str, Any] = {
        "compliance_verdict": verdict,
        "steps_used": step(state),
        "tokens_used": meta.get("tokens") or 0,
    }
    if blocked:
        updates["status"] = "needs_attention"
    elif verdict.cleaned_text:
        updates["draft_reply"] = verdict.cleaned_text

    return {
        **updates,
        "execution_trace": node_trace(
            "compliance_check",
            "guardrail",
            meta,
            result_summary={
                "status": verdict.status,
                "redactions": len(redactions),
                "violations": violations[:3],
            },
        ),
    }


def build_approval_request(state: ConversationState) -> dict[str, Any]:
    """What a human sees before a message goes to a client."""
    verdict = state.get("compliance_verdict")
    return {
        "kind": "message",
        "thread_id": state.get("thread_id"),
        "job_id": state.get("job_id"),
        "client_message": state.get("latest_message"),
        "client_message_sanitised": state.get("latest_message_sanitised"),
        "intent": state.get("intent"),
        "intent_confidence": state.get("intent_confidence"),
        "risk_flags": state.get("risk_flags") or [],
        "draft_reply": state.get("draft_reply"),
        "proposed_slots": state.get("proposed_slots") or [],
        "compliance": {
            "status": getattr(verdict, "status", None),
            "redactions": getattr(verdict, "redactions", []),
            "violations": getattr(verdict, "violations", []),
        },
        "trace_id": state.get("trace_id"),
        "actions": ["approve", "edit", "reject"],
        "warning": "Approving sends this to the client and cannot be undone.",
    }


async def approval(
    state: ConversationState, config: RunnableConfig | None = None
) -> dict[str, Any]:
    """Stop for a human before sending (R4, C5)."""
    if not (state.get("draft_reply") or "").strip():
        return {
            "status": "needs_attention",
            "steps_used": step(state),
            "execution_trace": node_trace(
                "approval", "decision", result_summary={"halted": "no draft"}
            ),
        }

    request = build_approval_request(state)
    decision = interrupt(request)
    if not isinstance(decision, dict):
        decision = {"action": str(decision or "reject")}

    action = str(decision.get("action") or "reject").lower()
    updates: dict[str, Any] = {
        "approval_request": request,
        "approval_decision": decision,
        "steps_used": step(state),
    }

    if action == "reject":
        return {
            **updates,
            "status": "rejected",
            "execution_trace": node_trace(
                "approval", "approval", result_summary={"action": "reject"}
            ),
        }

    if action == "edit":
        edited = str(decision.get("text") or decision.get("draft_reply") or "").strip()
        if not edited:
            return {
                **updates,
                "status": "needs_attention",
                "errors": [error_entry("approval", "edit action with no text")],
                "execution_trace": node_trace(
                    "approval", "approval", result_summary={"action": "edit", "error": "no text"}
                ),
            }
        original = state.get("draft_reply") or ""
        try:
            from agent.memory.store import get_memory

            get_memory(user_id=state.get("user_id") or "default").observe_edit(
                f"msg-{state.get('thread_id')}", original, edited
            )
        except Exception:
            pass
        updates["draft_reply"] = edited
        updates["user_edits"] = [
            {"thread_id": state.get("thread_id"), "chars_before": len(original),
             "chars_after": len(edited)}
        ]

    from agent.marketplace_port import new_idempotency_key

    key = state.get("idempotency_key") or new_idempotency_key(
        "message", str(state.get("thread_id") or "")
    )

    return {
        **updates,
        "status": "approved",
        "idempotency_key": key,
        "execution_trace": node_trace(
            "approval",
            "approval",
            result_summary={"action": action, "idempotency_key": key},
        ),
    }


async def send(state: ConversationState, config: RunnableConfig | None = None) -> dict[str, Any]:
    """Send the approved message. The only node here that writes to the marketplace."""
    if state.get("status") != "approved":
        return {
            "steps_used": step(state),
            "execution_trace": node_trace(
                "send", "decision", result_summary={"skipped": state.get("status")}
            ),
        }

    key = state.get("idempotency_key")
    text = state.get("draft_reply") or ""
    if not key:
        return {
            "status": "needs_attention",
            "steps_used": step(state),
            "errors": [error_entry("send", "no idempotency key; refusing to send", fatal=True)],
            "execution_trace": node_trace("send", "error", error="no idempotency key"),
        }

    # A human edit can reintroduce what the guard removed, so re-check before sending.
    if (state.get("approval_decision") or {}).get("action") == "edit":
        text, redactions, _ = apply_redactions(text)
        if redactions:
            return {
                "status": "needs_attention",
                "steps_used": step(state),
                "errors": [
                    error_entry("send", f"edited text breaks policy, not sent: {redactions[:2]}")
                ],
                "execution_trace": node_trace(
                    "send", "guardrail", result_summary={"blocked_user_edit": redactions[:2]}
                ),
            }

    from mcp_servers.marketplace import server as market

    try:
        result = await market.send_message(
            str(state.get("thread_id") or ""),
            text,
            idempotency_key=key,
            approved_by=state.get("user_id") or "unknown",
        )
    except Exception as exc:
        message = str(exc)
        unavailable = "MARKETPLACE_UNAVAILABLE" in message or "in flight" in message
        return {
            "status": "needs_attention" if unavailable else "approved",
            "steps_used": step(state),
            "errors": [error_entry("send", exc, fatal=unavailable)],
            "degraded": {"marketplace": message[:200]} if unavailable else {},
            "execution_trace": node_trace("send", "error", error=message[:300]),
        }

    return {
        "status": "submitted",
        "sent": result,
        "steps_used": step(state),
        "execution_trace": node_trace(
            "send",
            "tool_call",
            tool="send_message",
            result_summary={
                "thread_id": result.get("thread_id"),
                "idempotent_replay": result.get("idempotent_replay"),
                "client_replied": bool(result.get("client_reply")),
            },
        ),
    }


def route_after_compliance(state: ConversationState) -> str:
    verdict = state.get("compliance_verdict")
    if getattr(verdict, "status", "pass") == "blocked":
        return "flag"
    return "approval"


def route_after_approval(state: ConversationState) -> str:
    return "send" if state.get("status") == "approved" else "flag"


def build_conversation_graph(checkpointer: Any = None, store: Any = None) -> Any:
    """Compile the conversation graph."""
    graph = StateGraph(ConversationState)

    graph.add_node("read_messages", read_messages)
    graph.add_node("classify_intent", classify_intent)
    graph.add_node("draft_reply", draft_reply)
    graph.add_node("compliance_check", compliance_check)
    graph.add_node("approval", approval)
    graph.add_node("send", send)
    graph.add_node("flag_for_user", flag_for_user)

    graph.add_edge(START, "read_messages")
    graph.add_edge("read_messages", "classify_intent")
    graph.add_conditional_edges(
        "classify_intent", route_after_intent, {"draft": "draft_reply", "flag": "flag_for_user"}
    )
    graph.add_edge("draft_reply", "compliance_check")
    graph.add_conditional_edges(
        "compliance_check",
        route_after_compliance,
        {"approval": "approval", "flag": "flag_for_user"},
    )
    graph.add_conditional_edges(
        "approval", route_after_approval, {"send": "send", "flag": "flag_for_user"}
    )
    graph.add_edge("send", END)
    graph.add_edge("flag_for_user", END)

    return graph.compile(checkpointer=checkpointer, store=store)


def initial_state(
    thread_id: str,
    persona: str,
    *,
    job_id: str = "",
    user_id: str = "default",
    messages: list[dict[str, Any]] | None = None,
) -> ConversationState:
    return {
        "thread_id": thread_id,
        "job_id": job_id,
        "persona": persona,
        "user_id": user_id,
        "trace_id": new_trace_id(),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "messages": list(messages or []),
        "status": "running",
        "steps_used": 0,
        "tokens_used": 0,
    }


def render_mermaid() -> str:
    try:
        return build_conversation_graph().get_graph().draw_mermaid()
    except Exception as exc:  # pragma: no cover
        return f"(could not render: {exc})"
