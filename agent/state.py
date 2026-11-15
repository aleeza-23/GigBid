"""Typed graph state with reducers (C2).

C2 asks for three things to be kept apart, and to say so plainly:

**State** - what this job or conversation needs *right now*. The job, the analyses, the
evidence, the decision, the price, the drafts, the approvals. It is read by nodes and
written by nodes, and it is what a checkpoint restores.

**Execution trace** - every tool name, argument, result and observation, kept for audit
and debugging. It accumulates and is never read for control flow. Keeping it in state
rather than in a log means a resumed thread still knows what it did before the crash,
which is what makes viva question 4 ("walk through which node got it wrong") answerable
from a checkpoint alone.

**Long-term memory** - promoted knowledge only, in a LangGraph store, written by an
explicit rule in `agent/memory/promotion_rules.py`. Deliberately *not* in this file: state
dies with the thread, memory outlives it. Raw tool output never goes straight into memory.

### Why reducers

Job Analyst, Risk Assessor and Fit Matcher run in parallel with `Send`. Three nodes
returning `{"trace": [...]}` into a plain field is a lost update: LangGraph has no way to
know whether the second write should replace or extend the first, so it replaces. Every
field a parallel branch touches therefore declares how to merge.

Fields only ever written by one node at a time use the default last-write-wins, which is
correct and cheaper than merging.
"""

from __future__ import annotations

import operator
import time
from typing import Annotated, Any, Literal, TypedDict

from agent.handoffs import (
    BidDecision,
    ComplianceVerdict,
    Critique,
    FitReport,
    JobSpec,
    PriceQuote,
    ProposalDraft,
    RiskReport,
)

ThreadStatus = Literal[
    "running",
    "awaiting_approval",
    "approved",
    "rejected",
    "skipped",
    "submitted",
    "needs_attention",
    "dead_letter",
]


def merge_dicts(left: dict[str, Any] | None, right: dict[str, Any] | None) -> dict[str, Any]:
    """Shallow merge, right wins. For maps several branches contribute keys to."""
    return {**(left or {}), **(right or {})}


def keep_first(left: Any, right: Any) -> Any:
    """Keep the earliest non-null value.

    Used for `started_at` and the idempotency key: a resumed thread must not mint a new
    key, because a fresh key on retry is exactly how Connects get spent twice.
    """
    return left if left not in (None, "", [], {}) else right


def take_max(left: int | None, right: int | None) -> int:
    """Highest wins. Budgets and revision counters must never go backwards."""
    return max(left or 0, right or 0)


class TraceEntry(TypedDict, total=False):
    """One observation in the execution trace."""

    ts: str
    node: str
    kind: str  # tool_call | decision | model | error | guardrail | approval
    tool: str | None
    arguments: dict[str, Any] | None
    result_summary: Any
    tokens: int | None
    duration_ms: float | None
    error: str | None


def trace(node: str, kind: str, **fields: Any) -> list[TraceEntry]:
    """Build a single-entry trace list, ready to be returned from a node.

    Returns a list because the field's reducer concatenates; a node returning a bare dict
    would replace the whole trace with one entry.
    """
    entry: TraceEntry = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "node": node,
        "kind": kind,
    }
    entry.update(fields)  # type: ignore[typeddict-item]
    return [entry]


class GigBidState(TypedDict, total=False):
    """One thread per job, through the bid graph."""

    # ------------------------------------------------------------------ identity
    job_id: str
    persona: str
    freelancer_id: str
    user_id: str
    trace_id: str
    started_at: Annotated[str, keep_first]

    # --------------------------------------------------------------------- input
    # The raw post, as the marketplace returned it. Untrusted.
    job: dict[str, Any]
    # The post with instruction-shaped text removed. Nodes read this, never `job`.
    job_text_sanitised: str
    client: dict[str, Any]
    connects: dict[str, Any]
    profile: dict[str, Any]

    # ------------------------------------------------- parallel analysis branches
    # Written concurrently by three nodes, so each declares a reducer. JobSpec,
    # RiskReport and FitReport go to distinct keys, which is why last-write-wins is
    # safe for them individually.
    job_spec: JobSpec
    risk_report: RiskReport
    fit_report: FitReport
    # Evidence is contributed per requirement by the Fit Matcher and added to by the
    # Writer and Critic, so it concatenates and is deduplicated on read.
    evidence: Annotated[list[dict[str, Any]], operator.add]
    # Which branches have reported. The decision node waits on this rather than on a
    # node order, so adding a fourth analyst does not require rewiring the graph.
    analyses_complete: Annotated[list[str], operator.add]

    # ------------------------------------------------------------------ decision
    decision: BidDecision
    price_quote: PriceQuote

    # -------------------------------------------------------------------- drafting
    proposal_draft: ProposalDraft
    # Every revision kept, not just the latest: the Critic loop is bounded at two and the
    # report needs to show what changed between them.
    draft_history: Annotated[list[dict[str, Any]], operator.add]
    claim_verification: dict[str, Any]
    critique: Critique
    compliance_verdict: ComplianceVerdict
    revision_count: Annotated[int, take_max]

    # -------------------------------------------------------------------- approval
    status: ThreadStatus
    approval_request: dict[str, Any]
    approval_decision: dict[str, Any]
    # Minted once when the user approves, then reused on every retry (A2, C6).
    idempotency_key: Annotated[str, keep_first]
    submission: dict[str, Any]
    # The diff between what the agent wrote and what the human sent. Feeds style memory.
    user_edits: Annotated[list[dict[str, Any]], operator.add]

    # ------------------------------------------------------------------- budgets
    # C4: exceeding either moves the thread to a visible "needs attention" state.
    steps_used: Annotated[int, take_max]
    tokens_used: Annotated[int, operator.add]
    budget_exceeded: str | None

    # --------------------------------------------------------------------- trace
    execution_trace: Annotated[list[TraceEntry], operator.add]
    errors: Annotated[list[dict[str, Any]], operator.add]
    # Which MCP servers were unreachable, for the degraded banner (A6).
    degraded: Annotated[dict[str, Any], merge_dicts]


class ConversationState(TypedDict, total=False):
    """One thread per client conversation, through the conversation graph."""

    thread_id: str
    job_id: str
    persona: str
    user_id: str
    trace_id: str
    started_at: Annotated[str, keep_first]

    messages: list[dict[str, Any]]
    # The latest inbound client message, sanitised. Untrusted data, like a job post.
    latest_message: str
    latest_message_sanitised: str

    intent: str
    intent_confidence: float
    risk_flags: Annotated[list[str], operator.add]

    draft_reply: str
    proposed_slots: list[dict[str, Any]]
    compliance_verdict: ComplianceVerdict

    status: ThreadStatus
    approval_request: dict[str, Any]
    approval_decision: dict[str, Any]
    idempotency_key: Annotated[str, keep_first]
    sent: dict[str, Any]
    user_edits: Annotated[list[dict[str, Any]], operator.add]

    steps_used: Annotated[int, take_max]
    tokens_used: Annotated[int, operator.add]
    budget_exceeded: str | None

    execution_trace: Annotated[list[TraceEntry], operator.add]
    errors: Annotated[list[dict[str, Any]], operator.add]
    degraded: Annotated[dict[str, Any], merge_dicts]


def dedupe_evidence(evidence: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Evidence concatenates across branches, so reads deduplicate by id.

    Kept as a read-time helper rather than a reducer: the reducer has to be associative
    and commutative for parallel writes, and "keep the highest-scoring copy" is neither
    once three branches interleave.
    """
    best: dict[str, dict[str, Any]] = {}
    for item in evidence or []:
        key = str(item.get("id") or "")
        if not key:
            continue
        current = best.get(key)
        if current is None or (item.get("score") or 0) > (current.get("score") or 0):
            best[key] = item
    return sorted(best.values(), key=lambda i: -(i.get("score") or 0))


def summarise(state: GigBidState) -> dict[str, Any]:
    """A compact view for the run viewer and the pipeline board.

    Deliberately does not include the execution trace: the UI lists many threads at once
    and shipping every tool call for each would be megabytes.
    """
    decision = state.get("decision")
    fit = state.get("fit_report")
    risk = state.get("risk_report")
    return {
        "job_id": state.get("job_id"),
        "persona": state.get("persona"),
        "status": state.get("status", "running"),
        "trace_id": state.get("trace_id"),
        "decision": getattr(decision, "decision", None),
        "deciding_factor": getattr(decision, "deciding_factor", None),
        "fit_coverage": getattr(fit, "coverage", None),
        "risk_score": getattr(risk, "score", None),
        "price_usd": getattr(state.get("price_quote"), "amount_usd", None),
        "revision_count": state.get("revision_count", 0),
        "steps_used": state.get("steps_used", 0),
        "tokens_used": state.get("tokens_used", 0),
        "budget_exceeded": state.get("budget_exceeded"),
        "unsupported_claims": (state.get("claim_verification") or {}).get("unsupported"),
        "degraded": state.get("degraded") or {},
        "trace_entries": len(state.get("execution_trace") or []),
    }
