"""Scout: hand a job into the graph, sanitised.

The Scout's search-and-dedup work happens in the worker (E1) on a schedule; by the time a
bid thread starts, the job is already chosen. What this node does is the part that must
happen inside the graph, once, before any other node reads the post:

* fetch the job and its client through the marketplace port;
* **strip instruction-shaped text** and record what was removed.

Sanitising here rather than in each node is the whole defence. Nodes read
`job_text_sanitised`; nothing downstream touches `job["description"]`. A node added next
week inherits the defence by reading the field everything else reads.
"""

from __future__ import annotations

from typing import Any

from langchain_core.runnables import RunnableConfig

from agent.decide import detect_injection, strip_injection
from agent.nodes._common import error_entry, node_trace, step
from agent.state import GigBidState


async def scout(state: GigBidState, config: RunnableConfig | None = None) -> dict[str, Any]:
    """Load the job and client, then sanitise the post text."""
    # Adopt the thread's trace id into the context, first thing (E3).
    #
    # `initial_state` mints the id into state, but the spans every downstream node writes
    # are keyed on the ContextVar. Without this line those are two different ids, and the
    # trace splits in half: the API's id on one side, a freshly minted one on the other,
    # so "one trace id end to end" is false and a run cannot be reconstructed. The Scout
    # is the right place because it runs before the fan-out, and the parallel branches
    # inherit the context from here.
    from mcp_servers._common.trace import set_trace_id, set_user_id

    if state.get("trace_id"):
        set_trace_id(str(state["trace_id"]))
    if state.get("user_id"):
        set_user_id(str(state["user_id"]))

    job = dict(state.get("job") or {})
    job_id = state.get("job_id") or job.get("job_id") or ""
    updates: dict[str, Any] = {"steps_used": step(state)}

    # The job may already be in state because the worker put it there. Fetch only what is
    # absent, so a replayed thread does not re-hit the marketplace.
    if not job and job_id:
        from mcp_servers.marketplace import server as market

        try:
            job = await market.get_job(job_id)
            updates["job"] = job
        except Exception as exc:
            return {
                **updates,
                "status": "needs_attention",
                "errors": [error_entry("scout", exc, fatal=True)],
                "degraded": {"marketplace": str(exc)[:200]},
                "execution_trace": node_trace("scout", "error", error=str(exc)),
            }

    client = dict(state.get("client") or {})
    if not client and job.get("client_id"):
        from mcp_servers.marketplace import server as market

        try:
            client = await market.get_client_history(job["client_id"])
            updates["client"] = client
        except Exception as exc:
            # A missing client history is a degraded run, not a dead one: the Risk
            # Assessor can still read the post, and saying so beats failing the thread.
            updates["degraded"] = {"client_history": str(exc)[:200]}

    raw = f"{job.get('title') or ''}\n\n{job.get('description') or ''}"
    injected, spans = detect_injection(raw)
    updates["job_text_sanitised"] = strip_injection(raw)
    updates["job_id"] = job_id or job.get("job_id") or ""

    return {
        **updates,
        "execution_trace": node_trace(
            "scout",
            "guardrail" if injected else "tool_call",
            tool="get_job",
            result_summary={
                "job_id": updates["job_id"],
                "title": (job.get("title") or "")[:80],
                "injection_detected": injected,
                "spans_removed": len(spans),
            },
        ),
    }
