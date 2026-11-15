"""Server-Sent Events bridge from LangGraph's stream to the browser (D).

Part D requires node-by-node progress streamed from LangGraph, so the user never watches a
spinner through a sixty-second run. SSE rather than WebSockets: the data flows one way,
EventSource reconnects on its own, and it survives proxies that mangle upgrade headers.

Three details that matter more than the transport choice:

**The stream must not be the only record.** Every event here is also in the thread's
`execution_trace`, so a user who reloads mid-run sees the same history rather than an
empty panel. A UI built only on the live stream loses everything on refresh.

**An interrupt is an event, not a disconnect.** When the graph stops for approval the
stream emits `awaiting_approval` and ends cleanly, so the UI can swap the progress panel
for the approval screen instead of showing a dropped connection.

**Errors are events too.** A node that fails emits `node_error` and the stream continues,
because the thread continues: the other analysts still have work to do.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

HEARTBEAT_SEC = 15.0


def sse(event: str, data: Any) -> str:
    """Format one SSE frame. The blank line terminator is what flushes it."""
    payload = json.dumps(data, default=str, ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n"


def _node_event(node: str, update: dict[str, Any]) -> dict[str, Any]:
    """What a node finishing looks like to the UI.

    Only the last trace entry and a few scalars: a node's full update can carry the entire
    evidence list, and shipping that down the wire for every node turns a progress feed
    into megabytes.
    """
    trace = update.get("execution_trace") or []
    latest = trace[-1] if trace else {}
    decision = update.get("decision")
    fit = update.get("fit_report")
    risk = update.get("risk_report")
    quote = update.get("price_quote")
    critique = update.get("critique")
    verdict = update.get("compliance_verdict")

    return {
        "node": node,
        "kind": latest.get("kind"),
        "tool": latest.get("tool"),
        "prompt": latest.get("prompt"),
        "model": latest.get("model"),
        "tokens": latest.get("tokens"),
        "duration_ms": latest.get("duration_ms"),
        "summary": latest.get("result_summary"),
        "error": latest.get("error"),
        # Scalars the UI shows live as they are decided.
        "status": update.get("status"),
        "decision": getattr(decision, "decision", None),
        "fit_coverage": getattr(fit, "coverage", None),
        "risk_score": getattr(risk, "score", None),
        "price_usd": getattr(quote, "amount_usd", None),
        "critic_verdict": getattr(critique, "verdict", None),
        "compliance_status": getattr(verdict, "status", None),
        "revision_count": update.get("revision_count"),
        "steps_used": update.get("steps_used"),
        "tokens_used": update.get("tokens_used"),
    }


async def stream_graph(
    graph: Any,
    payload: Any,
    config: dict[str, Any],
    *,
    heartbeat: float = HEARTBEAT_SEC,
) -> AsyncIterator[str]:
    """Run a graph and yield SSE frames as each node completes.

    `payload` is the initial state for a new run, or a `Command(resume=...)` to continue
    an interrupted one, so the same function serves starting a run and resuming after
    approval.
    """
    yield sse("run_started", {"thread_id": config["configurable"]["thread_id"]})

    queue: asyncio.Queue[tuple[str, Any] | None] = asyncio.Queue()

    async def pump() -> None:
        """Consume the graph's stream into a queue so heartbeats can interleave."""
        try:
            async for chunk in graph.astream(payload, config, stream_mode="updates"):
                for node, update in (chunk or {}).items():
                    if isinstance(update, dict):
                        await queue.put(("node", _node_event(node, update)))
        except Exception as exc:
            await queue.put(("stream_error", {"error": f"{type(exc).__name__}: {exc}"}))
        finally:
            await queue.put(None)

    task = asyncio.create_task(pump())

    try:
        while True:
            try:
                item = await asyncio.wait_for(queue.get(), timeout=heartbeat)
            except TimeoutError:
                # A long node (the Writer on a throttled provider) can exceed any proxy's
                # idle timeout. A comment frame keeps the connection alive without the UI
                # having to handle a fake event.
                yield ": keepalive\n\n"
                continue

            if item is None:
                break
            event, data = item
            if event == "node" and data.get("error"):
                yield sse("node_error", data)
            else:
                yield sse(event, data)
    finally:
        task.cancel()

    # Whatever happened, tell the UI where the thread ended up, so it can switch panels.
    try:
        snapshot = await graph.aget_state(config)
        values = dict(snapshot.values or {})
        if snapshot.interrupts:
            yield sse(
                "awaiting_approval",
                {
                    "thread_id": config["configurable"]["thread_id"],
                    "request": snapshot.interrupts[0].value,
                },
            )
        else:
            yield sse(
                "run_finished",
                {
                    "thread_id": config["configurable"]["thread_id"],
                    "status": values.get("status"),
                    "decision": getattr(values.get("decision"), "decision", None),
                    "reasons": getattr(values.get("decision"), "reasons", []),
                    "tokens_used": values.get("tokens_used"),
                    "steps_used": values.get("steps_used"),
                },
            )
    except Exception as exc:
        yield sse("stream_error", {"error": f"could not read final state: {exc}"})


SSE_HEADERS = {
    "Cache-Control": "no-cache, no-transform",
    "Connection": "keep-alive",
    # Nginx buffers event streams by default, which makes a live feed arrive all at once
    # at the end. This is the header that stops it.
    "X-Accel-Buffering": "no",
}
