"""Starting bid runs, streaming their progress, and reading their traces.

One thread per job per user, with a deterministic id (`bid:<user>:<job>`). That is what
makes a run resumable: the UI can reconnect to a thread it did not start, and a page
refresh mid-run reattaches rather than starting a second one.

`POST /runs` returns immediately with the thread id; the work happens in
`GET /runs/{id}/stream`. Doing it the other way round — running the graph inside the POST —
means a sixty-second request, a proxy timeout, and no progress to show.
"""

from __future__ import annotations

from uuid import uuid4
from typing import Any

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import StreamingResponse

from apps.api.deps import CurrentUser, bid_graph, thread_id
from apps.api.schemas.models import RunDetail, RunSummary, StartRunRequest, TraceEntry
from apps.api.streaming import SSE_HEADERS, stream_graph

router = APIRouter(prefix="/runs", tags=["runs"])


def _summary(tid: str, values: dict[str, Any], awaiting: bool) -> RunSummary:
    from agent.state import summarise

    view = summarise(values) if values else {}
    return RunSummary(
        thread_id=tid,
        job_id=view.get("job_id"),
        status="awaiting_approval" if awaiting else (view.get("status") or "unknown"),
        trace_id=view.get("trace_id"),
        decision=view.get("decision"),
        deciding_factor=view.get("deciding_factor"),
        fit_coverage=view.get("fit_coverage"),
        risk_score=view.get("risk_score"),
        price_usd=view.get("price_usd"),
        revision_count=int(view.get("revision_count") or 0),
        steps_used=int(view.get("steps_used") or 0),
        tokens_used=int(view.get("tokens_used") or 0),
        budget_exceeded=view.get("budget_exceeded"),
        unsupported_claims=view.get("unsupported_claims"),
        awaiting_approval=awaiting,
        degraded=view.get("degraded") or {},
        trace_entries=int(view.get("trace_entries") or 0),
    )


async def _state(request: Request, tid: str) -> tuple[dict[str, Any], bool]:
    graph = bid_graph(request)
    snapshot = await graph.aget_state({"configurable": {"thread_id": tid}})
    if not snapshot or not snapshot.values:
        return {}, False
    return dict(snapshot.values), bool(snapshot.interrupts)


@router.post("", response_model=RunSummary, status_code=status.HTTP_202_ACCEPTED)
async def start_run(
    body: StartRunRequest, request: Request, user: dict = CurrentUser
) -> RunSummary:
    """Register a run and return its thread id. The work happens on the stream.

    Idempotent by thread id: starting a run for a job that already has one returns the
    existing thread rather than a duplicate, so a double-click cannot produce two threads
    bidding on the same job.
    """
    from mcp_servers.marketplace import server as market

    tid = thread_id("bid", user["user_id"], body.job_id)
    existing, awaiting = await _state(request, tid)
    if existing:
        if (
            body.retry
            and not awaiting
            and existing.get("status") in {"needs_attention", "dead_letter", "failed"}
        ):
            # Keep prior traces intact and give the retry its own durable checkpoint.
            tid = thread_id("bid", user["user_id"], f"{body.job_id}:retry:{uuid4().hex[:10]}")
        else:
            return _summary(tid, existing, awaiting)

    try:
        job = await market.get_job(body.job_id)
    except Exception as exc:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE, f"could not load job: {exc}"
        ) from exc

    # The initial state is stashed so the stream endpoint can start the graph without
    # re-fetching, and so a reconnect before the first node knows what it was given.
    request.app.state.pending = getattr(request.app.state, "pending", {})
    request.app.state.pending[tid] = {
        "job": job,
        "persona": body.persona or user["persona"],
        "user_id": user["user_id"],
    }

    return RunSummary(thread_id=tid, job_id=body.job_id, status="queued")


@router.get("/{tid}/stream")
async def stream_run(tid: str, request: Request, user: dict = CurrentUser) -> StreamingResponse:
    """Node-by-node progress over SSE. Starts the run, or resumes watching one."""
    if not tid.startswith(f"bid:{user['user_id']}:"):
        # Thread ids embed the user, so this is the tenant check.
        raise HTTPException(status.HTTP_403_FORBIDDEN, "not your thread")

    graph = bid_graph(request)
    config = {"configurable": {"thread_id": tid}}

    pending = getattr(request.app.state, "pending", {}).pop(tid, None)
    if pending is not None:
        from agent.graphs.bid_graph import initial_state

        payload: Any = initial_state(
            pending["job"],
            pending["persona"],
            freelancer_id=pending["persona"],
            user_id=pending["user_id"],
        )
    else:
        values, awaiting = await _state(request, tid)
        if not values:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "no such run; POST /runs first")
        if awaiting:
            # Already waiting for a human: emit the approval request and close, rather
            # than re-running anything.
            from apps.api.streaming import sse

            snapshot = await graph.aget_state(config)

            async def already_waiting():
                yield sse(
                    "awaiting_approval",
                    {"thread_id": tid, "request": snapshot.interrupts[0].value},
                )

            return StreamingResponse(
                already_waiting(), media_type="text/event-stream", headers=SSE_HEADERS
            )
        payload = None

    return StreamingResponse(
        stream_graph(graph, payload, config),
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )


@router.get("/{tid}", response_model=RunDetail)
async def run_detail(tid: str, request: Request, user: dict = CurrentUser) -> RunDetail:
    """The run viewer: the trace, node by node, with prompt versions and token counts."""
    if not tid.startswith(f"bid:{user['user_id']}:"):
        raise HTTPException(status.HTTP_403_FORBIDDEN, "not your thread")

    values, awaiting = await _state(request, tid)
    if not values:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "no such run")

    trace = [
        TraceEntry(
            ts=entry.get("ts"),
            node=str(entry.get("node") or "?"),
            kind=str(entry.get("kind") or "?"),
            tool=entry.get("tool"),
            prompt=entry.get("prompt"),
            model=entry.get("model"),
            tokens=entry.get("tokens"),
            duration_ms=entry.get("duration_ms"),
            result_summary=entry.get("result_summary"),
            error=entry.get("error"),
            arguments=entry.get("arguments"),
        )
        for entry in (values.get("execution_trace") or [])
    ]

    return RunDetail(
        summary=_summary(tid, values, awaiting),
        trace=trace,
        errors=list(values.get("errors") or []),
        approval_request=values.get("approval_request"),
    )


@router.get("", response_model=list[RunSummary])
async def list_runs(request: Request, user: dict = CurrentUser) -> list[RunSummary]:
    """Every run this user has. Backs the pipeline board and the runs list.

    Reads the checkpointer's thread list rather than a separate table, so there is one
    source of truth about what has happened and no chance of the two drifting.
    """
    prefix = f"bid:{user['user_id']}:"
    out: list[RunSummary] = []
    seen: set[str] = set()

    # Ask the checkpointer for its threads. Not every backend supports listing, which is
    # why the failure below is a 501 rather than an empty list: "no runs" and "I cannot
    # tell you" are different answers and the UI shows them differently.
    try:
        checkpointer = request.app.state.checkpointer
        async for tup in checkpointer.alist(None, limit=500):
            tid = str((tup.config or {}).get("configurable", {}).get("thread_id") or "")
            if not tid.startswith(prefix) or tid in seen:
                continue
            seen.add(tid)
            values, awaiting = await _state(request, tid)
            if values:
                out.append(_summary(tid, values, awaiting))
    except Exception:
        # No listing support: return nothing rather than a misleading empty-but-certain
        # answer. The UI shows "run history unavailable" instead of "no runs".
        raise HTTPException(
            status.HTTP_501_NOT_IMPLEMENTED,
            "this checkpointer cannot list threads; open runs by id",
        ) from None

    out.sort(key=lambda r: (not r.awaiting_approval, r.job_id or ""))
    return out
