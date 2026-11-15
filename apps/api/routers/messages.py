"""Client conversation threads, and running the conversation graph over them.

Mirrors the runs router, one graph down. The same discipline applies: a reply is drafted,
stopped at an interrupt, and only sent once a human approves it through
`POST /approvals/{thread_id}`.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import StreamingResponse

from apps.api.deps import CurrentUser, conversation_graph, thread_id
from apps.api.schemas.models import MessageThread, ThreadSummary
from apps.api.streaming import SSE_HEADERS, stream_graph

router = APIRouter(prefix="/messages", tags=["messages"])


async def _run_state(request: Request, tid: str) -> tuple[dict[str, Any], bool]:
    graph = conversation_graph(request)
    try:
        snapshot = await graph.aget_state({"configurable": {"thread_id": tid}})
    except Exception:
        return {}, False
    if not snapshot or not snapshot.values:
        return {}, False
    return dict(snapshot.values), bool(snapshot.interrupts)


@router.get("", response_model=list[ThreadSummary])
async def list_threads(request: Request, user: dict = CurrentUser) -> list[ThreadSummary]:
    """Client threads, with those awaiting approval first."""
    from mcp_servers.marketplace import server as market

    try:
        payload = await market.list_threads(user["persona"])
    except Exception as exc:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE, f"marketplace unavailable: {exc}"
        ) from exc

    out: list[ThreadSummary] = []
    for thread in payload.get("threads") or []:
        messages = thread.get("messages") or []
        last = messages[-1] if messages else {}
        tid = thread_id("conv", user["user_id"], str(thread.get("thread_id")))
        values, awaiting = await _run_state(request, tid)
        out.append(
            ThreadSummary(
                thread_id=str(thread.get("thread_id")),
                job_id=thread.get("job_id"),
                client_id=thread.get("client_id"),
                message_count=len(messages),
                last_from=last.get("from") or last.get("sender"),
                last_text=str(last.get("text") or "")[:200],
                awaiting_approval=awaiting,
                run_status=values.get("status") if values else None,
            )
        )

    out.sort(key=lambda t: (not t.awaiting_approval, t.thread_id))
    return out


@router.get("/{marketplace_thread_id}", response_model=MessageThread)
async def thread_detail(
    marketplace_thread_id: str, request: Request, user: dict = CurrentUser
) -> MessageThread:
    """One conversation, with the drafted reply if the graph has run."""
    from mcp_servers.marketplace import server as market

    try:
        payload = await market.list_messages(marketplace_thread_id)
    except Exception as exc:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE, f"could not load thread: {exc}"
        ) from exc

    tid = thread_id("conv", user["user_id"], marketplace_thread_id)
    values, awaiting = await _run_state(request, tid)

    return MessageThread(
        thread_id=marketplace_thread_id,
        job_id=values.get("job_id"),
        messages=list(payload.get("messages") or []),
        intent=values.get("intent"),
        risk_flags=list(values.get("risk_flags") or []),
        draft_reply=values.get("draft_reply"),
        proposed_slots=list(values.get("proposed_slots") or []),
        awaiting_approval=awaiting,
        run_status=values.get("status") if values else None,
    )


@router.post("/{marketplace_thread_id}/draft")
async def draft_reply(
    marketplace_thread_id: str, request: Request, user: dict = CurrentUser
) -> StreamingResponse:
    """Run the conversation graph over this thread, streaming progress.

    Ends at an interrupt with a drafted reply, or at `flag_for_user` when the client's
    message carries scam signals, in which case no reply is drafted at all.
    """
    from agent.graphs.conversation_graph import initial_state
    from mcp_servers.marketplace import server as market

    tid = thread_id("conv", user["user_id"], marketplace_thread_id)
    graph = conversation_graph(request)
    config = {"configurable": {"thread_id": tid}}

    _, awaiting = await _run_state(request, tid)
    if awaiting:
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

    try:
        payload = await market.list_messages(marketplace_thread_id)
    except Exception as exc:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE, f"could not load thread: {exc}"
        ) from exc

    state = initial_state(
        marketplace_thread_id,
        user["persona"],
        user_id=user["user_id"],
        messages=list(payload.get("messages") or []),
    )

    return StreamingResponse(
        stream_graph(graph, state, config),
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )
