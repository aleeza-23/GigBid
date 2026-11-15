"""Pending approvals, and resuming the graph when a human decides (C5, R4, D).

This is the hinge between Part C and Part D. The graph calls `interrupt()` and stops; this
router is what lets a person in a browser restart it.

Two properties the UI depends on, and both are about refreshes:

**A pending approval is server state, not client state.** It lives in the checkpoint, so
listing pending approvals is a query against the checkpointer. Close the tab, reopen it on
another machine, and the approval is still there. Part D requires that a page refresh never
loses a pending approval; the only way to get that is to never hold it in the browser.

**Resuming is idempotent at the thread level.** Approving a thread that is no longer
waiting returns its current status rather than re-running the submit node, so a
double-click or a retried request cannot submit twice. Beneath that, the idempotency key
makes the marketplace write exactly-once even if this check were bypassed.

An edit is not a shortcut past the guard: edited text is re-checked by Compliance inside
the graph before it is sent, because a human can reintroduce a phone number as easily as
a model can.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request, status
from langgraph.types import Command

from apps.api.deps import CurrentUser, bid_graph, conversation_graph
from apps.api.schemas.models import (
    ApprovalDecisionRequest,
    ApprovalResult,
    PendingApproval,
)

router = APIRouter(prefix="/approvals", tags=["approvals"])


def _graph_for(request: Request, tid: str) -> Any:
    return conversation_graph(request) if tid.startswith("conv:") else bid_graph(request)


def _owns(tid: str, user_id: str) -> bool:
    """Thread ids embed the user, so this is the tenant check (E4)."""
    return tid.startswith((f"bid:{user_id}:", f"conv:{user_id}:"))


@router.get("", response_model=list[PendingApproval])
async def list_pending(request: Request, user: dict = CurrentUser) -> list[PendingApproval]:
    """Every thread of this user's that is waiting for a decision.

    Queried from the checkpointer rather than an in-memory list, which is what makes a
    pending approval survive a restart of this process.
    """
    checkpointer = request.app.state.checkpointer
    user_id = user["user_id"]
    out: list[PendingApproval] = []
    seen: set[str] = set()

    try:
        async for tup in checkpointer.alist(None, limit=500):
            tid = str((tup.config or {}).get("configurable", {}).get("thread_id") or "")
            if not tid or tid in seen or not _owns(tid, user_id):
                continue
            seen.add(tid)
    except Exception as exc:
        raise HTTPException(
            status.HTTP_501_NOT_IMPLEMENTED,
            f"this checkpointer cannot list threads ({type(exc).__name__}); "
            "open approvals by thread id",
        ) from exc

    for tid in sorted(seen):
        graph = _graph_for(request, tid)
        try:
            snapshot = await graph.aget_state({"configurable": {"thread_id": tid}})
        except Exception:
            continue
        if not snapshot or not snapshot.interrupts:
            continue
        payload = snapshot.interrupts[0].value or {}
        values = dict(snapshot.values or {})
        out.append(
            PendingApproval(
                thread_id=tid,
                kind="message" if tid.startswith("conv:") else "proposal",
                job_id=values.get("job_id"),
                title=(values.get("job") or {}).get("title")
                or (payload.get("job") or {}).get("title"),
                payload=payload,
            )
        )
    return out


@router.get("/{tid}", response_model=PendingApproval)
async def get_pending(tid: str, request: Request, user: dict = CurrentUser) -> PendingApproval:
    """One approval payload: the draft, the evidence behind every claim, risk, price, cost."""
    if not _owns(tid, user["user_id"]):
        raise HTTPException(status.HTTP_403_FORBIDDEN, "not your thread")

    graph = _graph_for(request, tid)
    snapshot = await graph.aget_state({"configurable": {"thread_id": tid}})
    if not snapshot or not snapshot.values:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "no such thread")
    if not snapshot.interrupts:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            f"this thread is not waiting for approval (status "
            f"{dict(snapshot.values).get('status')})",
        )

    values = dict(snapshot.values)
    return PendingApproval(
        thread_id=tid,
        kind="message" if tid.startswith("conv:") else "proposal",
        job_id=values.get("job_id"),
        title=(values.get("job") or {}).get("title"),
        payload=snapshot.interrupts[0].value or {},
    )


@router.post("/{tid}", response_model=ApprovalResult)
async def decide(
    tid: str,
    body: ApprovalDecisionRequest,
    request: Request,
    user: dict = CurrentUser,
) -> ApprovalResult:
    """Approve, edit or reject, and resume the graph.

    This is the only path by which anything GigBid produces reaches a client.
    """
    if not _owns(tid, user["user_id"]):
        raise HTTPException(status.HTTP_403_FORBIDDEN, "not your thread")

    graph = _graph_for(request, tid)
    config = {"configurable": {"thread_id": tid}}
    snapshot = await graph.aget_state(config)
    if not snapshot or not snapshot.values:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "no such thread")

    if not snapshot.interrupts:
        # Already decided. Report where it ended up instead of re-running the submit
        # node, so a double-click or a retried request is harmless.
        values = dict(snapshot.values)
        submission = values.get("submission") or {}
        return ApprovalResult(
            thread_id=tid,
            status=str(values.get("status") or "unknown"),
            action=body.action,
            proposal_id=submission.get("proposal_id"),
            connects_spent=submission.get("connects_spent"),
            idempotent_replay=True,
            message_sent=bool(values.get("sent")),
            errors=list(values.get("errors") or []),
        )

    if body.action == "edit":
        text = (body.cover_letter or body.text or "").strip()
        if not text:
            # Sending the original would silently discard what the human wrote.
            raise HTTPException(
                status.HTTP_422_UNPROCESSABLE_ENTITY,
                "an edit needs the edited text; send cover_letter (proposal) or text (message)",
            )

    resume: dict[str, Any] = {"action": body.action}
    if body.reason:
        resume["reason"] = body.reason
    if body.action == "edit":
        # The two graphs name the field differently because one edits a proposal and the
        # other a message; send both so either graph finds what it expects.
        edited = (body.cover_letter or body.text or "").strip()
        resume["cover_letter"] = edited
        resume["text"] = edited

    try:
        result = await graph.ainvoke(Command(resume=resume), config)
    except Exception as exc:
        raise HTTPException(
            status.HTTP_502_BAD_GATEWAY, f"resuming the thread failed: {exc}"
        ) from exc

    submission = result.get("submission") or {}
    sent = result.get("sent") or {}

    # Promote memory after a decision: an edit contributes a style observation, and the
    # promotion rule decides whether it has been seen often enough to become a rule.
    try:
        from apps.api.deps import memory_for

        memory_for(request, user["user_id"]).promote()
    except Exception:
        pass

    return ApprovalResult(
        thread_id=tid,
        status=str(result.get("status") or "unknown"),
        action=body.action,
        proposal_id=submission.get("proposal_id"),
        connects_spent=submission.get("connects_spent"),
        idempotent_replay=bool(submission.get("idempotent_replay")),
        message_sent=bool(sent),
        client_reply=sent.get("client_reply"),
        errors=list(result.get("errors") or []),
    )
