"""Workspace MCP server (A3): interview slots, tentative holds and email drafts.

Five tools over a swappable backend. The default is the documented local mock; setting
WORKSPACE_BACKEND=google moves the same tools onto Calendar and Gmail without the Client
Liaison noticing.

Nothing here sends anything. `create_email_draft` creates a draft and there is no send
tool on this server, which is R4 expressed as an absence rather than a check: a bug
upstream cannot cause an email to go out, because no code path exists that would.

Run it:

    python -m mcp_servers.workspace.server            # stdio
    python -m mcp_servers.workspace.server --http     # Streamable HTTP on :8703
"""

from __future__ import annotations

import argparse
import os
import time
from typing import Any

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations

from mcp_servers._common.audit import AuditLog
from mcp_servers._common.trace import get_trace_id
from mcp_servers.workspace import calendar_mock as backend_module

SERVER_NAME = "gigbid-workspace"
INSTRUCTIONS = """The freelancer's own calendar and email drafts.

Use `free_interview_slots` before offering times to a client, so the offer is real. Hold a
slot with `create_interview_hold` only once the client has picked one; holding every
offered slot blocks the week for a conversation that may go nowhere.

This server drafts email but cannot send it. Every draft needs the user to review and send
it themselves."""

audit = AuditLog(server=SERVER_NAME)
mcp = FastMCP(SERVER_NAME, instructions=INSTRUCTIONS)

_backend: Any = None


def get_backend() -> Any:
    global _backend
    if _backend is None:
        _backend = backend_module.get_backend()
    return _backend


def set_backend_for_tests(instance: Any) -> None:
    global _backend
    _backend = instance


READ = ToolAnnotations(
    readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=False
)

# Creates a calendar entry. Additive, so destructiveHint=False; not idempotent, because
# calling it twice books two slots, which is exactly the mistake to warn a host about.
HOLD = ToolAnnotations(
    readOnlyHint=False, destructiveHint=False, idempotentHint=False, openWorldHint=False
)

# Removes a hold. Destructive by MCP's definition: it undoes existing state. Idempotent,
# because releasing an already-released hold is a no-op that reports false.
RELEASE = ToolAnnotations(
    readOnlyHint=False, destructiveHint=True, idempotentHint=True, openWorldHint=False
)

DRAFT = ToolAnnotations(
    readOnlyHint=False, destructiveHint=False, idempotentHint=False, openWorldHint=False
)


def _audited(tool: str, fn, arguments: dict[str, Any]) -> Any:
    started = time.perf_counter()
    try:
        result = fn()
    except ValueError as exc:
        audit.record(tool, arguments, status="rejected", error=str(exc))
        raise
    except Exception as exc:
        audit.record(tool, arguments, status="error", error=f"{type(exc).__name__}: {exc}")
        raise ValueError(f"workspace server failed handling {tool}: {exc}") from exc
    audit.record(tool, arguments, result=result, duration_ms=(time.perf_counter() - started) * 1000)
    return result


@mcp.tool(annotations=READ, structured_output=True)
def free_interview_slots(
    days_ahead: int = 7, duration_minutes: int = 30, limit: int = 5
) -> dict[str, Any]:
    """Find open interview slots in the freelancer's calendar.

    Returns slots inside working hours on weekdays, starting tomorrow, one per day so the
    options are spread across the week. Call this before offering times to a client; a
    slot offered without checking is a slot you may have to withdraw.
    """
    args = {"days_ahead": days_ahead, "duration_minutes": duration_minutes, "limit": limit}
    slots = _audited(
        "free_interview_slots",
        lambda: [s.to_dict() for s in get_backend().free_slots(days_ahead, duration_minutes, limit)],
        args,
    )
    return {
        "slots": slots,
        "timezone_offset_hours": backend_module.TZ_OFFSET_HOURS,
        "backend": getattr(get_backend(), "name", "unknown"),
        "trace_id": get_trace_id(),
    }


@mcp.tool(annotations=HOLD, structured_output=True)
def create_interview_hold(
    start: str, duration_minutes: int = 30, title: str = "Interview", notes: str = ""
) -> dict[str, Any]:
    """Hold a slot tentatively for an interview.

    `start` is ISO 8601; a bare local time is read in the user's timezone. The hold is
    tentative, not a confirmed meeting, and it is refused if it clashes with an existing
    one. Call this only after the client picks a time, not for every slot you offered.
    """
    parsed = backend_module.parse_start(start)
    args = {"start": start, "duration_minutes": duration_minutes, "title": title}
    return _audited(
        "create_interview_hold",
        lambda: get_backend().create_hold(parsed, duration_minutes, title, notes),
        args,
    )


@mcp.tool(annotations=RELEASE, structured_output=True)
def release_interview_hold(hold_id: str) -> dict[str, Any]:
    """Release a tentative hold, freeing the slot.

    Call when a client goes quiet or reschedules. Reports released=false when the hold was
    already gone, which is a success, not an error.
    """
    released = _audited(
        "release_interview_hold",
        lambda: get_backend().release_hold(hold_id),
        {"hold_id": hold_id},
    )
    return {"hold_id": hold_id, "released": bool(released)}


@mcp.tool(annotations=READ, structured_output=True)
def list_interview_holds() -> dict[str, Any]:
    """List the tentative interview holds currently on the calendar, soonest first.

    Released holds are excluded. Use this before offering new slots to see what is
    already promised, and to find the hold_id to release when a client reschedules or
    goes quiet.
    """
    holds = _audited("list_interview_holds", lambda: get_backend().list_holds(), {})
    return {"total": len(holds), "holds": holds}


@mcp.tool(annotations=DRAFT, structured_output=True)
def create_email_draft(
    to: str, subject: str, body: str, thread_ref: str | None = None
) -> dict[str, Any]:
    """Create an email draft for the user to review and send.

    This server cannot send email; it has no send tool. The draft waits for the user.
    Use `thread_ref` to tie the draft to a marketplace conversation so the UI can show
    them together.
    """
    if "@" not in to:
        raise ValueError(f"{to!r} is not an email address")
    args = {"to": to, "subject": subject, "chars": len(body), "thread_ref": thread_ref}
    result = _audited(
        "create_email_draft",
        lambda: get_backend().create_email_draft(to, subject, body, thread_ref),
        args,
    )
    return {**result, "trace_id": get_trace_id()}


@mcp.tool(annotations=READ, structured_output=True)
def workspace_health() -> dict[str, Any]:
    """Is the workspace backend usable? Drives the degraded-mode banner."""
    try:
        instance = get_backend()
        instance.free_slots(1, 30, 1)
    except Exception as exc:
        return {"status": "unavailable", "detail": str(exc), "trace_id": get_trace_id()}
    return {
        "status": "ok",
        "backend": getattr(instance, "name", "unknown"),
        "sends_email": False,
        "trace_id": get_trace_id(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="GigBid Workspace MCP server")
    parser.add_argument("--http", action="store_true")
    parser.add_argument("--port", type=int, default=int(os.getenv("WORKSPACE_MCP_PORT", "8703")))
    parser.add_argument("--host", default=os.getenv("WORKSPACE_MCP_HOST", "127.0.0.1"))
    args = parser.parse_args()
    if args.http:
        mcp.settings.host = args.host
        mcp.settings.port = args.port
        mcp.run(transport="streamable-http")
    else:
        mcp.run()


if __name__ == "__main__":
    main()
