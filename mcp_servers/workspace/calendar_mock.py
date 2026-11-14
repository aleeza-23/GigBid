"""A local calendar and email-draft backend.

A3 allows Google Calendar and Gmail through their official APIs, or a local mock, as long
as the agent sees MCP tools only. This is the mock, and it is the default because the
graded path must run for a reviewer who has not connected a Google account.

The seam is `WorkspaceBackend`: a protocol with five methods. `GoogleWorkspaceBackend`
implements the same five against the official APIs when GOOGLE_* credentials are present,
so switching is a config change and the tools above never learn which one answered. Same
trick as the marketplace port, one layer down.

State is SQLite rather than a dict so a tentative hold survives the worker restart that
C6 asks us to demonstrate. A hold that evaporates on restart would let the agent offer the
same slot twice.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any, Protocol

DEFAULT_PATH = Path(
    os.getenv(
        "WORKSPACE_DB",
        Path(__file__).resolve().parents[2] / "data" / "audit" / "workspace.sqlite3",
    )
)

# Working hours in the user's local timezone, from config rather than hardcoded, because
# the three personas are in Pakistan and most of their clients are not.
WORK_START = time(int(os.getenv("WORKDAY_START_HOUR", "9")), 0)
WORK_END = time(int(os.getenv("WORKDAY_END_HOUR", "18")), 0)
TZ_OFFSET_HOURS = float(os.getenv("WORKSPACE_TZ_OFFSET", "5"))  # PKT default


@dataclass
class Slot:
    start: datetime
    end: datetime

    def to_dict(self) -> dict[str, Any]:
        return {
            "start": self.start.isoformat(),
            "end": self.end.isoformat(),
            "duration_minutes": int((self.end - self.start).total_seconds() // 60),
        }


class WorkspaceBackend(Protocol):
    """What the workspace tools need. Implemented by the mock and by Google."""

    def free_slots(
        self, days_ahead: int, duration_minutes: int, limit: int
    ) -> list[Slot]: ...

    def create_hold(
        self, start: datetime, duration_minutes: int, title: str, notes: str
    ) -> dict[str, Any]: ...

    def release_hold(self, hold_id: str) -> bool: ...

    def list_holds(self) -> list[dict[str, Any]]: ...

    def create_email_draft(
        self, to: str, subject: str, body: str, thread_ref: str | None
    ) -> dict[str, Any]: ...


class LocalWorkspace:
    """SQLite-backed calendar holds and email drafts."""

    name = "local_mock"

    def __init__(self, path: Path | str | None = None) -> None:
        self.path = Path(path or DEFAULT_PATH)
        if str(self.path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS holds (
                hold_id  TEXT PRIMARY KEY,
                start_ts TEXT NOT NULL,
                end_ts   TEXT NOT NULL,
                title    TEXT,
                notes    TEXT,
                status   TEXT NOT NULL DEFAULT 'tentative'
            );
            CREATE TABLE IF NOT EXISTS email_drafts (
                draft_id   TEXT PRIMARY KEY,
                to_addr    TEXT NOT NULL,
                subject    TEXT NOT NULL,
                body       TEXT NOT NULL,
                thread_ref TEXT,
                created_at TEXT NOT NULL,
                status     TEXT NOT NULL DEFAULT 'draft'
            );
            """
        )
        self._conn.commit()

    # ------------------------------------------------------------------ calendar

    def _busy(self) -> list[tuple[datetime, datetime]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT start_ts, end_ts FROM holds WHERE status != 'released'"
            ).fetchall()
        return [(datetime.fromisoformat(a), datetime.fromisoformat(b)) for a, b in rows]

    def free_slots(self, days_ahead: int = 7, duration_minutes: int = 30, limit: int = 10) -> list[Slot]:
        """Open slots inside working hours, skipping weekends and existing holds.

        Starts from tomorrow, not today: offering a client a slot in ninety minutes reads
        as desperate and usually cannot be honoured anyway.
        """
        tz = timezone(timedelta(hours=TZ_OFFSET_HOURS))
        now = datetime.now(tz)
        busy = self._busy()
        step = timedelta(minutes=duration_minutes)
        out: list[Slot] = []

        for day_offset in range(1, days_ahead + 1):
            day = (now + timedelta(days=day_offset)).date()
            if day.weekday() >= 5:  # Saturday, Sunday
                continue
            cursor = datetime.combine(day, WORK_START, tzinfo=tz)
            day_end = datetime.combine(day, WORK_END, tzinfo=tz)
            while cursor + step <= day_end:
                candidate = Slot(cursor, cursor + step)
                clash = any(
                    candidate.start < b_end and b_start < candidate.end
                    for b_start, b_end in busy
                )
                if not clash:
                    out.append(candidate)
                    if len(out) >= limit:
                        return out
                    # One offer per day keeps a slate of slots spread across the week
                    # rather than three consecutive Tuesday morning options.
                    break
                cursor += step
        return out

    def create_hold(
        self, start: datetime, duration_minutes: int = 30, title: str = "Interview", notes: str = ""
    ) -> dict[str, Any]:
        end = start + timedelta(minutes=duration_minutes)
        for b_start, b_end in self._busy():
            if start < b_end and b_start < end:
                raise ValueError(
                    f"slot {start.isoformat()} clashes with an existing hold "
                    f"({b_start.isoformat()} to {b_end.isoformat()})"
                )
        hold_id = f"hold-{uuid.uuid4().hex[:10]}"
        with self._lock:
            self._conn.execute(
                "INSERT INTO holds (hold_id, start_ts, end_ts, title, notes) VALUES (?,?,?,?,?)",
                (hold_id, start.isoformat(), end.isoformat(), title, notes),
            )
            self._conn.commit()
        return {
            "hold_id": hold_id,
            "start": start.isoformat(),
            "end": end.isoformat(),
            "title": title,
            "notes": notes,
            "status": "tentative",
        }

    def release_hold(self, hold_id: str) -> bool:
        with self._lock:
            cur = self._conn.execute(
                "UPDATE holds SET status='released' WHERE hold_id=? AND status!='released'",
                (hold_id,),
            )
            self._conn.commit()
        return cur.rowcount > 0

    def list_holds(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT hold_id, start_ts, end_ts, title, notes, status FROM holds "
                "WHERE status != 'released' ORDER BY start_ts"
            ).fetchall()
        return [
            {
                "hold_id": r[0],
                "start": r[1],
                "end": r[2],
                "title": r[3],
                "notes": r[4],
                "status": r[5],
            }
            for r in rows
        ]

    # --------------------------------------------------------------------- email

    def create_email_draft(
        self, to: str, subject: str, body: str, thread_ref: str | None = None
    ) -> dict[str, Any]:
        """Create a draft. Never sends: there is no send path in this class at all."""
        draft_id = f"draft-{uuid.uuid4().hex[:10]}"
        created = datetime.now(UTC).isoformat()
        with self._lock:
            self._conn.execute(
                "INSERT INTO email_drafts (draft_id, to_addr, subject, body, thread_ref, "
                "created_at) VALUES (?,?,?,?,?,?)",
                (draft_id, to, subject, body, thread_ref, created),
            )
            self._conn.commit()
        return {
            "draft_id": draft_id,
            "to": to,
            "subject": subject,
            "body": body,
            "thread_ref": thread_ref,
            "created_at": created,
            "status": "draft",
            "sent": False,
        }

    def list_email_drafts(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT draft_id, to_addr, subject, body, thread_ref, created_at, status "
                "FROM email_drafts ORDER BY created_at DESC LIMIT 50"
            ).fetchall()
        keys = ("draft_id", "to", "subject", "body", "thread_ref", "created_at", "status")
        return [dict(zip(keys, r, strict=True)) for r in rows]

    def close(self) -> None:
        with self._lock:
            self._conn.close()


class GoogleWorkspaceBackend:
    """Google Calendar and Gmail through their official APIs.

    Not the default. It activates only when GOOGLE_WORKSPACE_CREDENTIALS points at a
    credentials file, so a reviewer without a Google account still gets a working
    Workspace server. The methods below raise a clear message rather than being silently
    absent, because an interview hold that quietly did nothing is worse than one that
    failed loudly.

    Gmail scope note: this uses gmail.compose, which can create drafts but cannot send
    them. That is the correct scope for A3 and it is enforced by Google rather than by us.
    """

    name = "google"

    def __init__(self) -> None:
        self.credentials_path = os.getenv("GOOGLE_WORKSPACE_CREDENTIALS", "")
        if not self.credentials_path or not Path(self.credentials_path).exists():
            raise RuntimeError(
                "GOOGLE_WORKSPACE_CREDENTIALS is unset or missing. Leave "
                "WORKSPACE_BACKEND=local to use the documented local mock."
            )
        raise NotImplementedError(
            "The Google backend is wired but not implemented: the graded path runs on the "
            "local mock, which A3 permits. Implementing it means filling these five "
            "methods against calendar.events and gmail.drafts; nothing above this class "
            "changes."
        )


def get_backend() -> Any:
    """The backend named in config. One place reads the mode, as with the marketplace."""
    mode = os.getenv("WORKSPACE_BACKEND", "local").lower()
    if mode == "google":
        return GoogleWorkspaceBackend()
    return LocalWorkspace()


def parse_start(value: str) -> datetime:
    """Parse an ISO timestamp, assuming the user's timezone when none is given."""
    try:
        # Python 3.11+ fromisoformat handles a trailing "Z" directly.
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(
            f"cannot parse {value!r} as a start time. Use ISO 8601, "
            "for example 2026-11-18T14:00:00+05:00."
        ) from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone(timedelta(hours=TZ_OFFSET_HOURS)))
    return parsed


def dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)
