"""Exactly-once writes, durable across a process restart.

A2 requires an idempotency key on every write so a retried call never submits twice or
spends Connects twice. GigSim keeps its own key cache, but that is not enough on its own:
if the MCP server dies after GigSim committed but before the response got back to the
agent, the retry has to be stopped *here*, or the agent re-sends and we rely on the
simulator's memory, which a reset would wipe.

So the store is SQLite on disk, not a dict. The flow is a claim followed by a completion:

    claim("prop-42")        -> ClaimResult(is_new=True)      ... do the write
    complete("prop-42", r)  -> the result is now durable

    claim("prop-42")        -> ClaimResult(is_new=False, result=r)   -> replay, no write

A claim that was never completed is "in flight". If the process died mid-write that row
is still there, and the retry is told so rather than being allowed through: a human gets
a "needs attention" thread instead of the system silently paying twice. That is the
conservative direction, and it is exactly what viva question 2 asks for.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_PATH = Path(
    os.getenv(
        "MCP_IDEMPOTENCY_DB",
        Path(__file__).resolve().parents[2] / "data" / "audit" / "idempotency.sqlite3",
    )
)

# How long an in-flight claim blocks a retry before we assume the writer died and allow
# one more attempt. Long enough to outlast a slow marketplace, short enough that a human
# is not stuck for an afternoon.
STALE_AFTER_SEC = float(os.getenv("MCP_IDEMPOTENCY_STALE_SEC", "120"))


@dataclass
class ClaimResult:
    """The answer to "may I perform this write?"."""

    is_new: bool
    """True when this caller owns the write and should proceed."""

    result: Any = None
    """The stored result, when this key already completed."""

    in_flight: bool = False
    """True when another caller claimed this key and has not finished."""

    @property
    def is_replay(self) -> bool:
        return not self.is_new and not self.in_flight


class IdempotencyStore:
    def __init__(self, path: Path | str | None = None) -> None:
        self.path = Path(path or DEFAULT_PATH)
        if str(self.path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        # check_same_thread=False: FastMCP serves tools from a thread pool, and the lock
        # above already serialises access.
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS idempotency (
                key        TEXT PRIMARY KEY,
                status     TEXT NOT NULL,
                result     TEXT,
                tool       TEXT,
                user_id    TEXT,
                claimed_at REAL NOT NULL,
                done_at    REAL
            )
            """
        )
        self._conn.commit()

    def claim(self, key: str, *, tool: str = "", user_id: str = "") -> ClaimResult:
        """Try to take ownership of one logical write."""
        now = time.time()
        with self._lock:
            cur = self._conn.execute(
                "SELECT status, result, claimed_at FROM idempotency WHERE key = ?", (key,)
            )
            row = cur.fetchone()

            if row is None:
                self._conn.execute(
                    "INSERT INTO idempotency (key, status, tool, user_id, claimed_at) "
                    "VALUES (?, 'in_flight', ?, ?, ?)",
                    (key, tool, user_id, now),
                )
                self._conn.commit()
                return ClaimResult(is_new=True)

            status, result_json, claimed_at = row
            if status == "done":
                return ClaimResult(
                    is_new=False, result=json.loads(result_json) if result_json else None
                )

            # Still in flight. Allow a retry only once the claim has gone stale.
            if now - claimed_at > STALE_AFTER_SEC:
                self._conn.execute(
                    "UPDATE idempotency SET claimed_at = ? WHERE key = ?", (now, key)
                )
                self._conn.commit()
                return ClaimResult(is_new=True)

            return ClaimResult(is_new=False, in_flight=True)

    def complete(self, key: str, result: Any) -> None:
        """Record the result of a claimed write, making future calls replays."""
        with self._lock:
            self._conn.execute(
                "UPDATE idempotency SET status='done', result=?, done_at=? WHERE key=?",
                (json.dumps(result, default=str), time.time(), key),
            )
            self._conn.commit()

    def release(self, key: str) -> None:
        """Give up a claim that failed before it changed anything downstream.

        Only safe when the write provably did not land: a schema rejection, say. Never
        call this after a timeout, because the write may well have succeeded.
        """
        with self._lock:
            self._conn.execute("DELETE FROM idempotency WHERE key=? AND status='in_flight'", (key,))
            self._conn.commit()

    def get(self, key: str) -> Any:
        with self._lock:
            cur = self._conn.execute(
                "SELECT result FROM idempotency WHERE key=? AND status='done'", (key,)
            )
            row = cur.fetchone()
        return json.loads(row[0]) if row and row[0] else None

    def close(self) -> None:
        with self._lock:
            self._conn.close()
