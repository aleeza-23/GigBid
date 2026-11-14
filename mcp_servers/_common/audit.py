"""Append-only audit log for every MCP tool call.

A2 requires a record of user, tool, arguments, result and trace id. "Append-only" is the
interesting part: the log is opened in append mode and written as JSON Lines, one call per
line, and nothing in this module can update or delete a line. If a tool spends Connects,
the evidence that it happened survives a crash and cannot be quietly rewritten.

Arguments and results pass through mask_pii and redact_secrets on the way in, so the log
is safe to read aloud during a viva.
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any

from mcp_servers._common.trace import get_trace_id, get_user_id, mask_pii, redact_secrets

DEFAULT_PATH = Path(
    os.getenv("MCP_AUDIT_LOG", Path(__file__).resolve().parents[2] / "data" / "audit" / "mcp.jsonl")
)

_lock = threading.Lock()


def _truncate(value: Any, limit: int = 2000) -> Any:
    """Keep the log readable. A 5,000-character cover letter is not useful here."""
    if isinstance(value, str) and len(value) > limit:
        return value[:limit] + f"...[+{len(value) - limit} chars]"
    if isinstance(value, dict):
        return {k: _truncate(v, limit) for k, v in value.items()}
    if isinstance(value, list):
        return [_truncate(v, limit) for v in value[:20]]
    return value


class AuditLog:
    """Append-only JSON Lines sink. One instance per server process."""

    def __init__(self, path: Path | None = None, server: str = "unknown") -> None:
        self.path = Path(path or DEFAULT_PATH)
        self.server = server
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(
        self,
        tool: str,
        arguments: dict[str, Any] | None = None,
        result: Any = None,
        *,
        status: str = "ok",
        error: str | None = None,
        user_id: str | None = None,
        duration_ms: float | None = None,
    ) -> dict[str, Any]:
        """Append one entry and return it, so a caller can show it in the UI."""
        entry = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "trace_id": get_trace_id(),
            "user_id": user_id or get_user_id() or "anonymous",
            "server": self.server,
            "tool": tool,
            "status": status,
            "arguments": _truncate(redact_secrets(arguments or {})),
            "result": _truncate(mask_pii(result)) if result is not None else None,
            "error": error,
            "duration_ms": round(duration_ms, 2) if duration_ms is not None else None,
        }
        line = json.dumps(entry, ensure_ascii=False, default=str)
        # "a" means the OS appends; two processes writing at once interleave whole lines
        # rather than corrupting each other. The lock covers threads in this process.
        with _lock, self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
        return entry

    def read_all(self) -> list[dict[str, Any]]:
        """Read the log back. Used by tests and the settings screen, never to mutate."""
        if not self.path.exists():
            return []
        with self.path.open(encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
