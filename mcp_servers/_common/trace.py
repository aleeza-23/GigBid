"""Trace ids and PII masking, shared by every MCP server.

E3 wants one trace id to flow from the UI request through the API, the graph and every
MCP server into the audit log. The id is carried in a ContextVar rather than threaded
through every function signature, because a tool implementation three frames down should
not have to know it is being traced.

E4 wants emails, phone numbers and payment details masked in logs and traces. Masking
happens here, at the boundary where things get written, rather than at each call site,
so a new tool cannot forget to do it.
"""

from __future__ import annotations

import contextvars
import re
import uuid
from typing import Any

# The current trace id. Set once per inbound request; read by the audit log.
_trace_id: contextvars.ContextVar[str | None] = contextvars.ContextVar("trace_id", default=None)
_user_id: contextvars.ContextVar[str | None] = contextvars.ContextVar("user_id", default=None)


def new_trace_id() -> str:
    return f"tr-{uuid.uuid4().hex[:16]}"


def set_trace_id(trace_id: str | None) -> str:
    """Adopt a caller's trace id, or mint one when the caller did not send it."""
    resolved = trace_id or new_trace_id()
    _trace_id.set(resolved)
    return resolved


def get_trace_id() -> str:
    """The current trace id, minting one if this code path started without a request."""
    current = _trace_id.get()
    if current is None:
        current = new_trace_id()
        _trace_id.set(current)
    return current


def set_user_id(user_id: str | None) -> None:
    _user_id.set(user_id)


def get_user_id() -> str | None:
    return _user_id.get()


# --------------------------------------------------------------------- masking

# Deliberately blunt patterns. A false positive costs a masked log line; a false negative
# leaks a client's phone number into a trace that gets pasted into a bug report.
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_PHONE = re.compile(r"(?<!\d)(?:\+?\d[\d\s().-]{7,}\d)(?!\d)")
_CARD = re.compile(r"(?<!\d)(?:\d[ -]?){13,19}(?!\d)")
_IBAN = re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{10,30}\b")

# Off-platform contact handles. The policy pack forbids them, so the Compliance Guard
# needs to find them, and the audit log should not store them verbatim.
_HANDLE = re.compile(
    r"\b(?:skype|telegram|whatsapp|wechat|discord)\s*[:=]?\s*[\w.@+-]{3,}", re.IGNORECASE
)


def mask_pii(value: Any) -> Any:
    """Recursively mask emails, phones, card numbers, IBANs and chat handles.

    Structure is preserved so a masked payload is still readable as the shape that was
    sent; only the sensitive spans are replaced.
    """
    if isinstance(value, str):
        out = _EMAIL.sub("[email]", value)
        out = _IBAN.sub("[iban]", out)
        out = _CARD.sub("[card]", out)
        out = _PHONE.sub("[phone]", out)
        out = _HANDLE.sub("[handle]", out)
        return out
    if isinstance(value, dict):
        return {k: mask_pii(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [mask_pii(v) for v in value]
    return value


# Argument names whose values are secrets and must never reach a log, masked or not.
_SECRET_KEYS = frozenset(
    {"token", "access_token", "refresh_token", "api_key", "authorization", "password", "secret"}
)


def redact_secrets(payload: dict[str, Any]) -> dict[str, Any]:
    """Drop secret-valued keys entirely. Masking is not enough for a bearer token."""
    return {
        k: ("[redacted]" if k.lower() in _SECRET_KEYS else mask_pii(v)) for k, v in payload.items()
    }
