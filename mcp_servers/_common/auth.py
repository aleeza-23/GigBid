"""Bearer-token auth and a per-user rate limiter for the MCP HTTP endpoints.

A2 asks for three things that only matter when the server is reachable over HTTP:

* a bearer token on the endpoint, so a stray process on the same machine cannot spend
  Connects;
* a per-user rate limiter that stays *under* GigSim's own limit, so GigBid is the polite
  caller and never the reason a 429 is returned;
* the identity of the caller, which the audit log records.

Tokens map to user ids. In a real deployment that mapping is a database lookup; here it
comes from MCP_TOKENS ("token:user,token:user") so a reviewer can run the stack from
.env alone. One shared dev token is the fallback, and it is refused when APP_ENV=production.
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field


class AuthError(RuntimeError):
    """The caller presented no token, or one we do not know."""


class RateLimited(RuntimeError):
    """The caller is over its budget. retry_after is in seconds."""

    def __init__(self, retry_after: float, limit: int) -> None:
        super().__init__(f"rate limit of {limit}/min exceeded; retry in {retry_after:.1f}s")
        self.retry_after = retry_after
        self.limit = limit


DEV_TOKEN = "mcp-dev-token"


def _token_map() -> dict[str, str]:
    """Parse MCP_TOKENS into {token: user_id}."""
    raw = os.getenv("MCP_TOKENS", "").strip()
    mapping: dict[str, str] = {}
    for pair in raw.split(","):
        if not pair.strip():
            continue
        token, _, user = pair.partition(":")
        if token.strip() and user.strip():
            mapping[token.strip()] = user.strip()
    if not mapping and os.getenv("APP_ENV", "development") != "production":
        # Development convenience only. In production an unset MCP_TOKENS means no
        # caller can authenticate, which is the safe direction to fail in.
        mapping[DEV_TOKEN] = "dev-user"
    return mapping


def resolve_user(authorization: str | None) -> str:
    """Turn an Authorization header into a user id, or raise AuthError.

    Accepts "Bearer <token>" and a bare token, because MCP clients differ in how they
    attach credentials.
    """
    if not authorization:
        raise AuthError("missing Authorization header")
    token = authorization.removeprefix("Bearer ").removeprefix("bearer ").strip()
    if not token:
        raise AuthError("empty bearer token")
    user = _token_map().get(token)
    if user is None:
        raise AuthError("unknown bearer token")
    return user


# ---------------------------------------------------------------- rate limiting


@dataclass
class _Bucket:
    tokens: float
    updated: float


@dataclass
class RateLimiter:
    """Token bucket per user.

    Defaults to 20/min against GigSim's 30/min so GigBid is never the caller that trips
    the marketplace's own limiter. Deliberately a touch conservative: being throttled by
    our own limiter is a queued job, being throttled by theirs is a failed run.
    """

    limit_per_min: int = field(
        default_factory=lambda: int(os.getenv("MCP_RATE_LIMIT_PER_MIN", "20"))
    )
    burst: int | None = None
    _buckets: dict[str, _Bucket] = field(default_factory=dict, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def __post_init__(self) -> None:
        self.capacity = float(self.burst or max(1, self.limit_per_min // 3))
        self.refill_per_sec = self.limit_per_min / 60.0

    def check(self, user_id: str, cost: float = 1.0) -> None:
        """Consume one unit for this user, or raise RateLimited."""
        now = time.monotonic()
        with self._lock:
            bucket = self._buckets.get(user_id)
            if bucket is None:
                bucket = _Bucket(tokens=self.capacity, updated=now)
                self._buckets[user_id] = bucket

            elapsed = now - bucket.updated
            bucket.tokens = min(self.capacity, bucket.tokens + elapsed * self.refill_per_sec)
            bucket.updated = now

            if bucket.tokens < cost:
                deficit = cost - bucket.tokens
                raise RateLimited(deficit / self.refill_per_sec, self.limit_per_min)
            bucket.tokens -= cost

    def reset(self, user_id: str | None = None) -> None:
        """Clear state. Tests use this; nothing in production should need it."""
        with self._lock:
            if user_id is None:
                self._buckets.clear()
            else:
                self._buckets.pop(user_id, None)
