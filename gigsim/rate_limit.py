"""Per-token rate limiting and synthetic failures.

A real marketplace pushes back. GigSim does the same so that the agent's retry, backoff
and degraded-mode paths get exercised in development rather than during the viva.
"""

from __future__ import annotations

import random
import threading
import time
from collections import defaultdict

from gigsim.config import settings


class TokenBucket:
    """One bucket per caller, refilled continuously at capacity requests per minute."""

    def __init__(self, capacity: int) -> None:
        self.capacity = float(capacity)
        self.rate = capacity / 60.0
        self._tokens: dict[str, float] = defaultdict(lambda: float(capacity))
        self._last: dict[str, float] = defaultdict(time.monotonic)
        self._lock = threading.Lock()

    def take(self, key: str) -> tuple[bool, float]:
        """Consume one token. Returns (allowed, retry_after_seconds)."""
        with self._lock:
            now = time.monotonic()
            elapsed = now - self._last[key]
            self._last[key] = now
            self._tokens[key] = min(self.capacity, self._tokens[key] + elapsed * self.rate)
            if self._tokens[key] >= 1.0:
                self._tokens[key] -= 1.0
                return True, 0.0
            deficit = 1.0 - self._tokens[key]
            return False, round(deficit / self.rate, 2)

    def reset(self) -> None:
        with self._lock:
            self._tokens.clear()
            self._last.clear()


bucket = TokenBucket(settings.rate_limit_per_min)

# Separate stream so error injection does not disturb the client-decision seed.
_rng = random.Random(settings.seed ^ 0x5F5E)


def should_fail() -> bool:
    """True when this request should return a synthetic 5xx."""
    if settings.error_rate <= 0:
        return False
    return _rng.random() < settings.error_rate
