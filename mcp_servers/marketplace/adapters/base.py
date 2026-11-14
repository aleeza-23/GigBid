"""What GigBid needs from any marketplace, and the errors it may raise.

This module is the bottom of the dependency graph on purpose: it imports nothing from
`agent/` or `mcp_servers/`, so both the adapters and the port above them can depend on it
without a cycle. The direction of dependency is adapters -> base, port -> adapters.

The two error classes matter more than they look. R9 and A6 require a visible degraded
mode, which is only possible if the system can tell "this job does not exist" apart from
"the marketplace is down":

* MarketplaceRejected - the marketplace understood and said no. Retrying is pointless and
  rude. Show the user the reason.
* MarketplaceUnavailable - we never got an answer. The work is still valid, so queue it,
  show the degraded banner, and try later.

Collapsing these into one exception is how a system ends up either retrying a 404 forever
or dropping a job because of a transient 503.
"""

from __future__ import annotations

import uuid
from typing import Any, Protocol, runtime_checkable


class MarketplaceError(RuntimeError):
    """Base class for anything the marketplace refused or could not do."""


class MarketplaceUnavailable(MarketplaceError):
    """The marketplace is unreachable or failing. Queue the work and tell the user."""


class MarketplaceRejected(MarketplaceError):
    """The marketplace understood the request and said no (404, 402, 422)."""

    def __init__(self, status: int, detail: str) -> None:
        super().__init__(f"{status}: {detail}")
        self.status = status
        self.detail = detail


@runtime_checkable
class MarketplacePort(Protocol):
    """The single interface every marketplace adapter implements (R1).

    Nothing above this line knows whether it is talking to GigSim or to live Upwork, which
    is what makes switching marketplaces a config change. Viva question 7 asks what
    changes in the code when you switch: the answer is this file's implementations, and
    nothing else.
    """

    async def search_jobs(self, **filters: Any) -> dict[str, Any]: ...

    async def get_job(self, job_id: str) -> dict[str, Any]: ...

    async def get_client_history(self, client_id: str) -> dict[str, Any]: ...

    async def get_rate_stats(
        self, category: str | None = None, domain: str | None = None
    ) -> dict[str, Any]: ...

    async def get_connects_balance(self, freelancer_id: str) -> dict[str, Any]: ...

    async def create_proposal_draft(self, payload: dict[str, Any]) -> dict[str, Any]:
        """A dry run: what would be sent and what it would cost. Spends nothing."""
        ...

    async def submit_proposal(
        self, payload: dict[str, Any], idempotency_key: str
    ) -> dict[str, Any]: ...

    async def list_threads(self, freelancer_id: str | None = None) -> dict[str, Any]: ...

    async def list_messages(self, thread_id: str) -> dict[str, Any]: ...

    async def send_message(
        self, thread_id: str, text: str, idempotency_key: str, dry_run: bool = False
    ) -> dict[str, Any]: ...

    async def close(self) -> None: ...


def new_idempotency_key(*parts: str) -> str:
    """A key for one logical write. Reuse it on every retry of that same write.

    The caller supplies the stable parts (thread id, job id) and the random suffix makes
    two genuinely different writes distinct. The important discipline is upstream: the key
    is minted once when the user approves and then reused, never regenerated per attempt,
    or idempotency buys nothing.
    """
    return "-".join([*parts, uuid.uuid4().hex[:12]]) if parts else uuid.uuid4().hex
