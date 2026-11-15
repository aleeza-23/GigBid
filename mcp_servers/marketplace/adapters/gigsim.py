"""GigSim adapter: the marketplace port spoken over GigSim's REST API.

Retries, backoff and the rejected/unavailable distinction live here rather than in the
nodes, so every caller inherits the same manners and the UI has one exception type to show
a degraded-mode banner for.

What is retried is a deliberate choice. 429 and 5xx mean "ask again later", so they get
bounded backoff with jitter. Every other 4xx means the request itself was wrong, and
retrying a malformed proposal just wastes GigSim's rate budget on a request that cannot
succeed.
"""

from __future__ import annotations

import asyncio
import os
import random
from typing import Any

import httpx

from mcp_servers.marketplace.adapters.base import MarketplaceRejected, MarketplaceUnavailable

# Status codes worth asking about again.
_TRANSIENT = frozenset({429, 500, 502, 503, 504})


def _detail(response: httpx.Response) -> str:
    """A short, readable reason from a JSON or text error body."""
    try:
        body = response.json()
    except ValueError:
        return response.text[:300]
    if isinstance(body, dict):
        return str(body.get("detail") or body.get("error") or body)[:300]
    return str(body)[:300]


class GigSimAdapter:
    """Talks to the GigSim simulator over HTTP."""

    def __init__(
        self,
        base_url: str | None = None,
        token: str | None = None,
        timeout: float | None = None,
        max_attempts: int | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.base_url = (base_url or os.getenv("GIGSIM_BASE_URL", "http://localhost:8081")).rstrip(
            "/"
        )
        self.token = token or os.getenv("GIGSIM_API_TOKEN", "gigsim-dev-token")
        # Three attempts over about two seconds is right for a marketplace that is up and
        # occasionally flaky, which is what the defaults assume. It is wrong for a free
        # hosting plan that suspends an idle service: there, the first request after a
        # quiet spell waits twenty seconds or more while the instance starts, and the
        # router answers 502 until it does. The API then reported the marketplace as
        # unavailable while the simulator was merely waking up.
        #
        # So both are environment-tunable, and a deployment on a plan that sleeps sets
        # them generously. Defaults are unchanged, so local runs and the test suite are
        # not slowed down by somebody else's hosting plan.
        self.max_attempts = max_attempts or int(os.getenv("GIGSIM_MAX_ATTEMPTS", "3"))
        timeout = timeout if timeout is not None else float(os.getenv("GIGSIM_TIMEOUT", "15"))
        self.max_backoff = float(os.getenv("GIGSIM_MAX_BACKOFF", "8"))
        # transport is for tests: pass an ASGI transport to exercise the port against the
        # simulator in-process, with no server and no sockets.
        self._client = httpx.AsyncClient(
            base_url=f"{self.base_url}/api/v1",
            headers={"Authorization": f"Bearer {self.token}"},
            timeout=timeout,
            transport=transport,
        )

    # ------------------------------------------------------------------ plumbing

    async def _request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        """One call, with bounded retries on the failures that are worth retrying."""
        last: Exception | None = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                response = await self._client.request(method, path, **kwargs)
            except (httpx.ConnectError, httpx.ReadTimeout, httpx.WriteTimeout) as exc:
                last = exc
            else:
                if response.status_code < 400:
                    return response.json()
                if response.status_code in _TRANSIENT:
                    retry_after = float(response.headers.get("Retry-After", 0) or 0)
                    last = MarketplaceUnavailable(f"{response.status_code} from marketplace")
                    if attempt < self.max_attempts:
                        await asyncio.sleep(
                            retry_after
                            or min(self.max_backoff, 0.5 * 2**attempt) + random.random() * 0.25
                        )
                    continue
                raise MarketplaceRejected(response.status_code, _detail(response))

            if attempt < self.max_attempts:
                await asyncio.sleep(min(self.max_backoff, 0.5 * 2**attempt) + random.random() * 0.25)

        # The base URL belongs in this message. "All connection attempts failed" with no
        # address is indistinguishable between a sleeping simulator, a wrong host and an
        # unset GIGSIM_BASE_URL falling back to localhost, and on a deployment the third
        # is by far the most likely. Naming the address turns a guess into a fix.
        raise MarketplaceUnavailable(
            f"{method} {self.base_url}{path} failed after {self.max_attempts} "
            f"attempts: {last}"
        )

    # --------------------------------------------------------------------- reads

    async def search_jobs(self, **filters: Any) -> dict[str, Any]:
        params = {k: v for k, v in filters.items() if v is not None}
        return await self._request("GET", "/jobs", params=params)

    async def get_job(self, job_id: str) -> dict[str, Any]:
        return await self._request("GET", f"/jobs/{job_id}")

    async def get_client_history(self, client_id: str) -> dict[str, Any]:
        return await self._request("GET", f"/clients/{client_id}/history")

    async def get_rate_stats(
        self, category: str | None = None, domain: str | None = None
    ) -> dict[str, Any]:
        params = {k: v for k, v in {"category": category, "domain": domain}.items() if v}
        return await self._request("GET", "/stats/rates", params=params)

    async def get_connects_balance(self, freelancer_id: str) -> dict[str, Any]:
        return await self._request("GET", f"/freelancers/{freelancer_id}/connects")

    async def list_threads(self, freelancer_id: str | None = None) -> dict[str, Any]:
        params = {"freelancer_id": freelancer_id} if freelancer_id else {}
        return await self._request("GET", "/threads", params=params)

    async def list_messages(self, thread_id: str) -> dict[str, Any]:
        return await self._request("GET", f"/threads/{thread_id}/messages")

    # -------------------------------------------------------------------- writes

    async def create_proposal_draft(self, payload: dict[str, Any]) -> dict[str, Any]:
        return await self._request("POST", "/proposals", json={**payload, "dry_run": True})

    async def submit_proposal(
        self, payload: dict[str, Any], idempotency_key: str
    ) -> dict[str, Any]:
        return await self._request(
            "POST",
            "/proposals",
            json={**payload, "dry_run": False},
            headers={"Idempotency-Key": idempotency_key},
        )

    async def send_message(
        self, thread_id: str, text: str, idempotency_key: str, dry_run: bool = False
    ) -> dict[str, Any]:
        return await self._request(
            "POST",
            "/messages",
            json={"thread_id": thread_id, "text": text, "dry_run": dry_run},
            headers={"Idempotency-Key": idempotency_key},
        )

    async def close(self) -> None:
        await self._client.aclose()
