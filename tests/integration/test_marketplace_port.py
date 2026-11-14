"""The marketplace port, exercised against GigSim in-process.

R1 says every marketplace call goes through one interface. These tests run the GigSim
adapter over an ASGI transport, so they prove the port works end to end without starting
a server, and they pin the behaviour the rest of the system relies on: retries on
transient failures, no retries on refusals, and a dry run that costs nothing.
"""

from __future__ import annotations

import httpx
import pytest

from agent.marketplace_port import (
    GigSimAdapter,
    MarketplacePort,
    MarketplaceRejected,
    MarketplaceUnavailable,
    UpworkLiveAdapter,
    get_marketplace,
    new_idempotency_key,
)
from gigsim.config import settings
from gigsim.rate_limit import bucket
from gigsim.server import app
from gigsim.store import store

LETTER = (
    "I shipped a retrieval service handling 40 requests per second for a fintech client "
    "and cut their lookup latency by 60 percent. I can bring the same approach here."
)


@pytest.fixture
def port() -> GigSimAdapter:
    store.reset()
    bucket.reset()
    return GigSimAdapter(
        token=settings.api_token,
        transport=httpx.ASGITransport(app=app),
    )


@pytest.fixture
def job() -> dict:
    return next(j for j in store.jobs.values() if j.get("connects_cost"))


def test_the_adapter_satisfies_the_port(port: GigSimAdapter) -> None:
    assert isinstance(port, MarketplacePort)


@pytest.mark.asyncio
async def test_reads_come_back_shaped_as_expected(port: GigSimAdapter, job: dict) -> None:
    page = await port.search_jobs(limit=3, skills=["Python"])
    assert page["total"] > 0
    assert len(page["jobs"]) <= 3

    one = await port.get_job(job["job_id"])
    assert one["job_id"] == job["job_id"]
    assert "duplicate_of" not in one

    history = await port.get_client_history(job["client_id"])
    assert "hidden_preferences" not in history["client"]

    stats = await port.get_rate_stats(domain="ai_ml")
    assert "benchmarks" in stats

    balance = await port.get_connects_balance("ai_ml_engineer")
    assert balance["balance"] > 0
    await port.close()


@pytest.mark.asyncio
async def test_a_missing_job_is_a_refusal_not_an_outage(port: GigSimAdapter) -> None:
    with pytest.raises(MarketplaceRejected) as caught:
        await port.get_job("no-such-job")
    assert caught.value.status == 404
    await port.close()


@pytest.mark.asyncio
async def test_draft_previews_cost_without_spending(port: GigSimAdapter, job: dict) -> None:
    before = await port.get_connects_balance("ai_ml_engineer")
    preview = await port.create_proposal_draft(
        {
            "freelancer_id": "ai_ml_engineer",
            "job_id": job["job_id"],
            "cover_letter": LETTER,
            "bid_rate": 48.0,
        }
    )
    assert preview["dry_run"] is True
    assert preview["connects_cost"] == job["connects_cost"]

    after = await port.get_connects_balance("ai_ml_engineer")
    assert after["balance"] == before["balance"]
    await port.close()


@pytest.mark.asyncio
async def test_submitting_twice_with_one_key_spends_once(port: GigSimAdapter, job: dict) -> None:
    payload = {
        "freelancer_id": "ai_ml_engineer",
        "job_id": job["job_id"],
        "cover_letter": LETTER,
        "bid_rate": 48.0,
    }
    key = new_idempotency_key("proposal", job["job_id"])
    before = await port.get_connects_balance("ai_ml_engineer")

    first = await port.submit_proposal(payload, key)
    second = await port.submit_proposal(payload, key)

    assert first["proposal_id"] == second["proposal_id"]
    after = await port.get_connects_balance("ai_ml_engineer")
    assert before["balance"] - after["balance"] == job["connects_cost"]
    await port.close()


@pytest.mark.asyncio
async def test_messages_round_trip(port: GigSimAdapter) -> None:
    threads = await port.list_threads(freelancer_id="ai_ml_engineer")
    thread_id = threads["threads"][0]["thread_id"]

    sent = await port.send_message(
        thread_id, "Could we talk on Thursday?", new_idempotency_key("msg")
    )
    assert sent["client_reply"]

    messages = await port.list_messages(thread_id)
    assert messages["messages"][-1]["from"] == "client"
    await port.close()


@pytest.mark.asyncio
async def test_an_outage_retries_then_surfaces_as_unavailable() -> None:
    """Three 503s in a row and the port gives up with a type the UI can act on."""
    attempts = {"n": 0}

    async def always_down(request: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        return httpx.Response(503, json={"error": "upstream_unavailable"})

    port = GigSimAdapter(
        token=settings.api_token,
        max_attempts=3,
        transport=httpx.MockTransport(always_down),
    )
    with pytest.raises(MarketplaceUnavailable):
        await port.get_job("anything")
    assert attempts["n"] == 3
    await port.close()


@pytest.mark.asyncio
async def test_a_rate_limit_is_retried_and_then_succeeds() -> None:
    calls = {"n": 0}

    async def throttled_once(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, json={"error": "rate_limited"}, headers={"Retry-After": "0"})
        return httpx.Response(200, json={"job_id": "ok"})

    port = GigSimAdapter(
        token=settings.api_token,
        max_attempts=3,
        transport=httpx.MockTransport(throttled_once),
    )
    assert (await port.get_job("x"))["job_id"] == "ok"
    assert calls["n"] == 2
    await port.close()


@pytest.mark.asyncio
async def test_a_refusal_is_not_retried() -> None:
    calls = {"n": 0}

    async def refuses(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(402, json={"detail": "insufficient_connects"})

    port = GigSimAdapter(
        token=settings.api_token,
        max_attempts=3,
        transport=httpx.MockTransport(refuses),
    )
    with pytest.raises(MarketplaceRejected):
        await port.submit_proposal({"job_id": "x"}, "key")
    assert calls["n"] == 1, "a refusal should not be retried"
    await port.close()


def test_config_picks_the_adapter() -> None:
    """The mode string is the only thing that decides which marketplace we talk to."""
    assert isinstance(get_marketplace({"marketplace": {"mode": "gigsim"}}), GigSimAdapter)
    assert isinstance(
        get_marketplace({"marketplace": {"mode": "upwork_live"}}), UpworkLiveAdapter
    )

    with pytest.raises(ValueError):
        get_marketplace({"marketplace": {"mode": "ebay"}})


def test_both_adapters_satisfy_one_interface() -> None:
    """The graph must not be able to tell the two apart (R1, viva question 7).

    Checked structurally rather than by duck-typing one call at a time: every method on
    the Protocol has to exist on both adapters with the same name, or a node written
    against GigSim breaks the day live mode is switched on.
    """
    for adapter in (GigSimAdapter, UpworkLiveAdapter):
        for method in (
            "search_jobs",
            "get_job",
            "get_client_history",
            "get_rate_stats",
            "get_connects_balance",
            "create_proposal_draft",
            "submit_proposal",
            "list_threads",
            "list_messages",
            "send_message",
            "close",
        ):
            assert callable(getattr(adapter, method, None)), f"{adapter.__name__} lacks {method}"


def test_building_the_live_adapter_touches_no_network() -> None:
    """Construction is lazy, so importing or instantiating it in CI is safe.

    The session opens on first use. This is what lets the factory above return a live
    adapter in a test that has no Upwork credentials and no network.
    """
    adapter = UpworkLiveAdapter(user_id="test-user")
    assert adapter._session is None
    assert adapter.url.startswith("https://")
    assert adapter.min_interval > 0, "live reads must be paced (R9)"


@pytest.mark.asyncio
async def test_live_mode_cannot_submit_without_a_token() -> None:
    """No stored token is a refusal, not a crash, and certainly not a silent send."""
    adapter = UpworkLiveAdapter(user_id="nobody-connected")
    with pytest.raises((MarketplaceRejected, MarketplaceUnavailable)):
        await adapter.submit_proposal({"job_id": "j1", "cover_letter": LETTER}, "key-1")
    await adapter.close()


@pytest.mark.asyncio
async def test_live_writes_are_drafts_that_need_confirmation() -> None:
    """A4: live mode never completes a submission, it only drafts one.

    The dry-run preview is the screen the user approves from, so it must say plainly that
    confirmation still happens on Upwork, and it must not invent a Connects cost that
    the Upwork API does not quote.
    """
    adapter = UpworkLiveAdapter(user_id="test-user")
    preview = await adapter.create_proposal_draft({"job_id": "j1", "cover_letter": LETTER})
    assert preview["dry_run"] is True
    assert preview["connects_cost"] is None, "do not fabricate a cost Upwork does not give"
    assert any("confirm" in w.lower() for w in preview["warnings"])
    await adapter.close()
