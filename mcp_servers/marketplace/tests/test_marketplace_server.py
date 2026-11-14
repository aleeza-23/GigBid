"""The Marketplace server through an in-memory MCP client (A7).

Run against GigSim in-process over an ASGI transport: no sockets, no server to start, and
the simulator's real validation and Connects accounting. These tests cover the four
promises A2 makes that are easy to claim and hard to keep, so each one is written to fail
loudly if the promise breaks:

* a dry run costs nothing;
* one idempotency key spends once, however many times it is called;
* a client's hidden preferences and scam flag never cross the tool boundary;
* an unreachable marketplace is distinguishable from a refusal.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
from mcp.shared.memory import create_connected_server_and_client_session

from gigsim.config import settings
from gigsim.rate_limit import bucket
from gigsim.server import app
from gigsim.store import store
from mcp_servers._common.idempotency import IdempotencyStore
from mcp_servers.marketplace import server as ms
from mcp_servers.marketplace.adapters.gigsim import GigSimAdapter

LETTER = (
    "I built a support ticket triage classifier that lifted routing accuracy from 71 to "
    "89 percent for a SaaS client, and the same approach fits this problem closely."
)


@pytest.fixture(autouse=True)
def _wire_gigsim(tmp_path, monkeypatch):
    """Point the server at an in-process GigSim and a throwaway idempotency store.

    The store is per-test on purpose: a shared one would let an earlier test's key make a
    later submission look like a replay, which is the kind of false pass that hides a real
    double-spend bug.
    """
    store.reset()
    bucket.reset()
    monkeypatch.setattr(settings, "error_rate", 0.0)
    monkeypatch.setattr(settings, "rate_limit_per_min", 10_000)

    ms.idempotency = IdempotencyStore(tmp_path / "idem.sqlite3")
    adapter = GigSimAdapter(
        base_url="http://sim", token=settings.api_token, transport=httpx.ASGITransport(app=app)
    )
    ms._port = adapter
    yield
    ms._port = None


async def _session():
    return create_connected_server_and_client_session(ms.mcp._mcp_server)


def _payload(result: Any) -> dict[str, Any]:
    if getattr(result, "structuredContent", None):
        return result.structuredContent
    return json.loads(result.content[0].text)


def _a_job() -> dict[str, Any]:
    return next(iter(store.jobs.values()))


def _a_freelancer() -> str:
    return next(iter(store.connects))


def _proposal(job_id: str, freelancer_id: str, **over: Any) -> dict[str, Any]:
    return {
        "freelancer_id": freelancer_id,
        "job_id": job_id,
        "cover_letter": LETTER,
        "bid_rate": 45.0,
        **over,
    }


# --------------------------------------------------------------------- registration


@pytest.mark.asyncio
async def test_all_required_tools_are_registered() -> None:
    async with await _session() as session:
        names = {t.name for t in (await session.list_tools()).tools}

    # A2's read list.
    for name in (
        "search_jobs",
        "get_job",
        "get_client_history",
        "get_connects_balance",
        "list_messages",
    ):
        assert name in names, f"A2 requires the read tool {name}"

    # A2's write list.
    for name in ("create_proposal_draft", "submit_proposal", "send_message"):
        assert name in names, f"A2 requires the write tool {name}"


@pytest.mark.asyncio
async def test_annotations_distinguish_reads_from_spends() -> None:
    """The nuance A2 is really asking about.

    `submit_proposal` is destructive (Connects are gone and not refunded) *and* idempotent
    (the key stops a repeat spending again). Those are separate questions, and collapsing
    them is how a host ends up either not confirming a spend or confirming a harmless
    preview.
    """
    async with await _session() as session:
        tools = {t.name: t for t in (await session.list_tools()).tools}

    for name in ("search_jobs", "get_job", "get_client_history", "get_connects_balance"):
        ann = tools[name].annotations
        assert ann.readOnlyHint is True, f"{name} is a read"
        assert ann.destructiveHint is False

    # A dry run touches nothing, so a host must not gate it behind a confirmation.
    preview = tools["create_proposal_draft"].annotations
    assert preview.readOnlyHint is True, "a dry run spends nothing and must read as a read"

    submit = tools["submit_proposal"].annotations
    assert submit.readOnlyHint is False
    assert submit.destructiveHint is True, "spending Connects is irreversible"
    assert submit.idempotentHint is True, "the idempotency key makes a repeat a no-op"

    # The marketplace is external, so nothing here is cacheable.
    for tool in tools.values():
        assert tool.annotations.openWorldHint is True


# -------------------------------------------------------------------------- schema


@pytest.mark.asyncio
async def test_the_schema_refuses_a_malformed_proposal() -> None:
    """Refused at the tool boundary, before it costs a request against the rate budget."""
    job, freelancer = _a_job(), _a_freelancer()
    async with await _session() as session:
        # Under GigSim's 50-character floor.
        result = await session.call_tool(
            "submit_proposal",
            {
                "proposal": _proposal(job["job_id"], freelancer, cover_letter="too short"),
                "idempotency_key": "key-aaaaaaaa",
            },
        )
        assert result.isError

        # A bid of zero.
        result = await session.call_tool(
            "submit_proposal",
            {
                "proposal": _proposal(job["job_id"], freelancer, bid_rate=0),
                "idempotency_key": "key-bbbbbbbb",
            },
        )
        assert result.isError

        # A key too short to be meaningful.
        result = await session.call_tool(
            "submit_proposal",
            {"proposal": _proposal(job["job_id"], freelancer), "idempotency_key": "abc"},
        )
        assert result.isError


@pytest.mark.asyncio
async def test_search_limit_is_capped_by_the_schema() -> None:
    async with await _session() as session:
        assert (await session.call_tool("search_jobs", {"limit": 5000})).isError


# --------------------------------------------------------------------- information


@pytest.mark.asyncio
async def test_client_history_never_leaks_hidden_preferences() -> None:
    """The boundary the whole evaluation rests on.

    If the scam flag or the client's hidden preferences reached the agent, scam precision
    and reply rate would both measure a system reading the answer key.
    """
    scam_id = next(
        (cid for cid, c in store.clients.items() if c.get("is_scam_account")),
        next(iter(store.clients)),
    )
    async with await _session() as session:
        data = _payload(await session.call_tool("get_client_history", {"client_id": scam_id}))

    blob = json.dumps(data)
    assert "hidden_preferences" not in blob
    assert "is_scam_account" not in blob
    # The inferable signals must still be there, or the Risk Assessor has nothing to work with.
    assert "hire_rate" in data
    assert "payment_verified" in data


@pytest.mark.asyncio
async def test_client_history_computes_the_ratios_the_model_should_not() -> None:
    client_id = next(
        cid for cid, c in store.clients.items() if (c.get("jobs_posted") or 0) > 0
    )
    async with await _session() as session:
        data = _payload(await session.call_tool("get_client_history", {"client_id": client_id}))

    if data["jobs_posted"]:
        assert data["hire_rate"] == pytest.approx(data["hires"] / data["jobs_posted"], abs=1e-3)


@pytest.mark.asyncio
async def test_rate_stats_flatten_into_usable_percentiles() -> None:
    async with await _session() as session:
        data = _payload(await session.call_tool("get_rate_stats", {"domain": "ai_ml"}))

    assert data["available"] is True
    earned = data["earned_hourly_usd"]
    assert earned["p50"] > 0
    assert earned["n"] > 100, "a percentile needs a sample size to be worth anything"


# -------------------------------------------------------------------------- writes


@pytest.mark.asyncio
async def test_a_dry_run_previews_the_cost_and_spends_nothing() -> None:
    job, freelancer = _a_job(), _a_freelancer()
    before = store.connects_balance(freelancer)["balance"]

    async with await _session() as session:
        data = _payload(
            await session.call_tool(
                "create_proposal_draft", {"proposal": _proposal(job["job_id"], freelancer)}
            )
        )

    assert data["dry_run"] is True
    assert data["connects_cost"] == job["connects_cost"]
    assert data["would_send"]["cover_letter"] == LETTER
    assert store.connects_balance(freelancer)["balance"] == before, "a preview must cost nothing"


@pytest.mark.asyncio
async def test_one_key_spends_once_however_often_it_is_called() -> None:
    """A2's idempotency promise, and viva question 2.

    Three calls with one key: one spend, one proposal, two replays.
    """
    job, freelancer = _a_job(), _a_freelancer()
    cost = job["connects_cost"]
    before = store.connects_balance(freelancer)["balance"]
    key = "approval-" + job["job_id"][:16]

    async with await _session() as session:
        first = _payload(
            await session.call_tool(
                "submit_proposal",
                {"proposal": _proposal(job["job_id"], freelancer), "idempotency_key": key},
            )
        )
        second = _payload(
            await session.call_tool(
                "submit_proposal",
                {"proposal": _proposal(job["job_id"], freelancer), "idempotency_key": key},
            )
        )
        third = _payload(
            await session.call_tool(
                "submit_proposal",
                {"proposal": _proposal(job["job_id"], freelancer), "idempotency_key": key},
            )
        )

    assert first["idempotent_replay"] is False
    assert second["idempotent_replay"] is True
    assert third["idempotent_replay"] is True
    assert first["proposal_id"] == second["proposal_id"] == third["proposal_id"]

    spent = before - store.connects_balance(freelancer)["balance"]
    assert spent == cost, f"expected one spend of {cost}, got {spent}"
    assert len([p for p in store.proposals.values() if p["job_id"] == job["job_id"]]) == 1


@pytest.mark.asyncio
async def test_a_claim_left_in_flight_blocks_a_retry() -> None:
    """A crash between claim and completion must not let the retry spend again.

    This is the half of exactly-once that a naive "cache the result" approach misses: at
    the moment of the crash there is no result to cache, so the key has to be claimed
    *before* the write, not after.
    """
    job, freelancer = _a_job(), _a_freelancer()
    key = "in-flight-key-123"
    ms.idempotency.claim(key, tool="submit_proposal", user_id="tester")

    async with await _session() as session:
        result = await session.call_tool(
            "submit_proposal",
            {"proposal": _proposal(job["job_id"], freelancer), "idempotency_key": key},
        )

    assert result.isError
    assert "in flight" in result.content[0].text.lower()


@pytest.mark.asyncio
async def test_a_refusal_frees_the_key_so_a_fixed_payload_can_retry() -> None:
    """A rejected write changed nothing, so the key must not be burned."""
    freelancer = _a_freelancer()
    key = "retry-after-fix-1"

    async with await _session() as session:
        bad = await session.call_tool(
            "submit_proposal",
            {"proposal": _proposal("no-such-job", freelancer), "idempotency_key": key},
        )
        assert bad.isError

        job = _a_job()
        good = _payload(
            await session.call_tool(
                "submit_proposal",
                {"proposal": _proposal(job["job_id"], freelancer), "idempotency_key": key},
            )
        )

    assert good["idempotent_replay"] is False, "the key should have been released"
    assert good["proposal_id"]


@pytest.mark.asyncio
async def test_sending_a_message_needs_a_key_but_previewing_does_not() -> None:
    thread_id = next(iter(store.threads))
    async with await _session() as session:
        preview = _payload(
            await session.call_tool(
                "send_message",
                {"thread_id": thread_id, "text": "Could we talk Tuesday?", "dry_run": True},
            )
        )
        assert preview["dry_run"] is True

        missing_key = await session.call_tool(
            "send_message", {"thread_id": thread_id, "text": "Sending for real"}
        )
        assert missing_key.isError

        sent = _payload(
            await session.call_tool(
                "send_message",
                {
                    "thread_id": thread_id,
                    "text": "Could we talk Tuesday?",
                    "idempotency_key": "msg-key-00001",
                },
            )
        )
    assert sent["idempotent_replay"] is False
    assert sent["client_reply"], "GigSim's client should answer"


# ------------------------------------------------------------------ degraded mode


@pytest.mark.asyncio
async def test_an_unreachable_marketplace_is_not_a_refusal() -> None:
    """A6: the UI must be able to tell "down" from "no".

    The error text carries a marker the UI matches on, so a 503 shows the degraded banner
    and queues the work, while a 404 shows the user a reason.
    """

    def always_down(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"detail": "simulated outage"})

    ms._port = GigSimAdapter(
        base_url="http://sim",
        token=settings.api_token,
        transport=httpx.MockTransport(always_down),
        max_attempts=1,
    )

    async with await _session() as session:
        health = _payload(await session.call_tool("marketplace_health", {}))
        assert health["status"] == "unavailable"

        result = await session.call_tool("get_job", {"job_id": "upw-1"})
        assert result.isError
        assert "MARKETPLACE_UNAVAILABLE" in result.content[0].text


@pytest.mark.asyncio
async def test_health_is_ok_when_the_marketplace_answers() -> None:
    async with await _session() as session:
        data = _payload(await session.call_tool("marketplace_health", {}))
    assert data["status"] == "ok"
    assert data["marketplace"] == "gigsim"
    assert data["trace_id"]


# ------------------------------------------------------------------------- audit


@pytest.mark.asyncio
async def test_writes_are_recorded_in_the_audit_log(tmp_path) -> None:
    """A2's append-only log: who spent what, under which trace id."""
    from mcp_servers._common.audit import AuditLog

    ms.audit = AuditLog(tmp_path / "audit.jsonl", server="test-marketplace")
    job, freelancer = _a_job(), _a_freelancer()

    async with await _session() as session:
        await session.call_tool(
            "submit_proposal",
            {
                "proposal": _proposal(job["job_id"], freelancer),
                "idempotency_key": "audited-key-1",
            },
        )

    entries = ms.audit.read_all()
    submits = [e for e in entries if e["tool"] == "submit_proposal"]
    assert submits, "a Connects spend must be recorded"
    entry = submits[-1]
    assert entry["trace_id"]
    assert entry["arguments"]["idempotency_key"] == "audited-key-1"
    assert entry["status"] == "ok"
