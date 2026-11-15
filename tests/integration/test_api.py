"""The API end to end: sign in, inbox, run, stream, approve (D).

Driven through ASGI rather than a live server, against GigSim in-process, with the model
forced offline. What is being tested is the API contract and the approval flow, not a
model's prose.

The fixture drives the **lifespan** explicitly. `httpx.ASGITransport` does not, so without
it `app.state.checkpointer` is never built and every graph endpoint fails with a confusing
`AttributeError` — which looks like a broken app rather than a test harness that skipped
startup. Uvicorn runs the lifespan properly in production; the test has to do it by hand.
"""

from __future__ import annotations

import os
from typing import Any

import httpx
import pytest
import pytest_asyncio

os.environ["GIGBID_FORCE_OFFLINE"] = "1"

from gigsim.config import settings
from gigsim.rate_limit import bucket
from gigsim.server import app as gigsim_app
from gigsim.store import store
from mcp_servers._common.idempotency import IdempotencyStore
from mcp_servers.marketplace import server as market
from mcp_servers.marketplace.adapters.gigsim import GigSimAdapter

PERSONA = "ai_ml_engineer"


@pytest_asyncio.fixture
async def client(tmp_path, monkeypatch):
    """An API client with the lifespan driven, GigSim in-process, model offline."""
    store.reset()
    bucket.reset()
    monkeypatch.setattr(settings, "error_rate", 0.0)
    monkeypatch.setattr(settings, "rate_limit_per_min", 10_000)
    monkeypatch.setenv("GIGBID_FORCE_OFFLINE", "1")
    # A durable checkpointer, so the pending-approval-survives-restart test is meaningful.
    monkeypatch.setenv("CHECKPOINTER", "sqlite")
    monkeypatch.setenv("CHECKPOINT_DB", str(tmp_path / "checkpoints.sqlite3"))
    monkeypatch.setenv("APP_SECRET_KEY", "test-secret-not-a-real-key")

    market.idempotency = IdempotencyStore(tmp_path / "idem.sqlite3")
    market._port = GigSimAdapter(
        base_url="http://sim",
        token=settings.api_token,
        transport=httpx.ASGITransport(app=gigsim_app),
    )

    import agent.llm as llm_module

    llm_module._default = None

    from apps.api.main import app

    async with app.router.lifespan_context(app):
        # The lifespan builds its own marketplace port from config; point it back at the
        # in-process simulator so the test never opens a socket.
        app.state.marketplace = market._port
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://api", timeout=120
        ) as http_client:
            yield http_client

    market._port = None
    llm_module._default = None


async def _login(client: httpx.AsyncClient) -> dict[str, Any]:
    response = await client.post(
        "/auth/login", json={"username": "reviewer", "password": "reviewer"}
    )
    assert response.status_code == 200, response.text
    return response.json()


def _viable_job(jobs: list[dict[str, Any]]) -> dict[str, Any]:
    """A job the graph will bid on, not merely one with matching skills."""
    from mcp_servers.portfolio import corpus

    profile = corpus.load_profile(PERSONA)
    rate_floor = float(profile.get("hourly_rate_usd") or 0)
    project_floor = float(profile.get("minimum_project_budget_usd") or 0)

    for job in jobs:
        client_row = store.clients.get(job.get("client_id") or "", {})
        if client_row.get("is_scam_account"):
            continue
        if job.get("is_hourly") and float(job.get("hourly_high") or 0) >= rate_floor:
            return job
        if not job.get("is_hourly") and float(job.get("budget") or 0) >= project_floor:
            return job
    pytest.skip("no economically viable job in this page")


# ------------------------------------------------------------------------ meta


@pytest.mark.asyncio
async def test_health_reports_every_subsystem(client: httpx.AsyncClient) -> None:
    body = (await client.get("/health")).json()

    assert body["status"] in ("ok", "degraded")
    # Each of these is a claim that is easy to assume and embarrassing to get wrong.
    assert body["persistence"]["checkpointer"] == "sqlite"
    assert body["persistence"]["survives_restart"] is True
    assert "mode" in body["marketplace"]
    assert body["models"]["offline"] is True
    assert "embeddings_backend" in body["retrieval"]
    # Offline is a degradation and must be declared, not hidden.
    assert any("offline" in note for note in body["degraded"])


@pytest.mark.asyncio
async def test_the_openapi_schema_is_complete_enough_to_generate_a_client(
    client: httpx.AsyncClient,
) -> None:
    """Part D requires the TypeScript client to be generated from this schema."""
    spec = (await client.get("/openapi.json")).json()
    paths = spec["paths"]

    for path in (
        "/auth/login",
        "/jobs",
        "/jobs/{job_id}",
        "/runs",
        "/runs/{tid}/stream",
        "/approvals",
        "/approvals/{tid}",
        "/messages",
        "/pipeline",
        "/analytics",
        "/memory",
        "/settings",
    ):
        assert path in paths, f"{path} is missing from the OpenAPI schema"

    # Response models, not bare dicts: a generated client needs named types.
    assert spec["components"]["schemas"].get("JobPage")
    assert spec["components"]["schemas"].get("PendingApproval")


# ------------------------------------------------------------------------ auth


@pytest.mark.asyncio
async def test_data_routes_require_a_session(client: httpx.AsyncClient) -> None:
    for path in ("/jobs", "/approvals", "/pipeline", "/analytics", "/memory", "/settings"):
        response = await client.get(path)
        assert response.status_code == 401, f"{path} served an anonymous request"


@pytest.mark.asyncio
async def test_a_wrong_password_does_not_say_which_part_was_wrong(
    client: httpx.AsyncClient,
) -> None:
    """The response must not confirm which accounts exist."""
    bad_user = await client.post(
        "/auth/login", json={"username": "nobody", "password": "x"}
    )
    bad_password = await client.post(
        "/auth/login", json={"username": "reviewer", "password": "wrong"}
    )
    assert bad_user.status_code == bad_password.status_code == 401
    assert bad_user.json()["detail"] == bad_password.json()["detail"]


@pytest.mark.asyncio
async def test_login_sets_an_httponly_cookie(client: httpx.AsyncClient) -> None:
    response = await client.post(
        "/auth/login", json={"username": "reviewer", "password": "reviewer"}
    )
    cookie = response.headers.get("set-cookie", "")
    # The token can spend the user's Connects, so JavaScript must not be able to read it.
    assert "httponly" in cookie.lower()

    me = (await client.get("/auth/me")).json()
    assert me["user_id"] == "reviewer"


# ------------------------------------------------------------------------ jobs


@pytest.mark.asyncio
async def test_the_inbox_ranks_and_distinguishes_unanalysed_jobs(
    client: httpx.AsyncClient,
) -> None:
    await _login(client)
    page = (await client.get("/jobs", params={"limit": 10})).json()

    assert page["total"] > 0
    assert page["jobs"]
    # Nothing has been analysed yet, so fit must be null rather than 0.0: "not analysed"
    # and "analysed, no fit" mean different things to a user choosing what to open.
    assert all(job["fit_coverage"] is None for job in page["jobs"])
    # Ranked by the cheap signal in the meantime.
    overlaps = [job["skill_overlap"] for job in page["jobs"]]
    assert overlaps == sorted(overlaps, reverse=True)


@pytest.mark.asyncio
async def test_job_detail_carries_the_requirement_matrix_after_a_run(
    client: httpx.AsyncClient,
) -> None:
    await _login(client)
    page = (await client.get("/jobs", params={"limit": 20})).json()
    job = _viable_job(page["jobs"])

    before = (await client.get(f"/jobs/{job['job_id']}")).json()
    assert before["analysed"] is False
    assert before["requirement_matrix"] == []

    run = (await client.post("/runs", json={"job_id": job["job_id"]})).json()
    async with client.stream("GET", f"/runs/{run['thread_id']}/stream") as response:
        async for _ in response.aiter_lines():
            pass

    after = (await client.get(f"/jobs/{job['job_id']}")).json()
    assert after["analysed"] is True
    assert after["requirement_matrix"], "the evidence matrix drives the job detail screen"
    assert after["risk"] is not None
    assert after["price"] is not None
    for row in after["requirement_matrix"]:
        assert "requirement" in row and "covered" in row


# ------------------------------------------------------------------------ runs


@pytest.mark.asyncio
async def test_starting_the_same_run_twice_reuses_one_thread(
    client: httpx.AsyncClient,
) -> None:
    """A double-click must not produce two threads bidding on one job."""
    await _login(client)
    page = (await client.get("/jobs", params={"limit": 20})).json()
    job = _viable_job(page["jobs"])

    first = (await client.post("/runs", json={"job_id": job["job_id"]})).json()
    second = (await client.post("/runs", json={"job_id": job["job_id"]})).json()
    assert first["thread_id"] == second["thread_id"]


@pytest.mark.asyncio
async def test_the_stream_emits_node_events_then_awaiting_approval(
    client: httpx.AsyncClient,
) -> None:
    """Part D: the user watches progress rather than a spinner."""
    await _login(client)
    page = (await client.get("/jobs", params={"limit": 20})).json()
    job = _viable_job(page["jobs"])
    run = (await client.post("/runs", json={"job_id": job["job_id"]})).json()

    events: list[str] = []
    async with client.stream("GET", f"/runs/{run['thread_id']}/stream") as response:
        assert response.headers["content-type"].startswith("text/event-stream")
        async for line in response.aiter_lines():
            if line.startswith("event: "):
                events.append(line[7:])

    assert events[0] == "run_started"
    assert "node" in events, f"no node progress events: {events}"
    # An interrupt ends the stream cleanly so the UI can swap in the approval screen,
    # rather than looking like a dropped connection.
    assert events[-1] in ("awaiting_approval", "run_finished")


@pytest.mark.asyncio
async def test_the_run_trace_records_prompt_versions(client: httpx.AsyncClient) -> None:
    await _login(client)
    page = (await client.get("/jobs", params={"limit": 20})).json()
    job = _viable_job(page["jobs"])
    run = (await client.post("/runs", json={"job_id": job["job_id"]})).json()
    async with client.stream("GET", f"/runs/{run['thread_id']}/stream") as response:
        async for _ in response.aiter_lines():
            pass

    detail = (await client.get(f"/runs/{run['thread_id']}")).json()
    assert detail["trace"], "the run viewer needs a trace"
    nodes = [entry["node"] for entry in detail["trace"]]
    assert "scout" in nodes and "fan_out" in nodes
    assert any("@v" in (entry.get("prompt") or "") for entry in detail["trace"])


@pytest.mark.asyncio
async def test_one_user_cannot_read_another_users_thread(
    client: httpx.AsyncClient,
) -> None:
    """E4: every query is scoped by tenant, and thread ids embed the user."""
    await _login(client)
    response = await client.get("/runs/bid:someone-else:upw-1")
    assert response.status_code == 403


# ------------------------------------------------------------------- approvals


@pytest.mark.asyncio
async def test_a_pending_approval_is_listed_and_carries_what_c5_requires(
    client: httpx.AsyncClient,
) -> None:
    await _login(client)
    page = (await client.get("/jobs", params={"limit": 20})).json()
    job = _viable_job(page["jobs"])
    run = (await client.post("/runs", json={"job_id": job["job_id"]})).json()
    async with client.stream("GET", f"/runs/{run['thread_id']}/stream") as response:
        async for _ in response.aiter_lines():
            pass

    pending = (await client.get("/approvals")).json()
    if not pending:
        pytest.skip("this job did not reach an approval (decision was skip)")

    approval = pending[0]
    payload = approval["payload"]
    for key in ("draft", "claims_with_evidence", "risk", "fit", "price", "dry_run_preview"):
        assert key in payload, f"the approval payload is missing {key}"
    assert "cannot be undone" in payload["warning"]

    single = (await client.get(f"/approvals/{approval['thread_id']}")).json()
    assert single["thread_id"] == approval["thread_id"]


@pytest.mark.asyncio
async def test_approving_submits_once_and_a_second_click_is_a_no_op(
    client: httpx.AsyncClient,
) -> None:
    """R4 and C5 through the API, which is how a human actually does it."""
    await _login(client)
    page = (await client.get("/jobs", params={"limit": 20})).json()
    job = _viable_job(page["jobs"])
    freelancer = PERSONA
    before = store.connects_balance(freelancer)["balance"]

    run = (await client.post("/runs", json={"job_id": job["job_id"]})).json()
    async with client.stream("GET", f"/runs/{run['thread_id']}/stream") as response:
        async for _ in response.aiter_lines():
            pass

    pending = (await client.get("/approvals")).json()
    if not pending:
        pytest.skip("this job did not reach an approval")
    tid = pending[0]["thread_id"]

    first = (await client.post(f"/approvals/{tid}", json={"action": "approve"})).json()
    assert first["status"] == "submitted"
    assert first["proposal_id"]

    second = (await client.post(f"/approvals/{tid}", json={"action": "approve"})).json()
    assert second["idempotent_replay"] is True

    assert len(store.proposals) == 1, "a second click must not submit again"
    spent = before - store.connects_balance(freelancer)["balance"]
    assert spent == job["connects_cost"]


@pytest.mark.asyncio
async def test_an_edit_with_no_text_is_rejected(client: httpx.AsyncClient) -> None:
    """Sending the original would silently discard what the human wrote."""
    await _login(client)
    page = (await client.get("/jobs", params={"limit": 20})).json()
    job = _viable_job(page["jobs"])
    run = (await client.post("/runs", json={"job_id": job["job_id"]})).json()
    async with client.stream("GET", f"/runs/{run['thread_id']}/stream") as response:
        async for _ in response.aiter_lines():
            pass

    pending = (await client.get("/approvals")).json()
    if not pending:
        pytest.skip("this job did not reach an approval")

    response = await client.post(
        f"/approvals/{pending[0]['thread_id']}", json={"action": "edit"}
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_rejecting_sends_nothing(client: httpx.AsyncClient) -> None:
    await _login(client)
    page = (await client.get("/jobs", params={"limit": 20})).json()
    job = _viable_job(page["jobs"])
    run = (await client.post("/runs", json={"job_id": job["job_id"]})).json()
    async with client.stream("GET", f"/runs/{run['thread_id']}/stream") as response:
        async for _ in response.aiter_lines():
            pass

    pending = (await client.get("/approvals")).json()
    if not pending:
        pytest.skip("this job did not reach an approval")

    result = (
        await client.post(
            f"/approvals/{pending[0]['thread_id']}",
            json={"action": "reject", "reason": "rate too low"},
        )
    ).json()
    assert result["status"] == "rejected"
    assert not store.proposals


# --------------------------------------------------------------- other screens


@pytest.mark.asyncio
async def test_pipeline_analytics_memory_and_settings_all_answer(
    client: httpx.AsyncClient,
) -> None:
    await _login(client)

    board = (await client.get("/pipeline")).json()
    assert set(board["columns"]) >= {"drafted", "submitted", "replied", "hired", "lost"}

    analytics = (await client.get("/analytics")).json()
    assert "proposals_submitted" in analytics
    assert "top_skip_reasons" in analytics

    memory = (await client.get("/memory")).json()
    # The viewer must explain the rules, not just list what was learned (viva question 6).
    assert len(memory["promotion_rules"]) == 2

    settings_body = (await client.get("/settings")).json()
    assert settings_body["persona"] == PERSONA
    assert "embeddings_spec_compliant" in settings_body


@pytest.mark.asyncio
async def test_a_trace_id_is_echoed_on_every_response(client: httpx.AsyncClient) -> None:
    """E3: one id from the UI request through the graph to the audit log."""
    response = await client.get("/health", headers={"X-Trace-Id": "tr-from-the-browser"})
    assert response.headers["X-Trace-Id"] == "tr-from-the-browser"
    assert "X-Response-Time-ms" in response.headers

    minted = await client.get("/health")
    assert minted.headers["X-Trace-Id"].startswith("tr-")


@pytest.mark.asyncio
async def test_messages_screen_lists_threads(client: httpx.AsyncClient) -> None:
    await _login(client)
    threads = (await client.get("/messages")).json()
    assert isinstance(threads, list)
    if threads:
        detail = (await client.get(f"/messages/{threads[0]['thread_id']}")).json()
        assert detail["thread_id"] == threads[0]["thread_id"]
        assert "messages" in detail
