"""Integration tests for the GigSim marketplace simulator.

These cover the behaviour the agent depends on: that hidden data stays hidden, that a
retried write never spends Connects twice, that dry runs cost nothing, and that the
marketplace pushes back when it is called too often.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from gigsim.config import settings
from gigsim.rate_limit import TokenBucket, bucket
from gigsim.server import app
from gigsim.store import store

AUTH = {"Authorization": f"Bearer {settings.api_token}"}

LETTER = (
    "I built an invoice extraction pipeline in Python that cut manual entry by 80 percent "
    "across 12000 documents, and I can do the same here within three weeks."
)


@pytest.fixture
def client() -> TestClient:
    store.reset()
    bucket.reset()
    return TestClient(app)


@pytest.fixture
def job() -> dict:
    """A real job with a known client and Connects cost."""
    return next(j for j in store.jobs.values() if j.get("connects_cost"))


# ------------------------------------------------------------------ the basics

def test_health_needs_no_token(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["jobs"] == len(store.jobs)


def test_every_other_route_needs_a_token(client: TestClient) -> None:
    assert client.get("/api/v1/jobs").status_code == 401
    assert client.get("/api/v1/jobs", headers={"Authorization": "Bearer wrong"}).status_code == 401


def test_search_filters_and_pages(client: TestClient) -> None:
    first = client.get("/api/v1/jobs", params={"limit": 5}, headers=AUTH).json()
    second = client.get("/api/v1/jobs", params={"limit": 5, "offset": 5}, headers=AUTH).json()
    assert first["total"] == second["total"] == len(store.jobs)
    assert len(first["jobs"]) == 5
    assert {j["job_id"] for j in first["jobs"]}.isdisjoint({j["job_id"] for j in second["jobs"]})

    hourly = client.get(
        "/api/v1/jobs", params={"is_hourly": True, "min_hourly": 40, "limit": 50}, headers=AUTH
    ).json()
    assert all(j["is_hourly"] and j["hourly_high"] >= 40 for j in hourly["jobs"])


# ------------------------------------------------------- what the agent cannot see

def test_jobs_never_reveal_which_post_they_duplicate(client: TestClient) -> None:
    """The Scout has to detect reposts itself, so the answer is not in the payload."""
    page = client.get("/api/v1/jobs", params={"limit": 100}, headers=AUTH).json()
    assert any(j.get("duplicate_of") for j in store.jobs.values()), "seed has no reposts to hide"
    assert all("duplicate_of" not in j for j in page["jobs"])


def test_clients_never_reveal_preferences_or_the_scam_flag(client: TestClient) -> None:
    scam_id = next(c["client_id"] for c in store.clients.values() if c.get("is_scam_account"))
    profile = client.get(f"/api/v1/clients/{scam_id}", headers=AUTH).json()
    assert "hidden_preferences" not in profile
    assert "is_scam_account" not in profile

    history = client.get(f"/api/v1/clients/{scam_id}/history", headers=AUTH).json()
    assert "hidden_preferences" not in history["client"]
    assert "is_scam_account" not in history["client"]


# ----------------------------------------------------------------- the write path

def test_dry_run_costs_nothing(client: TestClient, job: dict) -> None:
    before = client.get("/api/v1/freelancers/ai_ml_engineer/connects", headers=AUTH).json()
    response = client.post(
        "/api/v1/proposals",
        headers=AUTH,
        json={
            "freelancer_id": "ai_ml_engineer",
            "job_id": job["job_id"],
            "cover_letter": LETTER,
            "bid_rate": 55.0,
            "dry_run": True,
        },
    )
    preview = response.json()
    assert response.status_code == 200
    assert preview["dry_run"] is True
    assert preview["connects_cost"] == job["connects_cost"]
    assert preview["would_send"]["cover_letter"] == LETTER

    after = client.get("/api/v1/freelancers/ai_ml_engineer/connects", headers=AUTH).json()
    assert after == before
    assert store.proposals == {}


def test_write_without_an_idempotency_key_is_refused(client: TestClient, job: dict) -> None:
    response = client.post(
        "/api/v1/proposals",
        headers=AUTH,
        json={
            "freelancer_id": "ai_ml_engineer",
            "job_id": job["job_id"],
            "cover_letter": LETTER,
            "bid_rate": 55.0,
        },
    )
    assert response.status_code == 400


def test_a_retried_submission_spends_connects_once(client: TestClient, job: dict) -> None:
    """The exactly-once guarantee the viva asks you to prove."""
    body = {
        "freelancer_id": "ai_ml_engineer",
        "job_id": job["job_id"],
        "cover_letter": LETTER,
        "bid_rate": 55.0,
    }
    headers = {**AUTH, "Idempotency-Key": "key-abc-123"}
    start = client.get("/api/v1/freelancers/ai_ml_engineer/connects", headers=AUTH).json()

    first = client.post("/api/v1/proposals", headers=headers, json=body).json()
    second = client.post("/api/v1/proposals", headers=headers, json=body).json()

    assert first["proposal_id"] == second["proposal_id"]
    assert second["idempotent_replay"] is True
    assert len(store.proposals) == 1

    end = client.get("/api/v1/freelancers/ai_ml_engineer/connects", headers=AUTH).json()
    assert start["balance"] - end["balance"] == job["connects_cost"]
    assert end["spent_this_week"] == job["connects_cost"]


def test_the_weekly_budget_stops_runaway_bidding(client: TestClient) -> None:
    jobs = [j for j in store.jobs.values() if j.get("connects_cost")]
    refused = None
    for i, j in enumerate(jobs[:40]):
        response = client.post(
            "/api/v1/proposals",
            headers={**AUTH, "Idempotency-Key": f"budget-{i}"},
            json={
                "freelancer_id": "ai_ml_engineer",
                "job_id": j["job_id"],
                "cover_letter": LETTER,
                "bid_rate": 50.0,
            },
        )
        if response.status_code == 402:
            refused = response
            break
    assert refused is not None, "weekly budget never bit"
    assert refused.json()["detail"] in {"weekly_budget_exceeded", "insufficient_connects"}


def test_a_short_cover_letter_is_rejected_at_the_schema(client: TestClient, job: dict) -> None:
    response = client.post(
        "/api/v1/proposals",
        headers={**AUTH, "Idempotency-Key": "too-short"},
        json={
            "freelancer_id": "ai_ml_engineer",
            "job_id": job["job_id"],
            "cover_letter": "hi",
            "bid_rate": 55.0,
        },
    )
    assert response.status_code == 422


# --------------------------------------------------------------- client decisions

def _submit(client: TestClient, job: dict, letter: str, rate: float, key: str) -> dict:
    return client.post(
        "/api/v1/proposals",
        headers={**AUTH, "Idempotency-Key": key},
        json={
            "freelancer_id": "ai_ml_engineer",
            "job_id": job["job_id"],
            "cover_letter": letter,
            "bid_rate": rate,
        },
    ).json()


def test_tick_decides_and_is_reproducible(client: TestClient, job: dict) -> None:
    _submit(client, job, LETTER, 20.0, "tick-1")
    first = client.post("/api/v1/admin/tick", headers=AUTH).json()
    assert first["decided"] == 1
    assert first["details"][0]["reasons"], "a decision should say why"

    store.reset()
    _submit(client, job, LETTER, 20.0, "tick-1")
    second = client.post("/api/v1/admin/tick", headers=AUTH).json()
    assert first["details"][0]["outcome"] == second["details"][0]["outcome"]
    assert first["details"][0]["score"] == second["details"][0]["score"]


def test_a_wild_overbid_does_worse_than_a_sensible_one(client: TestClient) -> None:
    """Preferences the agent cannot see still move the outcome."""
    job = next(j for j in store.jobs.values() if j.get("connects_cost"))
    sensible = _submit(client, job, LETTER, 15.0, "cheap")
    store.reset()
    greedy = _submit(client, job, LETTER, 900.0, "dear")

    client_record = store.clients[job["client_id"]]
    from gigsim.clients_sim import score_proposal

    assert score_proposal(client_record, sensible).score > score_proposal(
        client_record, greedy
    ).score


def test_scam_clients_reply_but_never_hire(client: TestClient) -> None:
    from gigsim.clients_sim import score_proposal

    scam = next(c for c in store.clients.values() if c.get("is_scam_account"))
    decision = score_proposal(scam, {"job_id": "j1", "cover_letter": LETTER, "bid_rate": 20.0})
    assert decision.outcome == "replied"
    assert decision.reply_text is not None
    assert "scam" in " ".join(decision.reasons)


# --------------------------------------------------------------------- messages

def test_messaging_round_trip_is_idempotent(client: TestClient) -> None:
    thread_id = next(iter(store.threads))
    headers = {**AUTH, "Idempotency-Key": "msg-1"}
    body = {"thread_id": thread_id, "text": "Are you free for a call this week?"}

    first = client.post("/api/v1/messages", headers=headers, json=body).json()
    assert first["client_reply"]
    before = len(first["messages"])

    second = client.post("/api/v1/messages", headers=headers, json=body).json()
    assert second["idempotent_replay"] is True
    assert len(store.threads[thread_id]["messages"]) == before


def test_message_dry_run_sends_nothing(client: TestClient) -> None:
    thread_id = next(iter(store.threads))
    before = len(store.threads[thread_id]["messages"])
    response = client.post(
        "/api/v1/messages",
        headers=AUTH,
        json={"thread_id": thread_id, "text": "hello", "dry_run": True},
    ).json()
    assert response["dry_run"] is True
    assert len(store.threads[thread_id]["messages"]) == before


# ----------------------------------------------------- pushing back on the caller

def test_the_rate_limiter_refuses_and_then_recovers() -> None:
    small = TokenBucket(capacity=2)
    assert small.take("t")[0]
    assert small.take("t")[0]
    allowed, retry_after = small.take("t")
    assert not allowed
    assert retry_after > 0


def test_the_audit_log_records_every_write(client: TestClient, job: dict) -> None:
    _submit(client, job, LETTER, 20.0, "audit-1")
    client.post("/api/v1/admin/tick", headers=AUTH)
    events = client.get("/api/v1/admin/events", headers=AUTH).json()["events"]
    kinds = {e["kind"] for e in events}
    assert {"proposal_submitted", "client_decision"} <= kinds
    assert all("ts" in e for e in events)
