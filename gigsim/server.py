"""The GigSim HTTP API.

A stand-in marketplace: job search, client histories, Connects, proposals, messages,
rate limits and occasional 5xx. Writes are idempotent by key, so a retried submission
never spends Connects twice, and the simulation only advances when something asks it to,
which keeps evaluation runs repeatable.

    python -m gigsim          # http://localhost:8081, docs at /docs
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from fastapi import Body, FastAPI, Header, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from gigsim import __version__
from gigsim.clients_sim import decide, reply_to_message
from gigsim.config import settings
from gigsim.models import (
    ConnectsBalance,
    DryRunPreview,
    JobPage,
    Proposal,
    ProposalCreate,
    TickResult,
)
from gigsim.rate_limit import bucket, should_fail
from gigsim.store import store

API = "/api/v1"

app = FastAPI(
    title="GigSim",
    version=__version__,
    description=(
        "Marketplace simulator for GigBid. Clients have preferences the agent cannot see; "
        "outcomes come from scoring proposals against them."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

OPEN_PATHS = {"/health", "/docs", "/openapi.json", "/redoc"}


def _now() -> str:
    return datetime.now(UTC).isoformat()


@app.middleware("http")
async def gate(request: Request, call_next):
    """Bearer auth, per-token rate limiting and synthetic failures, in that order."""
    path = request.url.path
    if path in OPEN_PATHS or path.startswith("/docs"):
        return await call_next(request)

    auth = request.headers.get("authorization", "")
    token = auth[7:].strip() if auth.lower().startswith("bearer ") else ""
    if token != settings.api_token:
        return JSONResponse(
            status_code=401,
            content={"error": "unauthorized", "detail": "Send Authorization: Bearer <token>"},
        )

    allowed, retry_after = bucket.take(token)
    if not allowed:
        return JSONResponse(
            status_code=429,
            content={"error": "rate_limited", "detail": f"Retry in {retry_after}s"},
            headers={"Retry-After": str(retry_after)},
        )

    if request.method == "GET" and should_fail():
        return JSONResponse(
            status_code=503,
            content={"error": "upstream_unavailable", "detail": "Synthetic GigSim outage"},
        )

    return await call_next(request)


# ------------------------------------------------------------------ discovery

@app.get("/health", tags=["meta"])
async def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "version": __version__,
        "jobs": len(store.jobs),
        "clients": len(store.clients),
        "client_engine": settings.client_engine,
        "rate_limit_per_min": settings.rate_limit_per_min,
        "error_rate": settings.error_rate,
    }


# ---------------------------------------------------------------------- jobs

@app.get(f"{API}/jobs/historical", tags=["jobs"])
async def historical_jobs(
    category: str | None = None,
    limit: int = Query(100, ge=1, le=1000),
) -> dict[str, Any]:
    """Closed jobs, for pricing statistics rather than bidding."""
    rows = store.historical
    if category:
        rows = [j for j in rows if (j.get("category") or "").lower() == category.lower()]
    return {"total": len(rows), "jobs": rows[:limit]}


@app.get(f"{API}/jobs", response_model=JobPage, tags=["jobs"])
async def search_jobs(
    q: str | None = None,
    skills: list[str] | None = Query(default=None),
    category: str | None = None,
    country: str | None = None,
    is_hourly: bool | None = None,
    min_hourly: float | None = None,
    max_hourly: float | None = None,
    posted_after: str | None = None,
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
) -> JobPage:
    jobs, total = store.search_jobs(
        q=q,
        skills=skills,
        category=category,
        country=country,
        is_hourly=is_hourly,
        min_hourly=min_hourly,
        max_hourly=max_hourly,
        posted_after=posted_after,
        limit=limit,
        offset=offset,
    )
    return JobPage(total=total, limit=limit, offset=offset, jobs=jobs)


@app.get(f"{API}/jobs/{{job_id}}", tags=["jobs"])
async def get_job(job_id: str) -> dict[str, Any]:
    job = store.get_job(job_id)
    if job is None:
        raise HTTPException(404, "job not found")
    return job


# ------------------------------------------------------------------- clients

@app.get(f"{API}/clients/{{client_id}}", tags=["clients"])
async def get_client(client_id: str) -> dict[str, Any]:
    client = store.get_client(client_id)
    if client is None:
        raise HTTPException(404, "client not found")
    return client


@app.get(f"{API}/clients/{{client_id}}/history", tags=["clients"])
async def client_history(client_id: str) -> dict[str, Any]:
    history = store.client_history(client_id)
    if history is None:
        raise HTTPException(404, "client not found")
    return history


@app.get(f"{API}/stats/rates", tags=["clients"])
async def rate_stats(category: str | None = None, domain: str | None = None) -> dict[str, Any]:
    return store.rate_stats(category=category, domain=domain)


# ------------------------------------------------------------------ connects

@app.get(f"{API}/freelancers/{{freelancer_id}}/connects",
         response_model=ConnectsBalance, tags=["connects"])
async def connects(freelancer_id: str) -> ConnectsBalance:
    acct = store.connects_balance(freelancer_id)
    if acct is None:
        raise HTTPException(404, "freelancer not found")
    return ConnectsBalance(**acct)


# ----------------------------------------------------------------- proposals

def _preview(body: ProposalCreate, job: dict[str, Any]) -> DryRunPreview:
    cost = int(job.get("connects_cost") or 0)
    acct = store.connects_balance(body.freelancer_id) or {}
    balance = int(acct.get("balance", 0))
    spent = int(acct.get("spent_this_week", 0))
    budget = int(acct.get("weekly_budget", 0))

    warnings: list[str] = []
    if balance < cost:
        warnings.append(f"insufficient Connects: have {balance}, need {cost}")
    if spent + cost > budget:
        warnings.append(f"over weekly budget: {spent}+{cost} > {budget}")

    return DryRunPreview(
        job_id=body.job_id,
        would_send={
            "cover_letter": body.cover_letter,
            "screening_answers": [a.model_dump() for a in body.screening_answers],
            "bid_rate": body.bid_rate,
            "estimated_duration_days": body.estimated_duration_days,
        },
        connects_cost=cost,
        connects_balance_after=balance - cost,
        within_weekly_budget=spent + cost <= budget,
        warnings=warnings,
    )


@app.post(f"{API}/proposals", tags=["proposals"])
async def create_proposal(
    body: ProposalCreate,
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
) -> dict[str, Any]:
    """Submit a proposal. Spends Connects unless dry_run is set.

    Send an Idempotency-Key; a repeat of the same key returns the first result untouched
    and does not spend Connects again.
    """
    job = store.jobs.get(body.job_id)
    if job is None:
        raise HTTPException(404, "job not found")

    if body.dry_run:
        return _preview(body, job).model_dump()

    if not idempotency_key:
        raise HTTPException(400, "Idempotency-Key header is required for writes")

    cached = store.idempotency.get(idempotency_key)
    if cached is not None:
        return {**cached, "idempotent_replay": True}

    ok, reason = store.spend_connects(body.freelancer_id, int(job.get("connects_cost") or 0))
    if not ok:
        raise HTTPException(402, reason)

    proposal = Proposal(
        proposal_id=store.new_id("prop"),
        job_id=body.job_id,
        client_id=job["client_id"],
        freelancer_id=body.freelancer_id,
        cover_letter=body.cover_letter,
        screening_answers=body.screening_answers,
        bid_rate=body.bid_rate,
        estimated_duration_days=body.estimated_duration_days,
        status="submitted",
        connects_spent=int(job.get("connects_cost") or 0),
        created_at=_now(),
        idempotency_key=idempotency_key,
    ).model_dump()

    store.proposals[proposal["proposal_id"]] = proposal
    store.idempotency[idempotency_key] = proposal
    store.log(
        "proposal_submitted",
        proposal_id=proposal["proposal_id"],
        job_id=body.job_id,
        freelancer_id=body.freelancer_id,
        connects_spent=proposal["connects_spent"],
    )
    return proposal


@app.get(f"{API}/proposals", tags=["proposals"])
async def list_proposals(
    freelancer_id: str | None = None,
    status: str | None = None,
) -> dict[str, Any]:
    rows = list(store.proposals.values())
    if freelancer_id:
        rows = [p for p in rows if p["freelancer_id"] == freelancer_id]
    if status:
        rows = [p for p in rows if p["status"] == status]
    return {"total": len(rows), "proposals": rows}


@app.get(f"{API}/proposals/{{proposal_id}}", tags=["proposals"])
async def get_proposal(proposal_id: str) -> dict[str, Any]:
    proposal = store.proposals.get(proposal_id)
    if proposal is None:
        raise HTTPException(404, "proposal not found")
    return proposal


# ------------------------------------------------------------------ messages

@app.get(f"{API}/threads", tags=["messages"])
async def list_threads(freelancer_id: str | None = None) -> dict[str, Any]:
    rows = list(store.threads.values())
    if freelancer_id:
        rows = [t for t in rows if t["freelancer_id"] == freelancer_id]
    return {"total": len(rows), "threads": rows}


@app.get(f"{API}/threads/{{thread_id}}/messages", tags=["messages"])
async def list_messages(thread_id: str) -> dict[str, Any]:
    thread = store.threads.get(thread_id)
    if thread is None:
        raise HTTPException(404, "thread not found")
    return {"thread_id": thread_id, "messages": thread["messages"]}


@app.post(f"{API}/messages", tags=["messages"])
async def send_message(
    thread_id: str = Body(..., embed=True),
    text: str = Body(..., embed=True),
    dry_run: bool = Body(False, embed=True),
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
) -> dict[str, Any]:
    """Send a message as the freelancer. The client answers immediately."""
    thread = store.threads.get(thread_id)
    if thread is None:
        raise HTTPException(404, "thread not found")

    if dry_run:
        return {
            "dry_run": True,
            "thread_id": thread_id,
            "would_send": {"from": "freelancer", "text": text},
            "connects_cost": 0,
        }

    if not idempotency_key:
        raise HTTPException(400, "Idempotency-Key header is required for writes")

    cached = store.idempotency.get(idempotency_key)
    if cached is not None:
        return {**cached, "idempotent_replay": True}

    thread["messages"].append({"from": "freelancer", "text": text, "ts": _now()})
    client = store.clients.get(thread["client_id"], {})
    reply = reply_to_message(client, thread, text)
    thread["messages"].append({"from": "client", "text": reply, "ts": _now()})

    result = {
        "thread_id": thread_id,
        "sent": text,
        "client_reply": reply,
        "messages": thread["messages"],
    }
    store.idempotency[idempotency_key] = result
    store.log("message_sent", thread_id=thread_id, chars=len(text))
    return result


# --------------------------------------------------------------------- admin

@app.post(f"{API}/admin/tick", response_model=TickResult, tags=["admin"])
async def tick() -> TickResult:
    """Let clients act on every proposal they have not decided yet.

    Keeping this explicit means a run has no wall-clock dependence: submit, tick, read
    the outcomes, and the same inputs always give the same numbers.
    """
    outcomes: dict[str, int] = {}
    details: list[dict[str, Any]] = []

    for proposal in store.proposals.values():
        if proposal["status"] != "submitted":
            continue
        client = store.clients.get(proposal["client_id"])
        if client is None:
            continue

        decision = decide(client, proposal)
        proposal["status"] = decision.outcome if decision.outcome != "ignored" else "submitted"
        proposal["decided_at"] = _now()
        proposal["decision_score"] = round(decision.score, 4)
        outcomes[decision.outcome] = outcomes.get(decision.outcome, 0) + 1

        if decision.reply_text:
            thread = store.ensure_thread(
                proposal["job_id"],
                proposal["client_id"],
                proposal["freelancer_id"],
                proposal["proposal_id"],
            )
            thread["proposal_id"] = proposal["proposal_id"]
            thread["messages"].append(
                {"from": "client", "text": decision.reply_text, "ts": _now()}
            )

        details.append(
            {
                "proposal_id": proposal["proposal_id"],
                "job_id": proposal["job_id"],
                "outcome": decision.outcome,
                "score": round(decision.score, 4),
                "reasons": decision.reasons,
            }
        )
        store.log(
            "client_decision",
            proposal_id=proposal["proposal_id"],
            outcome=decision.outcome,
            score=round(decision.score, 4),
        )

    return TickResult(decided=len(details), outcomes=outcomes, details=details)


@app.post(f"{API}/admin/reset", tags=["admin"])
async def reset() -> dict[str, Any]:
    """Back to a clean marketplace. Seed data is never touched."""
    store.reset()
    bucket.reset()
    return {"status": "reset", "threads": len(store.threads), "proposals": 0}


@app.get(f"{API}/admin/events", tags=["admin"])
async def events(limit: int = Query(200, ge=1, le=5000)) -> dict[str, Any]:
    """Append-only log of everything that happened, for audit and the pipeline board."""
    return {"total": len(store.events), "events": store.events[-limit:]}
