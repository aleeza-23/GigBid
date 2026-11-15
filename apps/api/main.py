"""GigBid API.

FastAPI over the two LangGraph graphs and the three MCP servers. The TypeScript client the
frontend uses is generated from this app's OpenAPI schema rather than hand-written, so the
schemas in `apps/api/schemas/` are the contract between the two halves.

    make api            # uvicorn on :8000
    open http://localhost:8000/docs

`/health` is the one endpoint worth reading before a demo: it reports whether the
marketplace is reachable, whether the checkpointer survives a restart, and whether the
retrieval backend is the spec-compliant one. All three are claims that are easy to assume
and embarrassing to get wrong in front of an examiner.
"""

from __future__ import annotations

import os
import time
from typing import Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from apps.api.deps import lifespan
from apps.api.routers import approvals, auth, insights, jobs, messages, runs
from apps.api.schemas.models import Health

app = FastAPI(
    title="GigBid API",
    version="0.1.0",
    description=(
        "Multi-agent freelancer copilot. Every proposal and message waits for a human "
        "before it leaves the system."
    ),
    lifespan=lifespan,
)

# The Next.js dev server runs on another port, so the browser needs CORS. Credentials are
# allowed because the session is an HttpOnly cookie, which also means the origin list has
# to be explicit: `allow_origins=["*"]` with credentials is rejected by browsers and would
# silently break sign-in.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        origin.strip()
        for origin in os.getenv(
            "CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000"
        ).split(",")
        if origin.strip()
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router)
app.include_router(jobs.router)
app.include_router(runs.router)
app.include_router(approvals.router)
app.include_router(messages.router)
app.include_router(insights.router)


@app.middleware("http")
async def trace_and_time(request: Request, call_next):
    """One trace id from the UI request through the graph to the audit log (E3).

    Adopts the caller's `X-Trace-Id` when it sends one, mints one otherwise, and echoes it
    back so the browser can print a link to the trace next to the run.
    """
    from mcp_servers._common.trace import get_trace_id, set_trace_id, set_user_id

    set_trace_id(request.headers.get("X-Trace-Id"))
    set_user_id(None)
    started = time.perf_counter()
    response = await call_next(request)
    response.headers["X-Trace-Id"] = get_trace_id()
    response.headers["X-Response-Time-ms"] = f"{(time.perf_counter() - started) * 1000:.1f}"
    return response


@app.exception_handler(Exception)
async def unhandled(request: Request, exc: Exception) -> JSONResponse:
    """Never leak a traceback to a browser; keep the trace id so it can be correlated."""
    from mcp_servers._common.trace import get_trace_id

    return JSONResponse(
        status_code=500,
        content={
            "error": "internal error",
            "detail": f"{type(exc).__name__}",
            "trace_id": get_trace_id(),
        },
    )


# The marketplace ping is cached for a few seconds. Every page in the UI mounts the
# navigation bar, which calls /health, so a reviewer clicking through five screens spent
# five of the marketplace's twenty calls per minute on liveness checks and the e2e suite
# exhausted the budget outright, turning the inbox into a "marketplace unavailable"
# banner. A few seconds is short enough that a real outage still shows up promptly.
_MARKETPLACE_PING_TTL_SEC = 5.0
_marketplace_ping: tuple[float, dict[str, Any]] | None = None


async def _marketplace_status() -> dict[str, Any]:
    global _marketplace_ping

    now = time.monotonic()
    if _marketplace_ping and now - _marketplace_ping[0] < _MARKETPLACE_PING_TTL_SEC:
        cached = dict(_marketplace_ping[1])
        cached["cached"] = True
        return cached

    from mcp_servers.marketplace import server as market

    status = await market.marketplace_health()
    _marketplace_ping = (now, dict(status))
    return status


@app.get("/health", response_model=Health, tags=["meta"])
async def health(request: Request) -> Health:
    """Is everything the UI depends on actually working?"""
    from agent.llm import get_llm
    from agent.marketplace_port import load_config
    from rag.retriever import describe as retrieval_describe

    degraded: list[str] = []

    marketplace: dict[str, Any] = {}
    try:
        marketplace = await _marketplace_status()
        if marketplace.get("status") != "ok":
            degraded.append(f"marketplace: {marketplace.get('status')}")
    except Exception as exc:
        marketplace = {"status": "unavailable", "detail": str(exc)[:200]}
        degraded.append("marketplace: unreachable")

    persistence = dict(getattr(request.app.state, "persistence", {}) or {})
    if not persistence.get("survives_restart"):
        degraded.append("persistence: pending approvals will not survive a restart")
    if persistence.get("available") is False:
        degraded.append(f"persistence: {persistence.get('detail')}")

    models: dict[str, Any] = {}
    try:
        llm = get_llm()
        models = {
            "profile": llm.profile_name,
            "provider": llm.profile.get("provider"),
            "small": llm.model_for("analyst"),
            "large": llm.model_for("writer"),
            "offline": llm.offline,
        }
        if llm.offline:
            degraded.append("models: offline, drafts are placeholders")
    except Exception as exc:
        models = {"error": str(exc)[:200]}
        degraded.append("models: misconfigured")

    retrieval = retrieval_describe()
    if retrieval.get("embeddings_backend") == "lsa":
        degraded.append("retrieval: development embedding backend, numbers not reportable")

    config = load_config()
    return Health(
        status="degraded" if degraded else "ok",
        started_at=getattr(request.app.state, "started_at", None),
        marketplace={
            **marketplace,
            "mode": (config.get("marketplace") or {}).get("mode", "gigsim"),
        },
        persistence=persistence,
        models=models,
        retrieval=retrieval,
        degraded=degraded,
    )


@app.get("/", tags=["meta"])
async def root() -> dict[str, Any]:
    return {
        "service": "gigbid-api",
        "docs": "/docs",
        "openapi": "/openapi.json",
        "health": "/health",
    }
