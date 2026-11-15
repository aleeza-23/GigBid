"""Pipeline board, analytics, memory viewer and settings.

Grouped in one router because they are all read-mostly views over the same three sources:
the marketplace (what happened), the checkpointer (what we decided), and the memory store
(what we learned).

The analytics numbers come from GigSim's own proposal records rather than from our
threads, deliberately. Reply and hire rate are facts about what clients did, and taking
them from our own state would mean reporting our opinion of the outcome.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request, status

from apps.api.deps import CurrentUser, bid_graph, memory_for
from apps.api.schemas.models import (
    Analytics,
    ForgetRequest,
    MemoryView,
    PipelineBoard,
    PipelineCard,
    Settings,
)

router = APIRouter(tags=["insights"])

# GigSim proposal status -> the board column it belongs in.
COLUMNS = {
    "draft": "drafted",
    "submitted": "submitted",
    "viewed": "viewed",
    "replied": "replied",
    "interview": "interview",
    "hired": "hired",
    "declined": "lost",
}
COLUMN_ORDER = ["drafted", "submitted", "viewed", "replied", "interview", "hired", "lost"]


async def _proposals(persona: str) -> list[dict[str, Any]]:
    """This freelancer's proposals, straight from the marketplace."""
    from agent.marketplace_port import MarketplaceError
    from mcp_servers.marketplace import server as market

    port = await market._marketplace()
    try:
        payload = await port._request(
            "GET", "/proposals", params={"freelancer_id": persona, "limit": 200}
        )
    except MarketplaceError as exc:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE, f"marketplace unavailable: {exc}"
        ) from exc
    return list(payload.get("proposals") or [])


@router.get("/pipeline", response_model=PipelineBoard)
async def pipeline(request: Request, user: dict = CurrentUser) -> PipelineBoard:
    """The kanban board. Cards move as the marketplace reports outcomes."""
    rows = await _proposals(user["persona"])
    columns: dict[str, list[PipelineCard]] = {name: [] for name in COLUMN_ORDER}

    for proposal in rows:
        column = COLUMNS.get(str(proposal.get("status")), "submitted")
        columns[column].append(
            PipelineCard(
                job_id=str(proposal.get("job_id") or ""),
                title=None,
                column=column,
                proposal_id=proposal.get("proposal_id"),
                connects_spent=proposal.get("connects_spent"),
                price_usd=proposal.get("bid_rate"),
                updated_at=proposal.get("decided_at") or proposal.get("created_at"),
            )
        )

    # Threads that drafted but never submitted belong on the board too: a proposal the
    # user never approved is the most actionable card there is.
    try:
        checkpointer = request.app.state.checkpointer
        prefix = f"bid:{user['user_id']}:"
        graph = bid_graph(request)
        seen: set[str] = set()
        async for tup in checkpointer.alist(None, limit=500):
            tid = str((tup.config or {}).get("configurable", {}).get("thread_id") or "")
            if not tid.startswith(prefix) or tid in seen:
                continue
            seen.add(tid)
            snapshot = await graph.aget_state({"configurable": {"thread_id": tid}})
            values = dict(snapshot.values or {})
            if not values or values.get("submission"):
                continue
            if not values.get("proposal_draft"):
                continue
            columns["drafted"].append(
                PipelineCard(
                    job_id=str(values.get("job_id") or ""),
                    title=(values.get("job") or {}).get("title"),
                    column="drafted",
                    thread_id=tid,
                    price_usd=getattr(values.get("price_quote"), "amount_usd", None),
                    fit_coverage=getattr(values.get("fit_report"), "coverage", None),
                )
            )
    except Exception:
        # No thread listing on this checkpointer: the board still shows marketplace rows.
        pass

    return PipelineBoard(
        columns=columns, totals={name: len(cards) for name, cards in columns.items()}
    )


@router.get("/analytics", response_model=Analytics)
async def analytics(request: Request, user: dict = CurrentUser) -> Analytics:
    """Reply and hire rate by category, Connects against outcomes, cost and latency."""
    from mcp_servers.marketplace import server as market

    rows = await _proposals(user["persona"])
    submitted = [p for p in rows if p.get("status") != "draft"]
    replies = [p for p in submitted if p.get("status") in ("replied", "interview", "hired")]
    interviews = [p for p in submitted if p.get("status") in ("interview", "hired")]
    hires = [p for p in submitted if p.get("status") == "hired"]

    connects_spent = sum(int(p.get("connects_spent") or 0) for p in submitted)
    remaining: int | None = None
    try:
        balance = await market.get_connects_balance(user["persona"])
        remaining = int(balance.get("remaining_this_week") or 0)
    except Exception:
        remaining = None

    # Skip reasons come from our own decisions, because the marketplace never sees a job
    # we chose not to bid on.
    skip_reasons: dict[str, int] = {}
    latencies: list[float] = []
    tokens: list[int] = []
    unsupported: list[int] = []
    try:
        checkpointer = request.app.state.checkpointer
        graph = bid_graph(request)
        prefix = f"bid:{user['user_id']}:"
        seen: set[str] = set()
        async for tup in checkpointer.alist(None, limit=500):
            tid = str((tup.config or {}).get("configurable", {}).get("thread_id") or "")
            if not tid.startswith(prefix) or tid in seen:
                continue
            seen.add(tid)
            snapshot = await graph.aget_state({"configurable": {"thread_id": tid}})
            values = dict(snapshot.values or {})
            decision = values.get("decision")
            if getattr(decision, "decision", None) in ("skip", "watch"):
                factor = getattr(decision, "deciding_factor", "") or "unknown"
                skip_reasons[factor] = skip_reasons.get(factor, 0) + 1
            if values.get("tokens_used"):
                tokens.append(int(values["tokens_used"]))
            verification = values.get("claim_verification") or {}
            if verification.get("unsupported_after") is not None:
                unsupported.append(int(verification["unsupported_after"]))
            for entry in values.get("execution_trace") or []:
                if entry.get("duration_ms"):
                    latencies.append(float(entry["duration_ms"]))
    except Exception:
        pass

    def percentile(values: list[float], fraction: float) -> float | None:
        if not values:
            return None
        ordered = sorted(values)
        return round(ordered[min(len(ordered) - 1, int(len(ordered) * fraction))], 1)

    by_category: dict[str, dict[str, int]] = {}
    for proposal in submitted:
        key = str(proposal.get("job_id") or "")[:3] or "other"
        bucket = by_category.setdefault(key, {"submitted": 0, "replied": 0, "hired": 0})
        bucket["submitted"] += 1
        if proposal.get("status") in ("replied", "interview", "hired"):
            bucket["replied"] += 1
        if proposal.get("status") == "hired":
            bucket["hired"] += 1

    return Analytics(
        proposals_submitted=len(submitted),
        replies=len(replies),
        interviews=len(interviews),
        hires=len(hires),
        reply_rate=round(len(replies) / len(submitted), 4) if submitted else None,
        hire_rate=round(len(hires) / len(submitted), 4) if submitted else None,
        connects_spent=connects_spent,
        connects_remaining=remaining,
        by_category=[{"category": k, **v} for k, v in sorted(by_category.items())],
        cost={
            "total_tokens": sum(tokens),
            "mean_tokens_per_run": round(sum(tokens) / len(tokens), 1) if tokens else None,
            "runs": len(tokens),
        },
        latency={
            "node_p50_ms": percentile(latencies, 0.5),
            "node_p95_ms": percentile(latencies, 0.95),
            "samples": len(latencies),
        },
        top_skip_reasons=[
            {"reason": k, "count": v}
            for k, v in sorted(skip_reasons.items(), key=lambda kv: -kv[1])
        ],
        unsupported_claim_rate=(
            round(sum(1 for u in unsupported if u) / len(unsupported), 4)
            if unsupported
            else None
        ),
    )


@router.get("/memory", response_model=MemoryView)
async def view_memory(request: Request, user: dict = CurrentUser) -> MemoryView:
    """What GigBid has learned about this user, and the rule that let each item in."""
    view = memory_for(request, user["user_id"]).viewer()
    return MemoryView(**view)


@router.post("/memory/forget")
async def forget(
    body: ForgetRequest, request: Request, user: dict = CurrentUser
) -> dict[str, Any]:
    """Delete one learned item. Viva question 6 asks for exactly this, from the UI.

    By default only the *rule* is deleted, so a later promotion could re-derive it from
    the same evidence. `forget_evidence` removes the observations too, which is the
    difference between "stop applying this" and "this was never my preference".
    """
    memory = memory_for(request, user["user_id"])
    removed_rule = memory.forget(body.kind, body.key)
    removed_evidence = (
        memory.forget_observations(body.kind, body.key) if body.forget_evidence else 0
    )
    return {
        "forgotten": removed_rule,
        "observations_removed": removed_evidence,
        "note": (
            "the evidence remains, so this rule may be re-derived"
            if not body.forget_evidence
            else "the evidence was removed too, so it cannot be re-derived"
        ),
    }


@router.get("/settings", response_model=Settings)
async def settings(request: Request, user: dict = CurrentUser) -> Settings:
    """Autonomy, budgets, connected accounts, and what is actually configured."""
    from agent.marketplace_port import load_config
    from mcp_servers.marketplace import server as market
    from mcp_servers.portfolio import corpus
    from rag.retriever import describe as retrieval_describe

    config = load_config()
    retrieval = retrieval_describe()

    profile: dict[str, Any] = {}
    try:
        profile = corpus.load_profile(user["persona"])
    except Exception:
        profile = {}

    budget: dict[str, Any] = {}
    try:
        budget = await market.get_connects_balance(user["persona"])
    except Exception:
        budget = {}

    return Settings(
        persona=user["persona"],
        autonomy="manual",
        weekly_connects_budget=budget.get("weekly_budget"),
        connects_balance=budget.get("balance"),
        hourly_rate_usd=profile.get("hourly_rate_usd"),
        minimum_project_budget_usd=profile.get("minimum_project_budget_usd"),
        marketplace_mode=(config.get("marketplace") or {}).get("mode", "gigsim"),
        model_profile=(config.get("models") or {}).get("profile", "open"),
        retrieval_variant=str(retrieval.get("configured")),
        embeddings_backend=str(retrieval.get("embeddings_backend")),
        # Surfaced in the UI because a number produced on the development embedding
        # backend must not be mistaken for a measured one.
        embeddings_spec_compliant=retrieval.get("embeddings_backend") != "lsa",
        saved_searches=[],
    )
