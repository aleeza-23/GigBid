"""Job inbox and job detail.

The inbox is ranked by fit, which raises a question the assignment implies but does not
answer: fit comes from running the bid graph, and running it on every job in the feed
costs tokens on jobs the user will never look at.

So there are two tiers, and the UI shows them differently:

* **not analysed** — ranked by skill overlap between the post and the persona's evidenced
  skills. Free, instant, and good enough to sort an inbox.
* **analysed** — real coverage, risk and price from a completed bid thread.

`fit_coverage: null` means the first tier. The inbox badge says "not analysed yet" rather
than implying a score of zero, because those are very different things to a user deciding
what to open.
"""

from __future__ import annotations

import asyncio
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request, status

from apps.api.deps import CurrentUser, bid_graph, thread_id
from apps.api.schemas.models import JobDetail, JobPage, JobSummary, RequirementEvidence

router = APIRouter(prefix="/jobs", tags=["jobs"])


async def _thread_values(request: Request, user_id: str, job_id: str) -> dict[str, Any]:
    """The bid thread's state for one job, or {} if it has never been run."""
    graph = bid_graph(request)
    config = {"configurable": {"thread_id": thread_id("bid", user_id, job_id)}}
    try:
        snapshot = await graph.aget_state(config)
    except Exception:
        return {}
    if not snapshot or not snapshot.values:
        return {}
    values = dict(snapshot.values)
    values["_awaiting_approval"] = bool(snapshot.interrupts)
    return values
THREAD_LOOKUP_TIMEOUT_S = 3.0
SEARCH_TIMEOUT_S = 25.0
CLIENT_HISTORY_TIMEOUT_S = 8.0


async def _thread_values_capped(request: Request, user_id: str, job_id: str) -> dict[str, Any]:
    try:
        return await asyncio.wait_for(
            _thread_values(request, user_id, job_id), timeout=THREAD_LOOKUP_TIMEOUT_S
        )
    except Exception:
        return {}

def _skill_overlap(job: dict[str, Any], own: set[str]) -> int:
    return len(own & {str(s).lower() for s in (job.get("skills") or [])})


def _summary(job: dict[str, Any], values: dict[str, Any], overlap: int) -> JobSummary:
    fit = values.get("fit_report")
    risk = values.get("risk_report")
    decision = values.get("decision")
    quote = values.get("price_quote")
    status_value = values.get("status")
    if values.get("_awaiting_approval"):
        status_value = "awaiting_approval"

    return JobSummary(
        job_id=str(job.get("job_id") or ""),
        title=str(job.get("title") or ""),
        client_id=job.get("client_id"),
        category=job.get("category"),
        skills=[str(s) for s in (job.get("skills") or [])][:12],
        is_hourly=job.get("is_hourly"),
        hourly_low=job.get("hourly_low"),
        hourly_high=job.get("hourly_high"),
        budget=job.get("budget"),
        connects_cost=job.get("connects_cost"),
        published_date=job.get("published_date"),
        country=job.get("country"),
        fit_coverage=getattr(fit, "coverage", None),
        risk_score=getattr(risk, "score", None),
        recommended_action=getattr(decision, "decision", None),
        recommended_price=getattr(quote, "amount_usd", None),
        thread_status=status_value,
        skill_overlap=overlap,
    )


@router.get("", response_model=JobPage)
async def list_jobs(
    request: Request,
    q: str | None = None,
    skills: list[str] | None = Query(default=None),
    min_hourly: float | None = None,
    limit: int = Query(25, ge=1, le=100),
    offset: int = Query(0, ge=0),
    analysed_only: bool = False,
    user: dict = CurrentUser,
) -> JobPage:
    """The job inbox, ranked by analysed fit first and skill overlap second."""
    from mcp_servers.marketplace import server as market
    from mcp_servers.portfolio import tools as portfolio

    try:
        page = await asyncio.wait_for(
            market.search_jobs(
                q=q, skills=skills, min_hourly=min_hourly, limit=limit, offset=offset
            ),
            timeout=SEARCH_TIMEOUT_S,
        )
    except Exception as exc:
        # A5/A6: the marketplace being down is a degraded UI, not a 500 page.
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE, f"marketplace unavailable: {exc}"
        ) from exc

    try:
        skills_result = await asyncio.wait_for(
            asyncio.to_thread(portfolio.list_skills, user["persona"]), timeout=10.0
        )
        own = {row["skill"].lower() for row in skills_result["skills"]}
    except Exception:
        own = set()

    jobs = list(page.get("jobs") or [])
    all_values = await asyncio.gather(
        *(_thread_values_capped(request, user["user_id"], str(j.get("job_id"))) for j in jobs)
    )
    summaries: list[JobSummary] = []
    for job, values in zip(jobs, all_values):
        if analysed_only and not values:
            continue
        summaries.append(_summary(job, values, _skill_overlap(job, own)))

    # Analysed jobs first, by coverage; then unanalysed by skill overlap. A user wants the
    # jobs the system has an opinion about at the top.
    summaries.sort(
        key=lambda s: (
            s.fit_coverage is None,
            -(s.fit_coverage or 0),
            -s.skill_overlap,
        )
    )

    return JobPage(
        total=int(page.get("total") or 0),
        limit=limit,
        offset=offset,
        jobs=summaries,
        marketplace=str(page.get("marketplace") or "gigsim"),
    )


@router.get("/{job_id}", response_model=JobDetail)
async def job_detail(job_id: str, request: Request, user: dict = CurrentUser) -> JobDetail:
    """Everything the job detail screen shows, including the evidence matrix."""
    from mcp_servers.marketplace import server as market

    try:
        job = await asyncio.wait_for(
            market.get_job(job_id), timeout=SEARCH_TIMEOUT_S
        )
    except Exception as exc:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE, f"could not load job: {exc}"
        ) from exc

    client: dict[str, Any] = {}
    if job.get("client_id"):
        try:
            client = await asyncio.wait_for(
                market.get_client_history(job["client_id"]), timeout=CLIENT_HISTORY_TIMEOUT_S
            )
        except Exception:
            # Client history is supplementary. A slow history service must not hold the
            # entire job detail screen open; render the post and label the history missing.
            client = {}

    values = await _thread_values_capped(request, user["user_id"], job_id)
    fit = values.get("fit_report")
    risk = values.get("risk_report")
    quote = values.get("price_quote")
    decision = values.get("decision")
    spec = values.get("job_spec")

    matrix = [
        RequirementEvidence(
            requirement=str(row.get("requirement") or ""),
            covered=bool(row.get("covered")),
            evidence=list(row.get("evidence") or []),
            overruled_by_model=row.get("overruled_by_model"),
        )
        for row in (getattr(fit, "requirement_matrix", []) or [])
    ]

    status_value = values.get("status")
    if values.get("_awaiting_approval"):
        status_value = "awaiting_approval"

    return JobDetail(
        job=job,
        client=client,
        analysed=bool(values),
        job_spec=spec.model_dump() if spec is not None else None,
        risk=risk.model_dump() if risk is not None else None,
        fit=fit.model_dump() if fit is not None else None,
        requirement_matrix=matrix,
        price=quote.model_dump() if quote is not None else None,
        decision=decision.model_dump() if decision is not None else None,
        thread_id=thread_id("bid", user["user_id"], job_id) if values else None,
        thread_status=status_value,
        injection_removed="[removed:" in str(values.get("job_text_sanitised") or ""),
    )
