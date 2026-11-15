"""Agency mode (X1): route a job to the team member whose portfolio actually covers it.

A single freelancer is the default. With `agency.enabled` set in `agent/config.yaml`, the
same system serves a small team: before any analysis runs, this node asks which member's
portfolio has evidence for the job's requirements, and the rest of the graph proceeds as
that member. The proposal is drafted from their corpus and in their promoted style, so
agency mode changes who is bidding rather than adding a layer of supervision.

Routing is done by retrieval rather than by a model, which is the point worth defending.
Requirement coverage is the measurement the bid decision already runs on, so routing on
coverage uses evidence the system has to gather anyway and leaves a number a person can
check: "routed to the data analyst, 5 of 6 requirements covered, next best 2 of 6". A
model asked "who should take this job?" would answer from the job's wording, which is how
a Python job reaches the member whose profile says Python most often rather than the one
with the matching work.

Three rules keep it honest:

* The requirements are decomposed once and every member is scored against the same list.
  Scoring each member on their own reading of the post would compare different questions.
* A win needs a margin. Two members within `agency.min_margin` of each other is not a
  routing decision, so the job goes to the configured default member and is flagged for a
  person. Coverage differences of one requirement are noise at this corpus size.
* Nobody clearing `agency.min_coverage` means the team cannot do this job. The job is
  routed to the default member and the decision node sees the low coverage and skips,
  rather than the best of three bad options looking like a choice.

The cost is one retrieval pass per member, with no extra model call. At three members and
hybrid retrieval that is about 10 ms per requirement per member; the cross-encoder would
make routing cost seconds, which is one more reason the shipped retrieval variant matters.
"""

from __future__ import annotations

from typing import Any

from langchain_core.runnables import RunnableConfig

from agent.nodes._common import app_config, node_trace, retrieval_config, step
from agent.state import GigBidState

DEFAULT_MIN_COVERAGE = 0.34
DEFAULT_MIN_MARGIN = 0.15


def agency_config(runnable_config: Any = None) -> dict[str, Any]:
    """The `agency` block, with defaults. Absent or disabled means single-member mode."""
    block = dict(app_config(runnable_config).get("agency") or {})
    block.setdefault("enabled", False)
    block.setdefault("members", [])
    block.setdefault("min_coverage", DEFAULT_MIN_COVERAGE)
    block.setdefault("min_margin", DEFAULT_MIN_MARGIN)
    return block


def score_members(
    job: dict[str, Any],
    members: list[str],
    *,
    config: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Coverage per member over one shared requirement list.

    Returned sorted by coverage, then by the number of distinct projects behind it, then
    by name so the order is stable. The tie-break on project count matters: two members
    covering four requirements each are not equally strong if one does it from a single
    project and the other from four.
    """
    from mcp_servers.portfolio.corpus import load_corpus
    from rag.variants.v1_decompose import decompose
    from rag.variants.v8_corrective import retrieve_with_report

    requirements = decompose(job)
    if not requirements:
        return []

    scored: list[dict[str, Any]] = []
    for member in members:
        try:
            items = load_corpus(member)
        except Exception as exc:
            scored.append(
                {
                    "member": member,
                    "coverage": 0.0,
                    "covered": [],
                    "missing": list(requirements),
                    "projects": 0,
                    "error": str(exc)[:160],
                }
            )
            continue

        covered: list[str] = []
        projects: set[str] = set()
        for requirement in requirements:
            report = retrieve_with_report(requirement, items, 3, config=config)
            if report["sufficient"]:
                covered.append(requirement)
                projects |= {h.project_id for h in report["hits"] if h.project_id}

        scored.append(
            {
                "member": member,
                "coverage": round(len(covered) / len(requirements), 3),
                "covered": covered,
                "missing": [r for r in requirements if r not in covered],
                "projects": len(projects),
            }
        )

    scored.sort(key=lambda row: (-row["coverage"], -row["projects"], row["member"]))
    return scored


def choose(
    scored: list[dict[str, Any]],
    *,
    default_member: str,
    min_coverage: float = DEFAULT_MIN_COVERAGE,
    min_margin: float = DEFAULT_MIN_MARGIN,
) -> dict[str, Any]:
    """Pick a member, or decline to and say why."""
    if not scored:
        return {
            "member": default_member,
            "routed": False,
            "reason": "no requirements could be extracted from the post",
        }

    best = scored[0]
    runner_up = scored[1] if len(scored) > 1 else None

    if best["coverage"] < min_coverage:
        return {
            "member": default_member,
            "routed": False,
            "reason": (
                f"no member covers enough of this job: best is {best['member']} at "
                f"{best['coverage']:.0%}, below the {min_coverage:.0%} floor"
            ),
            "best": best["member"],
            "best_coverage": best["coverage"],
        }

    if runner_up and (best["coverage"] - runner_up["coverage"]) < min_margin:
        return {
            "member": default_member,
            "routed": False,
            "reason": (
                f"{best['member']} at {best['coverage']:.0%} and {runner_up['member']} at "
                f"{runner_up['coverage']:.0%} are within {min_margin:.0%}; too close to call"
            ),
            "best": best["member"],
            "best_coverage": best["coverage"],
            "runner_up": runner_up["member"],
        }

    return {
        "member": best["member"],
        "routed": True,
        "reason": (
            f"{best['member']} covers {len(best['covered'])} of "
            f"{len(best['covered']) + len(best['missing'])} requirements from "
            f"{best['projects']} project(s)"
            + (
                f"; next best {runner_up['member']} at {runner_up['coverage']:.0%}"
                if runner_up
                else ""
            )
        ),
        "best_coverage": best["coverage"],
        "runner_up": runner_up["member"] if runner_up else None,
    }


async def route_member(
    state: GigBidState, config: RunnableConfig | None = None
) -> dict[str, Any]:
    """Set `persona` for the rest of the thread. A pass-through outside agency mode."""
    settings = agency_config(config)
    default_member = state.get("persona") or "ai_ml_engineer"

    if not settings.get("enabled"):
        return {
            "execution_trace": node_trace(
                "route_member",
                "decision",
                result_summary={"agency_mode": False, "member": default_member},
            )
        }

    members = [str(m) for m in (settings.get("members") or []) if m]
    if default_member not in members:
        members = [*members, default_member]

    job = dict(state.get("job") or {})
    job["description"] = state.get("job_text_sanitised") or job.get("description") or ""

    scored = score_members(job, members, config=retrieval_config(config))
    verdict = choose(
        scored,
        default_member=default_member,
        min_coverage=float(settings.get("min_coverage") or DEFAULT_MIN_COVERAGE),
        min_margin=float(settings.get("min_margin") or DEFAULT_MIN_MARGIN),
    )

    updates: dict[str, Any] = {
        "persona": verdict["member"],
        "freelancer_id": verdict["member"],
        "steps_used": step(state),
        "execution_trace": node_trace(
            "route_member",
            "decision",
            result_summary={
                "agency_mode": True,
                "member": verdict["member"],
                "routed": verdict["routed"],
                "why": verdict["reason"],
                "scores": [
                    {"member": row["member"], "coverage": row["coverage"]} for row in scored
                ],
            },
        ),
    }
    if not verdict["routed"]:
        # A routing decision nobody made is something a person should see, not a silent
        # fallback to whoever was configured first.
        updates["degraded"] = {"agency_routing": verdict["reason"]}
    return updates
