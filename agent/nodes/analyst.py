"""Job Analyst: extract a JobSpec from the sanitised post. No tools (A6).

Model-driven, because no rule turns prose into a requirement list. It falls back to the
same rule-based extraction the Part B evaluation used, so a provider outage degrades the
spec rather than failing the thread.

Holds no tools by design: everything it needs is already in state, and a node with search
tools is a node that can be talked into using them by the text it is reading.
"""

from __future__ import annotations

from typing import Any

from langchain_core.runnables import RunnableConfig

from agent.handoffs import JobSpec
from agent.nodes._common import call_model_async, node_trace, step
from agent.state import GigBidState


def _rule_based(job: dict[str, Any], text: str) -> dict[str, Any]:
    from rag.variants.v1_decompose import decompose

    return {
        "requirements": decompose({**job, "description": text}),
        "required_skills": [str(s) for s in (job.get("skills") or [])],
        "deliverables": [],
        "screening_questions": list(job.get("screening_questions") or []),
        "hidden_needs": [],
    }


async def analyst(state: GigBidState, config: RunnableConfig | None = None) -> dict[str, Any]:
    job = state.get("job") or {}
    text = state.get("job_text_sanitised") or ""

    parsed, meta = await call_model_async(
        "analyst",
        prompt_values={"job_text": text[:4000]},
        trace_id=state.get("trace_id"),
        job_id=state.get("job_id"),
    )
    source = "model"
    if not isinstance(parsed, dict) or not parsed.get("requirements"):
        parsed = _rule_based(job, text)
        source = "rule_fallback"

    spec = JobSpec(
        job_id=state.get("job_id") or job.get("job_id") or "",
        title=str(job.get("title") or "")[:300],
        deliverables=[str(d) for d in (parsed.get("deliverables") or [])][:12],
        required_skills=[str(s) for s in (parsed.get("required_skills") or [])][:20],
        requirements=[str(r) for r in (parsed.get("requirements") or [])][:8],
        screening_questions=[str(q) for q in (parsed.get("screening_questions") or [])][:10],
        hidden_needs=[str(h) for h in (parsed.get("hidden_needs") or [])][:6],
        budget_min=job.get("hourly_low") or None,
        budget_max=job.get("hourly_high") or job.get("budget") or None,
        is_hourly=job.get("is_hourly"),
        timeline_days=parsed.get("timeline_days"),
        injection_detected="[removed:" in text,
    )

    return {
        "job_spec": spec,
        "analyses_complete": ["analyst"],
        "steps_used": step(state),
        "tokens_used": meta.get("tokens") or 0,
        "execution_trace": node_trace(
            "analyst",
            "model",
            meta,
            result_summary={"requirements": len(spec.requirements), "source": source},
        ),
    }
