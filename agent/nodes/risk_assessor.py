"""Risk Assessor: score client quality and scam signals. Marketplace read only (A6).

Rule-driven rather than model-driven, and that is a measured choice rather than a
shortcut. The heuristics in `agent/decide.py` reach 0.857 F1 on scam detection over the
dev set with perfect recall on all 24 planted scams
(`evals/results/decision_eval.json`). A small model asked to judge the same evidence is
slower, costs tokens, and cannot be regression-tested.

The model does the one thing the rule cannot: write the explanation a human reads on the
approval screen. The rule keeps the score. If the model fails, the rule's own flags are
shown instead, so the node never depends on a provider being up.
"""

from __future__ import annotations

from typing import Any

from langchain_core.runnables import RunnableConfig

from agent.decide import assess_risk
from agent.nodes._common import call_model, compact, node_trace, step, thresholds
from agent.state import GigBidState


async def risk_assessor(
    state: GigBidState, config: RunnableConfig | None = None
) -> dict[str, Any]:
    job = dict(state.get("job") or {})
    # Judge the sanitised text. An injected instruction is itself a risk signal, which the
    # Scout already recorded, but it must not reach a model through this node either.
    job["description"] = state.get("job_text_sanitised") or job.get("description") or ""
    client = state.get("client") or {}

    report = assess_risk(job, client, thresholds(config))
    tokens = 0
    meta: dict[str, Any] = {}

    # Only ask a model to explain a finding worth explaining.
    if report.flags:
        parsed, meta = call_model(
            "risk_assessor",
            prompt_values={
                "client_history": compact(client, 900),
                "job_text": (state.get("job_text_sanitised") or "")[:2500],
            },
            trace_id=state.get("trace_id"),
            job_id=state.get("job_id"),
        )
        tokens = meta.get("tokens") or 0
        if isinstance(parsed, dict) and parsed.get("explanation"):
            # The rule owns the score; the model only phrases it. Letting the model
            # overwrite the score would discard the one number that is regression-tested.
            report.explanation = str(parsed["explanation"])[:400]

    return {
        "risk_report": report,
        "analyses_complete": ["risk_assessor"],
        "steps_used": step(state),
        "tokens_used": tokens,
        "execution_trace": node_trace(
            "risk_assessor",
            "decision",
            meta,
            result_summary={
                "score": report.score,
                "is_likely_scam": report.is_likely_scam,
                "flags": report.flags[:4],
                "scored_by": "rule",
            },
        ),
    }
