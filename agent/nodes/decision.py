"""The decision node: skip, watch or bid, after the three parallel branches converge.

Entirely rule-driven, and the one place in the graph where that is non-negotiable. This is
the node whose accuracy is measured (0.783 bid/skip over 198 dev jobs, 0.857 scam F1,
`evals/results/decision_eval.json`), the node whose thresholds the change request will
move, and the node whose reasoning a user reads when they ask why a job was skipped. All
three of those want a function with thresholds in config, not a model.

The ordering inside `decide()` matters more than it looks: risk is checked before fit, so
a scam that happens to match the portfolio well is reported as a scam rather than as a
coverage number. The reason the user sees is the reason that actually settled it.
"""

from __future__ import annotations

from typing import Any

from langchain_core.runnables import RunnableConfig

from agent.decide import decide
from agent.handoffs import BidDecision, FitReport, RiskReport
from agent.nodes._common import check_budget, node_trace, step, thresholds
from agent.state import GigBidState


async def decision(state: GigBidState, config: RunnableConfig | None = None) -> dict[str, Any]:
    # C4: a thread over budget stops here with a visible state rather than draining the
    # rest of the graph and producing a half-finished proposal.
    over = check_budget(state, config)
    if over:
        return {
            "status": "needs_attention",
            "budget_exceeded": over,
            "steps_used": step(state),
            "execution_trace": node_trace("decision", "decision", result_summary={"halted": over}),
        }

    fit = state.get("fit_report") or FitReport(score=0.0, coverage=0.0)
    risk = state.get("risk_report") or RiskReport(
        score=0.5, explanation="no risk assessment completed"
    )
    quote = state.get("price_quote")
    limits = thresholds(config)

    # A thread that reached the decision with no requirements extracted has not been
    # analysed, and treating that as a confident skip would hide a failure as a decision.
    if not (state.get("job_spec") and getattr(state["job_spec"], "requirements", None)):
        return {
            "decision": BidDecision(
                decision="watch",
                reasons=["the job could not be analysed into requirements"],
                deciding_factor="analysis_failed",
                fit_score=fit.score,
                risk_score=risk.score,
                confidence=0.2,
            ),
            "status": "needs_attention",
            "steps_used": step(state),
            "execution_trace": node_trace(
                "decision", "decision", result_summary={"decision": "watch", "reason": "no spec"}
            ),
        }

    verdict = decide(fit, risk, quote, state.get("connects"), limits)

    return {
        "decision": verdict,
        "status": "skipped" if verdict.decision == "skip" else state.get("status", "running"),
        "steps_used": step(state),
        "execution_trace": node_trace(
            "decision",
            "decision",
            result_summary={
                "decision": verdict.decision,
                "deciding_factor": verdict.deciding_factor,
                "reasons": verdict.reasons[:3],
                "fit": fit.coverage,
                "risk": risk.score,
                "decided_by": "rule",
            },
        ),
    }


def route_after_decision(state: GigBidState) -> str:
    """Conditional edge: where a thread goes once the decision is made.

    `watch` ends the run like a skip does, but keeps the thread visible on the pipeline
    board for a human to pick up. Neither path reaches the Writer, so neither spends
    Writer tokens on a job nobody will bid on.
    """
    if state.get("status") in ("needs_attention", "dead_letter"):
        return "halt"
    decision_obj = state.get("decision")
    verdict = getattr(decision_obj, "decision", "skip")
    return "bid" if verdict == "bid" else "halt"
