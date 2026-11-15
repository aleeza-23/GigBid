"""Pricing Strategist: propose a bid and decide whether the Connects are worth it.

Marketplace read only (A6): market statistics, client spend, Connects balance.

Rule-driven. Part B measured this: pricing inside the fair band went from 0.100 to 0.752
once the logic stopped ignoring the client's posted range
(`evals/results/decision_eval.json`). The bug worth remembering is that an earlier version
priced from the market median and the freelancer's floor alone, and bid $45/hr on jobs
advertising $20-30. The floor is a walk-away line, not an opening offer.

The model is not used at all here. There is nothing for it to add to a clamp between three
numbers, and a model that occasionally returns $4,500 instead of $45 is a worse trade than
no model.
"""

from __future__ import annotations

from typing import Any

from langchain_core.runnables import RunnableConfig

from agent.decide import price_bid
from agent.nodes._common import node_trace, step, thresholds
from agent.state import GigBidState

DOMAIN = {
    "ai_ml_engineer": "ai_ml",
    "fullstack_web_developer": "fullstack_web",
    "data_analyst": "data_analysis",
}


async def pricing(state: GigBidState, config: RunnableConfig | None = None) -> dict[str, Any]:
    from mcp_servers.marketplace import server as market
    from mcp_servers.portfolio import corpus

    job = state.get("job") or {}
    persona = state.get("persona") or "ai_ml_engineer"
    client = state.get("client") or {}
    degraded: dict[str, Any] = {}

    profile = state.get("profile") or {}
    if not profile:
        try:
            profile = corpus.load_profile(persona)
        except Exception as exc:
            degraded["portfolio"] = str(exc)[:200]
            profile = {}

    rate_stats: dict[str, Any] = {}
    try:
        rate_stats = await market.get_rate_stats(domain=DOMAIN.get(persona))
    except Exception as exc:
        # No market anchor: price from the floor and the posted range, and say so. The
        # rationale will record that the market figure was unavailable rather than
        # silently pricing as if the median were zero.
        degraded["rate_stats"] = str(exc)[:200]

    connects = state.get("connects") or {}
    if not connects and state.get("freelancer_id"):
        try:
            connects = await market.get_connects_balance(state["freelancer_id"])
        except Exception as exc:
            degraded["connects"] = str(exc)[:200]

    quote = price_bid(job, rate_stats, profile, client, thresholds(config))

    # Connects are a budget, not a rounding error: a good job we cannot afford this week
    # is a decision for the user, which the decision node turns into `watch`.
    cost = quote.connects_cost or 0
    remaining = int(connects.get("remaining_this_week", connects.get("balance", 0)) or 0)
    quote.connects_worth_it = bool(cost == 0 or remaining >= cost)

    return {
        "price_quote": quote,
        "profile": profile,
        "connects": connects,
        "degraded": degraded,
        "steps_used": step(state),
        "execution_trace": node_trace(
            "pricing",
            "decision",
            tool="get_rate_stats",
            result_summary={
                "amount_usd": quote.amount_usd,
                "type": quote.type,
                "above_floor": quote.above_floor,
                "connects_cost": cost,
                "connects_remaining": remaining,
                "rationale": quote.rationale[:160],
                "priced_by": "rule",
            },
        ),
    }
