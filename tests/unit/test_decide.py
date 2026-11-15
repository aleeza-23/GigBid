"""The Connects reserve rule, added as a rehearsal of the change request.

A platform rule that introduces a cost or a cap is the change the brief warns about. This
one keeps a reserve rather than spending the balance to zero: the threshold lives in
`agent/config.yaml` and the guard reads it, so changing the number is a config edit.

See docs/CHANGE_REQUEST.md for the timings and what the exercise showed.
"""

from __future__ import annotations

from agent.decide import decide
from agent.handoffs import FitReport, PriceQuote, RiskReport


def good_fit() -> FitReport:
    return FitReport(score=0.95, coverage=0.95, evidence_sufficient=True)


def no_risk() -> RiskReport:
    return RiskReport(score=0.0, explanation="no signals")


def quote(connects_cost: int = 12) -> PriceQuote:
    return PriceQuote(
        amount_usd=60.0,
        rationale="market median plus client history",
        connects_cost=connects_cost,
        above_floor=True,
    )


def test_a_bid_that_would_breach_the_reserve_becomes_a_watch():
    verdict = decide(
        good_fit(),
        no_risk(),
        quote(connects_cost=12),
        {"balance": 16},
        {"connects_reserve": 8},
    )
    assert verdict.decision == "watch"
    assert verdict.deciding_factor == "connects_reserve"
    assert "reserve" in " ".join(verdict.reasons)


def test_a_bid_that_leaves_the_reserve_intact_still_bids():
    verdict = decide(
        good_fit(),
        no_risk(),
        quote(connects_cost=12),
        {"balance": 40},
        {"connects_reserve": 8},
    )
    assert verdict.decision == "bid"


def test_the_reserve_can_be_switched_off_from_config():
    """A reserve of zero means spend the balance, which is the previous behaviour."""
    verdict = decide(
        good_fit(),
        no_risk(),
        quote(connects_cost=12),
        {"balance": 12},
        {"connects_reserve": 0},
    )
    assert verdict.decision == "bid"


def test_running_out_entirely_is_still_reported_as_the_budget_not_the_reserve():
    """The more specific reason wins: no budget at all is not the same as a reserve."""
    verdict = decide(
        good_fit(),
        no_risk(),
        quote(connects_cost=12),
        {"balance": 4},
        {"connects_reserve": 8},
    )
    assert verdict.decision == "watch"
    assert verdict.deciding_factor == "connects_budget"
