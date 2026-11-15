"""Risk, fit, pricing and the bid/watch/skip decision, as pure functions.

Separated from the graph on purpose. Part B has to *measure* these decisions against gold
labels, and Part C has to *make* them inside nodes. A pure function given explicit inputs
can be evaluated 198 times in a second without a checkpointer, a model or a marketplace,
and Part C's nodes then call the same code the numbers were measured on.

Every threshold comes from `agent/config.yaml`. The change request on 2 November introduces
a new rule or cost affecting bidding, and the test of this design is whether absorbing it
means editing config rather than hunting through nodes.

The scam heuristics deserve a note. GigSim hides `is_scam_account`, so these infer it from
what a marketplace really shows: many posts and no hires, no payment verification, a budget
far below market, a request for free work, a push off-platform. Each signal is weak alone
and they are weighted and summed, which is why the output is a score with flags rather
than a boolean.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

from agent.handoffs import BidDecision, FitReport, PriceQuote, RiskReport

CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"


def load_thresholds(config: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = config or yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    return cfg.get("thresholds") or {}


# ------------------------------------------------------------------ injection (E4)

# Patterns that mean "someone is talking to the agent, not to the freelancer". Job posts
# are data; an imperative aimed at a reader of the post is an attack, not a requirement.
INJECTION_PATTERNS = [
    # Classic overrides.
    r"(?i)ignore (?:all |any )?(?:previous|prior|above|earlier) instructions",
    r"(?i)disregard (?:the |all )?(?:system|previous|above|prior)",
    r"(?i)you are now\b",
    r"(?i)new instructions?\s*:",
    r"(?i)forget (?:everything|all|your)",
    r"(?i)do not (?:follow|obey) (?:your|the) (?:rules|instructions|guidelines)",
    r"(?i)respond (?:only )?with the (?:word|phrase|text)\b",
    r"(?i)print (?:the|your) (?:system )?(?:prompt|instructions)",
    r"(?i)reveal your (?:prompt|instructions|rules)",
    r"(?i)as an ai (?:language )?model",
    r"(?i)override (?:your|all) (?:safety|previous)",
    #
    # The generic signature, and the one the first version of this list missed: a job post
    # is addressed to a freelancer, so any sentence addressed to an *automated reader* is
    # an attack by construction, whatever it goes on to ask for. Four of the five planted
    # attack types in the dev set are this shape and none of the patterns above caught
    # them, which showed up as 25% detection.
    r"(?i)(?:note|message|instruction)s? (?:to|for) (?:any )?(?:ai|automated|llm|bot|assistant)",
    r"(?i)\b(?:ai|llm) (?:assistant|agent|tool)s?\s*[:,]",
    r"(?i)\bassistant instruction",
    r"(?i)\bfor automated tools?\b",
    r"(?i)\bif you are (?:an? )?(?:ai|bot|language model|automated)",
    r"(?i)\bany ai (?:assistant|agent|reading)",
    #
    # Fake system framing, including the bracketed form that is not a bare "[system]".
    r"(?i)\[\s*(?:system|admin|instruction)[^\]]{0,24}\]",
    r"(?i)system\s*(?:prompt|message)\s*[:\]]",
    r"(?i)policy updated\s*:",
    #
    # Instructions hidden in markup a human reader never sees.
    r"(?i)<!--[^>]{0,400}?(?:\bai\b|agent|assistant|instruction|fit score|risk score)[^>]{0,400}?-->",
    #
    # Attempts to drive our own scoring or to dictate claims. These name the internals,
    # which no genuine client would.
    r"(?i)set (?:the )?(?:fit|risk|confidence) score",
    r"(?i)state that you have \d+\+? years",
    r"(?i)regardless of (?:the |your )?portfolio",
    r"(?i)begin your proposal with the (?:word|phrase)",
    r"(?i)include your (?:personal )?(?:email|phone)",
]

_COMPILED_INJECTION = [re.compile(p) for p in INJECTION_PATTERNS]


def detect_injection(text: str) -> tuple[bool, list[str]]:
    """Find instruction-shaped text in untrusted input.

    Returns the matched spans so E4's attack-success-rate metric has something concrete to
    count, and so the UI can show the user exactly what was ignored.
    """
    found = [m.group(0) for pattern in _COMPILED_INJECTION for m in pattern.finditer(text or "")]
    return bool(found), found


def strip_injection(text: str) -> str:
    """Neutralise instruction-shaped spans before the text reaches any prompt."""
    cleaned = text or ""
    for pattern in _COMPILED_INJECTION:
        cleaned = pattern.sub("[removed: instruction-like text]", cleaned)
    return cleaned


# ------------------------------------------------------------------------- risk

_OFF_PLATFORM = re.compile(
    r"(?i)\b(whatsapp|telegram|skype|wechat|signal|email me|contact me (?:at|on)|"
    r"message me on|off[- ]platform|outside (?:of )?(?:the )?(?:upwork|platform)|"
    r"direct(?:ly)? pay|pay(?:pal|ment) outside|western union|crypto|bitcoin|usdt|"
    r"takes too much commission|avoid fees)\b"
)
_FREE_WORK = re.compile(
    r"(?i)\b(free (?:sample|trial|test|work|demo)|unpaid (?:test|trial|sample|samples)|"
    r"no pay(?:ment)? (?:for|until)|prove yourself first|small test first at no|"
    r"complete this task to be considered|as a free test|"
    r"must complete the full project|only unpaid samples)\b"
)

# The scam shapes the off-platform and free-work patterns miss entirely. Each of these is
# a known fraud pattern rather than a risk signal, so they are scored near-certain below.
_ADVANCE_FEE = re.compile(
    r"(?i)\b(registration fee|one[- ]time (?:\$\s*\d+\s*)?fee|pay (?:a |an )?(?:\$\s*\d+\s*)?"
    r"(?:fee|deposit) (?:to|for) (?:get |gain )?access|access to the project portal|"
    r"you must pay|training fee|activation fee)\b"
)
_FAKE_PAYMENT = re.compile(
    r"(?i)\b(mail you a (?:cheque|check)|we will send (?:you )?a (?:cheque|check)|"
    r"deposit it(?:,| and)|keep \$?\d+ for your time|overpay|money order|"
    r"buy the (?:software )?licences? (?:we|from))\b"
)
_CREDENTIAL_HARVEST = re.compile(
    r"(?i)\b(bank login|bank(?:ing)? (?:details|credentials)|send your (?:password|pin)|"
    r"photo of your (?:national )?id|national id|passport (?:photo|scan|copy)|"
    r"social security|routing number)\b"
)


def assess_risk(
    job: dict[str, Any],
    client: dict[str, Any] | None = None,
    thresholds: dict[str, Any] | None = None,
) -> RiskReport:
    """Score a client and a post for scam risk, from what a marketplace really exposes."""
    th = thresholds or load_thresholds()
    client = client or {}
    text = f"{job.get('title') or ''}\n{job.get('description') or ''}"

    flags: list[str] = []
    score = 0.0

    posted = int(client.get("jobs_posted") or 0)
    hires = int(client.get("hires") or 0)
    hire_rate = client.get("hire_rate")
    if hire_rate is None and posted:
        hire_rate = hires / posted

    # The strongest single signal in this corpus: a client who posts repeatedly and hires
    # nobody is harvesting proposals, not buying work.
    min_posts = int(th.get("scam_min_posts_for_signal", 3))
    if posted >= min_posts and (hire_rate or 0) <= float(th.get("scam_hire_rate_below", 0.05)):
        score += 0.35
        flags.append(f"{posted} jobs posted, {hires} hires ({(hire_rate or 0):.0%} hire rate)")

    verified = client.get("payment_verified")
    if verified is False:
        score += 0.25
        flags.append("payment method not verified")

    # A brand-new account that has never paid anyone. Weaker than the posting signal
    # because plenty of legitimate clients are hiring for the first time, which is why
    # it needs corroboration from the post text to reach the threshold.
    if (
        client.get("total_spent_usd") is not None
        and float(client["total_spent_usd"]) == 0
        and hires == 0
        and posted >= 1
    ):
        score += 0.20
        flags.append("has never hired or spent anything")

    # Weighted so that one near-certain signal alone clears the scam threshold. An earlier
    # calibration gave off-platform payment 0.35 and unverified payment 0.25, summing to
    # 0.60 against a 0.70 threshold, so a post asking to be paid in USDT via Telegram was
    # not flagged as a scam. Asking to be paid off-platform in crypto is not a 0.35
    # concern; it is the thing itself.
    off_platform = bool(_OFF_PLATFORM.search(text))
    if off_platform:
        score += 0.55
        flags.append("asks to move the conversation or payment off-platform")

    free_work = bool(_FREE_WORK.search(text))
    if free_work:
        score += 0.55
        flags.append("asks for free or unpaid work")

    if _ADVANCE_FEE.search(text):
        score += 0.75
        flags.append("asks the freelancer to pay a fee: advance-fee fraud")

    if _FAKE_PAYMENT.search(text):
        score += 0.75
        flags.append("cheque or overpayment scheme")

    if _CREDENTIAL_HARVEST.search(text):
        score += 0.80
        flags.append("asks for bank credentials or identity documents")

    injected, spans = detect_injection(text)
    if injected:
        # Not a scam signal on its own, but a post carrying an injection is manipulating
        # automated readers, which correlates strongly here.
        score += 0.20
        flags.append(f"post contains instruction-like text: {spans[0][:60]!r}")

    # A budget far below the market for the work is the classic underpay-or-scam shape.
    budget = job.get("budget") or job.get("hourly_high")
    if budget and float(budget) > 0 and float(budget) < 15 and not job.get("is_hourly"):
        score += 0.10
        flags.append(f"budget ${float(budget):.0f} is far below market for this work")

    score = min(1.0, score)
    is_scam = score >= float(th.get("maximum_risk_score", 0.70))

    return RiskReport(
        score=round(score, 3),
        flags=flags,
        explanation=(
            "; ".join(flags) if flags else "no risk signals in the client history or the post"
        ),
        is_likely_scam=is_scam,
        client_hire_rate=round(hire_rate, 3) if hire_rate is not None else None,
        client_payment_verified=verified,
        off_platform_contact_requested=off_platform,
        free_work_requested=free_work,
    )


# -------------------------------------------------------------------------- fit


def assess_fit(
    requirements: list[str],
    hits_per_requirement: dict[str, list[dict[str, Any]]],
    *,
    min_score: float = 2.0,
    evidence_sufficient: bool = True,
    covered_flags: dict[str, bool] | None = None,
) -> FitReport:
    """Turn per-requirement retrieval into a coverage figure and a matrix.

    Coverage, not mean similarity: the question the decision rests on is "how many of the
    things they asked for can I prove?", and an average hides one uncovered requirement
    behind four strong ones.

    Two ways to decide coverage, and the distinction matters:

    * `min_score` thresholds the top hit's score. Simple, and what the Part B evaluation
      measured, but **scale-dependent**: BM25 scores run 5 to 13 while reciprocal rank
      fusion scores run 0.01 to 0.05, so a threshold tuned for one silently rejects
      everything from the other. That bug produced 0% coverage on every job in the graph
      until it was found.
    * `covered_flags` supplies the verdict per requirement, from something scale-free such
      as V8's term-coverage grading. The graph uses this; the evaluation keeps the
      threshold so its numbers stay comparable.
    """
    matrix: list[dict[str, Any]] = []
    covered: list[str] = []
    missing: list[str] = []
    evidence_ids: list[str] = []

    for requirement in requirements:
        hits = hits_per_requirement.get(requirement) or []
        best = hits[0] if hits else None
        if covered_flags is not None:
            is_covered = bool(best) and bool(covered_flags.get(requirement))
        else:
            is_covered = bool(best) and float(best.get("score") or 0) >= min_score
        (covered if is_covered else missing).append(requirement)
        for hit in hits[:3]:
            if hit.get("id") and hit["id"] not in evidence_ids:
                evidence_ids.append(hit["id"])
        matrix.append(
            {
                "requirement": requirement,
                "covered": is_covered,
                "evidence": [
                    {
                        "id": h.get("id"),
                        "citation": h.get("citation"),
                        "score": h.get("score"),
                        "project": h.get("project"),
                        "type": h.get("type"),
                    }
                    for h in hits[:3]
                ],
            }
        )

    coverage = len(covered) / len(requirements) if requirements else 0.0
    return FitReport(
        score=round(coverage, 3),
        coverage=round(coverage, 3),
        matched_skills=covered,
        missing_skills=missing,
        evidence_ids=evidence_ids,
        requirement_matrix=matrix,
        evidence_sufficient=evidence_sufficient,
    )


# ----------------------------------------------------------------------- pricing


def price_bid(
    job: dict[str, Any],
    rate_stats: dict[str, Any] | None,
    profile: dict[str, Any],
    client: dict[str, Any] | None = None,
    thresholds: dict[str, Any] | None = None,
) -> PriceQuote:
    """Propose a bid from market statistics, client spend and the user's floor.

    The floor wins. A rate below what the freelancer will work for is not a cheap bid, it
    is a bid they should not win, so `respect_profile_floor` is checked before anything
    else is considered.
    """
    th = thresholds or load_thresholds()
    stats = rate_stats or {}
    earned = stats.get("earned_hourly_usd") or {}

    floor = float(profile.get("hourly_rate_usd") or 0)
    min_project = float(profile.get("minimum_project_budget_usd") or 0)
    market_p50 = earned.get("p50")
    samples = earned.get("n")
    spend_per_hire = (client or {}).get("spend_per_hire")

    is_hourly = bool(job.get("is_hourly", True))
    reasons: list[str] = []

    # What the client actually advertised. This is the binding constraint and the thing
    # an earlier version of this function ignored, which produced $45/hr bids on jobs
    # posted at $20-30 and put 90% of quotes outside the fair band.
    if is_hourly:
        posted_low = float(job.get("hourly_low") or 0) or None
        posted_high = float(job.get("hourly_high") or 0) or None
        anchor = floor
    else:
        posted_low = posted_high = float(job.get("budget") or 0) or None
        anchor = min_project or floor

    if market_p50:
        candidate = float(market_p50)
        reasons.append(f"market p50 ${candidate:.0f} from {samples or 0} contracts")
    else:
        candidate = anchor
        reasons.append("no market statistics; anchoring on the profile")

    if spend_per_hire and float(spend_per_hire) > 1000:
        candidate *= 1.10
        reasons.append(f"client averages ${float(spend_per_hire):.0f} per hire, so +10%")

    # A bid above what they posted is not a bid, it is a decline with extra steps.
    if posted_high and candidate > posted_high:
        candidate = posted_high
        reasons.append(f"capped at the posted ceiling of ${posted_high:.0f}")
    if posted_low and candidate < posted_low:
        candidate = posted_low
        reasons.append(f"raised to the posted floor of ${posted_low:.0f}")

    # The floor is the walk-away line. If the client's ceiling sits below it, there is no
    # price that works: report that rather than bidding below the floor or above the post.
    respect_floor = bool(th.get("respect_profile_floor", True))
    below_floor = False
    if is_hourly and floor:
        if posted_high and posted_high < floor:
            below_floor = True
            candidate = posted_high
            reasons.append(
                f"the posted ceiling ${posted_high:.0f} is under the ${floor:.0f} floor"
            )
        elif respect_floor and candidate < floor:
            candidate = min(floor, posted_high) if posted_high else floor
            reasons.append(f"raised toward the ${floor:.0f} floor")
            below_floor = candidate < floor
    elif not is_hourly and min_project and candidate < min_project:
        below_floor = not (posted_high and posted_high >= min_project)
        candidate = max(candidate, min(min_project, posted_high or min_project))
        reasons.append(f"minimum project budget ${min_project:.0f}")

    connects_cost = job.get("connects_cost")
    return PriceQuote(
        amount_usd=round(max(candidate, 1.0), 2),
        type="hourly" if is_hourly else "fixed",
        rationale="; ".join(reasons),
        profile_floor_usd=(floor if is_hourly else min_project) or None,
        market_p50_usd=float(market_p50) if market_p50 else None,
        market_sample_size=int(samples) if samples else None,
        client_spend_per_hire_usd=float(spend_per_hire) if spend_per_hire else None,
        above_floor=not below_floor,
        connects_cost=int(connects_cost) if connects_cost is not None else None,
    )


# ---------------------------------------------------------------------- decision


def decide(
    fit: FitReport,
    risk: RiskReport,
    price: PriceQuote | None = None,
    connects: dict[str, Any] | None = None,
    thresholds: dict[str, Any] | None = None,
) -> BidDecision:
    """Choose bid, watch or skip.

    Ordered so that the most decisive reason is the one reported. Risk first: a scam is a
    skip no matter how well the portfolio fits, and saying "skipped: 20 posts, 0 hires"
    is more useful than "skipped: coverage 45%" when both are true.

    `watch` exists for the honest middle: a job worth a human glance but not worth
    Connects automatically. A per-user autonomy setting may auto-skip low fit, but
    nothing here ever auto-sends.
    """
    th = thresholds or load_thresholds()
    min_fit = float(th.get("minimum_fit_score", 0.60))
    max_risk = float(th.get("maximum_risk_score", 0.70))
    min_coverage = float(th.get("minimum_requirement_coverage", 0.50))

    reasons: list[str] = []

    if risk.score >= max_risk or risk.is_likely_scam:
        return BidDecision(
            decision="skip",
            reasons=[f"risk score {risk.score:.2f} at or above the {max_risk:.2f} limit"]
            + risk.flags[:3],
            deciding_factor="risk",
            fit_score=fit.score,
            risk_score=risk.score,
            confidence=min(1.0, 0.5 + risk.score / 2),
        )

    if risk.free_work_requested:
        return BidDecision(
            decision="skip",
            reasons=["the post asks for free or unpaid work, which the policy pack forbids"],
            deciding_factor="free_work",
            fit_score=fit.score,
            risk_score=risk.score,
            confidence=0.9,
        )

    if not fit.evidence_sufficient:
        # V8's corrective loop decided the evidence is too thin to write a grounded
        # proposal from. Bidding anyway is how unsupported claims get written.
        return BidDecision(
            decision="skip",
            reasons=["the retrieved evidence is too thin to support a grounded proposal"]
            + [f"uncovered: {', '.join(fit.missing_skills[:4])}" if fit.missing_skills else ""],
            deciding_factor="evidence_insufficient",
            fit_score=fit.score,
            risk_score=risk.score,
            confidence=0.75,
        )

    if fit.coverage < min_coverage:
        return BidDecision(
            decision="skip",
            reasons=[
                f"requirement coverage {fit.coverage:.0%} below the {min_coverage:.0%} threshold",
                f"no evidence for: {', '.join(fit.missing_skills[:4])}"
                if fit.missing_skills
                else "",
            ],
            deciding_factor="coverage",
            fit_score=fit.score,
            risk_score=risk.score,
            confidence=0.8,
        )

    if price and not price.above_floor:
        # Name the floor correctly. An hourly rate and a minimum project budget are
        # different numbers, and reporting "$200/hr floor" for a project minimum makes a
        # correct skip look like a bug.
        floor_label = (
            f"${price.profile_floor_usd:.0f}/hr rate floor"
            if price.type == "hourly"
            else f"${price.profile_floor_usd:.0f} minimum project budget"
        )
        return BidDecision(
            decision="skip",
            reasons=[f"the most this job would pay is below the {floor_label}"],
            deciding_factor="below_floor",
            fit_score=fit.score,
            risk_score=risk.score,
            confidence=0.85,
        )

    # Connects are a budget, not an afterthought: a job worth bidding on with no budget
    # left is a watch, so the user can decide whether to raise the cap.
    if connects:
        remaining = int(connects.get("remaining_this_week", connects.get("balance", 0)) or 0)
        cost = int((price.connects_cost if price else 0) or 0)
        if cost and remaining < cost:
            return BidDecision(
                decision="watch",
                reasons=[
                    f"good fit, but {cost} Connects needed and only {remaining} left this week"
                ],
                deciding_factor="connects_budget",
                fit_score=fit.score,
                risk_score=risk.score,
                confidence=0.7,
            )
        # A reserve, so a good week of bidding cannot leave the user unable to answer the
        # one job that matters on Friday. Spending to zero is a decision for a person, not
        # a side effect of the agent finding enough work it liked.
        reserve = int(th.get("connects_reserve", 0) or 0)
        if cost and reserve and remaining - cost < reserve:
            return BidDecision(
                decision="watch",
                reasons=[
                    f"good fit, but bidding would leave {remaining - cost} Connects, "
                    f"under the {reserve} held in reserve"
                ],
                deciding_factor="connects_reserve",
                fit_score=fit.score,
                risk_score=risk.score,
                confidence=0.7,
            )

    if fit.score < min_fit:
        reasons.append(f"fit {fit.score:.0%} below the {min_fit:.0%} bar for an automatic bid")
        return BidDecision(
            decision="watch",
            reasons=reasons,
            deciding_factor="fit_borderline",
            fit_score=fit.score,
            risk_score=risk.score,
            confidence=0.6,
        )

    reasons.append(f"{fit.coverage:.0%} requirement coverage with cited evidence")
    if risk.flags:
        reasons.append(f"risk {risk.score:.2f}, acceptable: {risk.flags[0]}")
    else:
        reasons.append("no risk signals")
    return BidDecision(
        decision="bid",
        reasons=reasons,
        deciding_factor="fit",
        fit_score=fit.score,
        risk_score=risk.score,
        confidence=min(1.0, 0.5 + fit.coverage / 2),
    )
