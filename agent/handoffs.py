"""Typed hand-offs between agents (C3).

Agents exchange these objects, not free text, for anything a later step decides on. The
reason is specific: a decision node parsing "the fit looks pretty good" out of prose is a
decision node that fails silently when the wording drifts. A `FitReport` with
`score: 0.42` either validates or raises.

All eight models the assignment names are here. Each carries the *reasons* behind its
numbers as well as the numbers, because the approval screen (C5) has to show a human why
a recommendation was made, and because a trace with scores but no reasoning cannot answer
viva question 4 — "walk through which node got this wrong".
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Decision = Literal["bid", "watch", "skip"]
ComplianceStatus = Literal["pass", "fixed", "blocked"]
CriticVerdict = Literal["approve", "revise"]


class JobSpec(BaseModel):
    """What the Job Analyst extracts from an untrusted job post."""

    model_config = ConfigDict(extra="forbid")

    job_id: str
    title: str
    deliverables: list[str] = Field(default_factory=list)
    required_skills: list[str] = Field(default_factory=list)
    requirements: list[str] = Field(
        default_factory=list,
        description="Distinct capabilities, one per entry. What V1 retrieves against.",
    )
    screening_questions: list[str] = Field(default_factory=list)
    hidden_needs: list[str] = Field(
        default_factory=list,
        description="What the client needs but did not ask for, inferred from the post.",
    )
    budget_min: float | None = None
    budget_max: float | None = None
    is_hourly: bool | None = None
    timeline_days: int | None = None
    injection_detected: bool = Field(
        default=False,
        description="The post contained text shaped like an instruction to the agent.",
    )
    injected_text: list[str] = Field(
        default_factory=list, description="What was stripped, kept for the trace and E4's metric."
    )


class RiskReport(BaseModel):
    """The Risk Assessor's read on a client and a post."""

    model_config = ConfigDict(extra="forbid")

    score: float = Field(ge=0, le=1, description="0 is safe, 1 is almost certainly a scam.")
    flags: list[str] = Field(default_factory=list)
    explanation: str
    is_likely_scam: bool = False
    client_hire_rate: float | None = None
    client_payment_verified: bool | None = None
    off_platform_contact_requested: bool = False
    free_work_requested: bool = False


class FitReport(BaseModel):
    """The Fit Matcher's requirement-to-evidence mapping."""

    model_config = ConfigDict(extra="forbid")

    score: float = Field(ge=0, le=1)
    coverage: float = Field(
        ge=0, le=1, description="Fraction of requirements with supporting evidence."
    )
    matched_skills: list[str] = Field(default_factory=list)
    missing_skills: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    requirement_matrix: list[dict] = Field(
        default_factory=list,
        description="Per requirement: covered, and the evidence that proves it. Drives the UI.",
    )
    evidence_sufficient: bool = Field(
        default=True,
        description="False when V8's corrective loop judged the evidence too thin to claim from.",
    )


class PriceQuote(BaseModel):
    """The Pricing Strategist's bid and the arithmetic behind it."""

    model_config = ConfigDict(extra="forbid")

    amount_usd: float = Field(gt=0)
    type: Literal["hourly", "fixed"] = "hourly"
    rationale: str
    profile_floor_usd: float | None = None
    market_p50_usd: float | None = None
    market_sample_size: int | None = None
    client_spend_per_hire_usd: float | None = None
    above_floor: bool = True
    connects_cost: int | None = None
    connects_worth_it: bool = True


class BidDecision(BaseModel):
    """The decision node's output: bid, watch or skip, with the reason."""

    model_config = ConfigDict(extra="forbid")

    decision: Decision
    reasons: list[str] = Field(
        default_factory=list, description="Why. Shown to the user; never empty for a skip."
    )
    deciding_factor: str = Field(
        default="", description="The single thing that settled it, for the trace."
    )
    fit_score: float | None = None
    risk_score: float | None = None
    confidence: float = Field(default=0.5, ge=0, le=1)


class ProposalDraft(BaseModel):
    """A draft, with every claim's evidence attached."""

    model_config = ConfigDict(extra="forbid")

    job_id: str
    cover_letter: str
    screening_answers: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    claim_citations: dict[str, list[str]] = Field(
        default_factory=dict, description="Claim text -> evidence ids. Powers evidence-on-hover."
    )
    revision: int = 0
    status: str = "pending_review"
    unsupported_claims_removed: list[str] = Field(default_factory=list)


class Critique(BaseModel):
    """The Critic's verdict. Bounded to two revisions by the graph (C4)."""

    model_config = ConfigDict(extra="forbid")

    verdict: CriticVerdict
    reasons: list[str] = Field(default_factory=list)
    grounding_ok: bool = True
    personalised: bool = True
    length_ok: bool = True
    opening_is_generic: bool = False
    suggested_fixes: list[str] = Field(default_factory=list)
    revision_requested: int = 0


class ComplianceVerdict(BaseModel):
    """The Compliance Guard's ruling against the policy pack.

    `fixed` matters as much as `blocked`: stripping a phone number is a repair the user
    should see, not a silent edit, and the UI shows both the redactions and the reason.
    """

    model_config = ConfigDict(extra="forbid")

    status: ComplianceStatus
    violations: list[str] = Field(default_factory=list)
    redactions: list[str] = Field(
        default_factory=list, description="What was removed: contact details, guarantees."
    )
    policy_rules_cited: list[str] = Field(default_factory=list)
    explanation: str = ""
    cleaned_text: str | None = None
