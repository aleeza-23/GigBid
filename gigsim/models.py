"""Request and response schemas for the GigSim API.

The constraints here are deliberately strict. The marketplace MCP server mirrors them in
its own tool schemas, so a malformed proposal is rejected at the tool boundary rather
than halfway through a graph run.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

OutcomeLiteral = Literal["ignored", "viewed", "replied", "interview", "hired"]
ProposalStatus = Literal["draft", "submitted", "viewed", "replied", "interview", "hired", "declined"]


class ScreeningAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1, max_length=1000)
    answer: str = Field(min_length=1, max_length=4000)


class ProposalCreate(BaseModel):
    """A proposal as submitted to the marketplace.

    dry_run returns the cost and the exact payload that would be sent without spending
    Connects or creating anything, which is what the approval screen previews.
    """

    model_config = ConfigDict(extra="forbid")

    freelancer_id: str = Field(min_length=1, max_length=64)
    job_id: str = Field(min_length=1, max_length=64)
    cover_letter: str = Field(min_length=50, max_length=5000)
    screening_answers: list[ScreeningAnswer] = Field(default_factory=list, max_length=10)
    bid_rate: float = Field(gt=0, le=1000, description="Hourly rate or fixed bid in USD")
    estimated_duration_days: int | None = Field(default=None, ge=1, le=365)
    dry_run: bool = False


class Proposal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    proposal_id: str
    job_id: str
    client_id: str
    freelancer_id: str
    cover_letter: str
    screening_answers: list[ScreeningAnswer] = Field(default_factory=list)
    bid_rate: float
    estimated_duration_days: int | None = None
    status: ProposalStatus
    connects_spent: int
    created_at: str
    decided_at: str | None = None
    decision_score: float | None = None
    idempotency_key: str | None = None


class DryRunPreview(BaseModel):
    """Exactly what would be sent, and what it would cost."""

    model_config = ConfigDict(extra="forbid")

    dry_run: Literal[True] = True
    job_id: str
    would_send: dict[str, Any]
    connects_cost: int
    connects_balance_after: int
    within_weekly_budget: bool
    warnings: list[str] = Field(default_factory=list)


class MessageCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    thread_id: str = Field(min_length=1, max_length=64)
    text: str = Field(min_length=1, max_length=4000)
    dry_run: bool = False


class Message(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    sender: Literal["freelancer", "client"] = Field(alias="from")
    text: str
    ts: str


class Thread(BaseModel):
    model_config = ConfigDict(extra="forbid")

    thread_id: str
    job_id: str
    client_id: str
    freelancer_id: str
    proposal_id: str | None = None
    messages: list[dict[str, Any]] = Field(default_factory=list)
    data_origin: str = "synthetic"


class ConnectsBalance(BaseModel):
    model_config = ConfigDict(extra="forbid")

    freelancer_id: str
    balance: int
    weekly_budget: int
    spent_this_week: int


class JobPage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    total: int
    limit: int
    offset: int
    jobs: list[dict[str, Any]]


class TickResult(BaseModel):
    """What the clients did when the simulation last advanced."""

    model_config = ConfigDict(extra="forbid")

    decided: int
    outcomes: dict[str, int]
    details: list[dict[str, Any]] = Field(default_factory=list)


class ErrorBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    error: str
    detail: str | None = None
