"""Input and output schemas for the marketplace tools.

A2 asks for Pydantic input schemas with *real* constraints and declared output schemas.
The constraints here mirror `gigsim/models.py` deliberately: a cover letter under 50
characters is refused at the tool boundary rather than after a round trip, which saves a
request against the rate budget and gives the model an error it can actually act on.

Output schemas are declared for the same reason the graph uses typed hand-offs (C3): a
node that receives `{"total": 12, "jobs": [...]}` can be written against a contract, and
a change to that contract is a schema change CI notices via tests/contract.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

# ----------------------------------------------------------- argument annotations
#
# Reused in server.py's registered signatures. FastMCP derives the published JSON Schema
# from the signature of the function it registers, so a constraint declared anywhere else
# is documentation rather than validation. Declaring them once keeps the two identical.

JobIdArg = Annotated[
    str, Field(min_length=1, max_length=64, description="Marketplace job id.")
]
ClientIdArg = Annotated[
    str, Field(min_length=1, max_length=64, description="Marketplace client id.")
]
FreelancerIdArg = Annotated[
    str, Field(min_length=1, max_length=64, description="The freelancer whose account to read.")
]
ThreadIdArg = Annotated[
    str, Field(min_length=1, max_length=64, description="Conversation thread id.")
]
LimitArg = Annotated[int, Field(ge=1, le=100, description="Page size, at most 100.")]
OffsetArg = Annotated[int, Field(ge=0, le=10000, description="Page offset.")]
HourlyArg = Annotated[
    float | None, Field(default=None, ge=0, le=1000, description="Hourly rate bound in USD.")
]
MessageTextArg = Annotated[
    str,
    Field(
        min_length=1,
        max_length=4000,
        description="The message body. No contact details, links off-platform or guarantees.",
    ),
]
IdempotencyKeyArg = Annotated[
    str,
    Field(
        min_length=8,
        max_length=128,
        description=(
            "Mint ONE key when the user approves and reuse it on every retry of that "
            "approval. A fresh key per attempt defeats the purpose and can spend twice."
        ),
    ),
]

# Mirrors gigsim.models. Duplicated rather than imported because the MCP server must also
# describe live Upwork, which has no GigSim import to borrow from, and because the tool
# boundary should not break if the simulator's internals move.
ProposalStatus = Literal[
    "draft", "submitted", "viewed", "replied", "interview", "hired", "declined"
]


class ScreeningAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1, max_length=1000)
    answer: str = Field(min_length=1, max_length=4000)


# ------------------------------------------------------------------------ reads


class JobSearchResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    total: int
    limit: int
    offset: int
    jobs: list[dict[str, Any]]
    marketplace: str = Field(description="Which adapter answered: gigsim or upwork_live.")


class ClientHistoryResult(BaseModel):
    """What the marketplace will say about a client.

    Note what is absent: hidden preferences and the scam flag. GigSim enforces that
    boundary in its store, and the Risk Assessor has to infer a scam from spend history,
    review text and the post itself. A tool that leaked the flag would make the scam
    precision and recall numbers in B4 meaningless.

    `hire_rate` and `spend_per_hire` are computed here rather than left to the model. A
    client with 20 posts and no hires is the single strongest scam signal in this corpus,
    and asking a 4B model to divide two numbers reliably is a worse design than doing the
    division in the tool.
    """

    model_config = ConfigDict(extra="forbid")

    client_id: str
    name: str | None = None
    country: str | None = None
    payment_verified: bool | None = None
    total_spent_usd: float | None = None
    hires: int | None = None
    jobs_posted: int | None = None
    avg_rating_given: float | None = None
    member_since: str | None = None
    reviews_from_freelancers: list[str] = Field(default_factory=list)

    hire_rate: float | None = Field(
        default=None, description="hires / jobs_posted. Near zero over many posts is a risk signal."
    )
    spend_per_hire: float | None = Field(
        default=None, description="total_spent_usd / hires, or null when nothing was hired."
    )
    open_job_count: int = 0
    past_job_count: int = 0
    open_jobs: list[dict[str, Any]] = Field(default_factory=list)
    past_jobs: list[dict[str, Any]] = Field(default_factory=list)
    marketplace: str = "gigsim"


class ConnectsResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    freelancer_id: str
    balance: int
    weekly_budget: int
    spent_this_week: int
    remaining_this_week: int = Field(description="weekly_budget minus spent_this_week, floored at 0.")
    marketplace: str = "gigsim"


class Percentiles(BaseModel):
    """A distribution, reported with its sample size.

    `n` is not decoration. A p50 drawn from 7,286 contracts and one drawn from 12 are
    different kinds of claim, and the Pricing Strategist should discount the second.
    """

    model_config = ConfigDict(extra="forbid")

    p10: float | None = None
    p25: float | None = None
    p50: float | None = None
    p75: float | None = None
    p90: float | None = None
    n: int = 0


class RateStatsResult(BaseModel):
    """Market rates for pricing, flattened into the three questions actually asked.

    GigSim nests this data three ways. Rather than hand a model the nested blob and hope,
    the tool answers: what do freelancers *earn* hourly here, what do clients *post* as a
    budget, and what do they post as an hourly range.
    """

    model_config = ConfigDict(extra="forbid")

    available: bool = True
    category: str | None = None
    domain: str | None = None
    earned_hourly_usd: Percentiles | None = Field(
        default=None, description="What freelancers are actually paid per hour in this domain."
    )
    posted_fixed_budget_usd: Percentiles | None = Field(
        default=None, description="Fixed budgets clients post in this domain or category."
    )
    posted_hourly_low_usd: Percentiles | None = None
    posted_hourly_high_usd: Percentiles | None = None
    top_skills: list[str] = Field(default_factory=list)
    reason: str | None = Field(
        default=None, description="Why statistics are unavailable, when they are."
    )
    marketplace: str = "gigsim"


class MessagesResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    thread_id: str
    messages: list[dict[str, Any]]
    marketplace: str = "gigsim"


class ThreadsResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    total: int
    threads: list[dict[str, Any]]
    marketplace: str = "gigsim"


# ----------------------------------------------------------------------- writes


class ProposalInput(BaseModel):
    """A proposal, validated before it costs anything.

    `cover_letter` has a 50-character floor because GigSim rejects anything shorter, and
    discovering that after spending a request is worse than discovering it here.
    """

    model_config = ConfigDict(extra="forbid")

    freelancer_id: str = Field(min_length=1, max_length=64)
    job_id: str = Field(min_length=1, max_length=64)
    cover_letter: str = Field(
        min_length=50,
        max_length=5000,
        description="The proposal body. Every claim must cite portfolio evidence.",
    )
    screening_answers: list[ScreeningAnswer] = Field(default_factory=list, max_length=10)
    bid_rate: float = Field(gt=0, le=1000, description="Hourly rate or fixed bid in USD.")
    estimated_duration_days: int | None = Field(default=None, ge=1, le=365)


class DryRunPreview(BaseModel):
    """Exactly what would be sent, and what it would cost.

    This is the payload the approval screen renders (C5). Every field here is something a
    human needs in order to say yes: the text, the price, the Connects hit, and whether
    it breaks the weekly budget. `None` on a cost field means the marketplace does not
    quote one, not that it is free.
    """

    model_config = ConfigDict(extra="forbid")

    dry_run: Literal[True] = True
    job_id: str
    would_send: dict[str, Any]
    connects_cost: int | None
    connects_balance_after: int | None
    within_weekly_budget: bool | None
    warnings: list[str] = Field(default_factory=list)
    marketplace: str = "gigsim"
    mode: str | None = None


class SubmitResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    proposal_id: str | None = None
    job_id: str | None = None
    client_id: str | None = None
    freelancer_id: str | None = None
    status: str
    connects_spent: int | None = None
    created_at: str | None = None
    idempotency_key: str
    idempotent_replay: bool = Field(
        default=False,
        description="True when this key had already been used; nothing was spent again.",
    )
    requires_user_confirmation_on_upwork: bool = False
    marketplace: str = "gigsim"
    trace_id: str | None = None


class SendMessageResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    thread_id: str
    sent: str | None = None
    client_reply: str | None = None
    messages: list[dict[str, Any]] = Field(default_factory=list)
    status: str = "sent"
    idempotency_key: str
    idempotent_replay: bool = False
    requires_user_confirmation_on_upwork: bool = False
    marketplace: str = "gigsim"
    trace_id: str | None = None


class MessageDryRun(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dry_run: Literal[True] = True
    thread_id: str
    would_send: dict[str, Any]
    connects_cost: int = 0
    warnings: list[str] = Field(default_factory=list)
    marketplace: str = "gigsim"
