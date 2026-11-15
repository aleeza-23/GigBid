"""Request and response models for the API.

These exist so the TypeScript client can be generated from the OpenAPI schema rather than
hand-written, which Part D requires. Every response the UI renders is declared here, which
also means a change to a payload is a schema change the frontend's type check notices
rather than a runtime surprise.

Response models are deliberately flat and specific. The temptation with a graph behind an
API is to return the whole state blob and let the UI dig; that produces a UI coupled to
internal state shape and a payload measured in hundreds of kilobytes.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

Decision = Literal["bid", "watch", "skip"]
ApprovalAction = Literal["approve", "edit", "reject"]


class LoginRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    username: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=1, max_length=256)


class SessionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    user_id: str
    display_name: str
    persona: str
    token: str = Field(description="Also set as an HttpOnly cookie.")


class Health(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: str
    service: str = "gigbid-api"
    started_at: str | None = None
    marketplace: dict[str, Any] = Field(default_factory=dict)
    persistence: dict[str, Any] = Field(default_factory=dict)
    models: dict[str, Any] = Field(default_factory=dict)
    retrieval: dict[str, Any] = Field(default_factory=dict)
    degraded: list[str] = Field(
        default_factory=list,
        description="Subsystems that are down. The UI shows a banner per entry.",
    )


# ---------------------------------------------------------------------- jobs


class JobSummary(BaseModel):
    """A row in the job inbox, ranked by fit."""

    model_config = ConfigDict(extra="forbid")

    job_id: str
    title: str
    client_id: str | None = None
    category: str | None = None
    skills: list[str] = Field(default_factory=list)
    is_hourly: bool | None = None
    hourly_low: float | None = None
    hourly_high: float | None = None
    budget: float | None = None
    connects_cost: int | None = None
    published_date: str | None = None
    country: str | None = None

    # Present once a bid thread has run. Null means "not analysed yet", which the inbox
    # shows differently from "analysed and skipped".
    fit_coverage: float | None = None
    risk_score: float | None = None
    recommended_action: Decision | None = None
    recommended_price: float | None = None
    thread_status: str | None = None
    skill_overlap: int = Field(default=0, description="Cheap pre-analysis ranking signal.")


class JobPage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    total: int
    limit: int
    offset: int
    jobs: list[JobSummary]
    marketplace: str = "gigsim"


class RequirementEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    requirement: str
    covered: bool
    evidence: list[dict[str, Any]] = Field(default_factory=list)
    overruled_by_model: str | None = None


class JobDetail(BaseModel):
    """The job detail screen: spec, risk, the requirement-to-evidence matrix, price."""

    model_config = ConfigDict(extra="forbid")

    job: dict[str, Any]
    client: dict[str, Any] = Field(default_factory=dict)
    analysed: bool = False
    job_spec: dict[str, Any] | None = None
    risk: dict[str, Any] | None = None
    fit: dict[str, Any] | None = None
    requirement_matrix: list[RequirementEvidence] = Field(default_factory=list)
    price: dict[str, Any] | None = None
    decision: dict[str, Any] | None = None
    thread_id: str | None = None
    thread_status: str | None = None
    injection_removed: bool = False


# ----------------------------------------------------------------------- runs


class StartRunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    job_id: str = Field(min_length=1, max_length=64)
    persona: str | None = Field(
        default=None, description="Defaults to the signed-in user's persona."
    )


class RunSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    thread_id: str
    job_id: str | None = None
    status: str
    trace_id: str | None = None
    decision: str | None = None
    deciding_factor: str | None = None
    fit_coverage: float | None = None
    risk_score: float | None = None
    price_usd: float | None = None
    revision_count: int = 0
    steps_used: int = 0
    tokens_used: int = 0
    budget_exceeded: str | None = None
    unsupported_claims: int | None = None
    awaiting_approval: bool = False
    degraded: dict[str, Any] = Field(default_factory=dict)
    trace_entries: int = 0


class TraceEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ts: str | None = None
    node: str
    kind: str
    tool: str | None = None
    prompt: str | None = Field(default=None, description="Prompt version, e.g. writer@v3.")
    model: str | None = None
    tokens: int | None = None
    duration_ms: float | None = None
    result_summary: Any = None
    error: str | None = None
    arguments: dict[str, Any] | None = None


class RunDetail(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summary: RunSummary
    trace: list[TraceEntry] = Field(default_factory=list)
    errors: list[dict[str, Any]] = Field(default_factory=list)
    approval_request: dict[str, Any] | None = None


# ------------------------------------------------------------------ approvals


class PendingApproval(BaseModel):
    model_config = ConfigDict(extra="forbid")

    thread_id: str
    kind: Literal["proposal", "message"]
    job_id: str | None = None
    title: str | None = None
    payload: dict[str, Any] = Field(
        description="Everything the human needs: draft, evidence per claim, risk, price, "
        "the dry-run preview and its cost."
    )


class ApprovalDecisionRequest(BaseModel):
    """What the approval screen sends back.

    `cover_letter` or `text` is required for an edit; the server rejects an edit with no
    text rather than sending the original, because that would silently discard the
    human's change.
    """

    model_config = ConfigDict(extra="forbid")

    action: ApprovalAction
    cover_letter: str | None = Field(default=None, max_length=5000)
    text: str | None = Field(default=None, max_length=4000)
    reason: str | None = Field(default=None, max_length=500)


class ApprovalResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    thread_id: str
    status: str
    action: ApprovalAction
    proposal_id: str | None = None
    connects_spent: int | None = None
    idempotent_replay: bool = False
    message_sent: bool = False
    client_reply: str | None = None
    errors: list[dict[str, Any]] = Field(default_factory=list)


# ------------------------------------------------------------------- messages


class ThreadSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    thread_id: str
    job_id: str | None = None
    client_id: str | None = None
    message_count: int = 0
    last_from: str | None = None
    last_text: str | None = None
    awaiting_approval: bool = False
    run_status: str | None = None


class MessageThread(BaseModel):
    model_config = ConfigDict(extra="forbid")

    thread_id: str
    job_id: str | None = None
    messages: list[dict[str, Any]] = Field(default_factory=list)
    intent: str | None = None
    risk_flags: list[str] = Field(default_factory=list)
    draft_reply: str | None = None
    proposed_slots: list[dict[str, Any]] = Field(default_factory=list)
    awaiting_approval: bool = False
    run_status: str | None = None


# ------------------------------------------------------------------- pipeline


class PipelineCard(BaseModel):
    model_config = ConfigDict(extra="forbid")

    job_id: str
    title: str | None = None
    column: str = Field(description="drafted|submitted|viewed|replied|interview|hired|lost")
    thread_id: str | None = None
    proposal_id: str | None = None
    connects_spent: int | None = None
    price_usd: float | None = None
    fit_coverage: float | None = None
    updated_at: str | None = None


class PipelineBoard(BaseModel):
    model_config = ConfigDict(extra="forbid")

    columns: dict[str, list[PipelineCard]]
    totals: dict[str, int]


# ------------------------------------------------------------------ analytics


class Analytics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    proposals_submitted: int = 0
    replies: int = 0
    interviews: int = 0
    hires: int = 0
    reply_rate: float | None = None
    hire_rate: float | None = None
    connects_spent: int = 0
    connects_remaining: int | None = None
    by_category: list[dict[str, Any]] = Field(default_factory=list)
    cost: dict[str, Any] = Field(default_factory=dict)
    latency: dict[str, Any] = Field(default_factory=dict)
    top_skip_reasons: list[dict[str, Any]] = Field(default_factory=list)
    unsupported_claim_rate: float | None = None


# --------------------------------------------------------------------- memory


class MemoryView(BaseModel):
    """The settings screen's memory viewer: what was learned, why, and how to delete it."""

    model_config = ConfigDict(extra="forbid")

    user_id: str
    style_rules: list[dict[str, Any]] = Field(default_factory=list)
    outcome_rules: list[dict[str, Any]] = Field(default_factory=list)
    pending_style_observations: int = 0
    pending_outcome_observations: int = 0
    promotion_rules: list[dict[str, Any]] = Field(default_factory=list)


class ForgetRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["style", "outcome"]
    key: str = Field(min_length=1, max_length=120)
    forget_evidence: bool = Field(
        default=False,
        description="Also delete the observations, so promotion cannot re-derive the rule.",
    )


class Settings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    persona: str
    autonomy: Literal["manual", "auto_skip_low_fit"] = "manual"
    weekly_connects_budget: int | None = None
    connects_balance: int | None = None
    hourly_rate_usd: float | None = None
    minimum_project_budget_usd: float | None = None
    marketplace_mode: str = "gigsim"
    model_profile: str = "open"
    retrieval_variant: str = "builtin_bm25"
    embeddings_backend: str = "lsa"
    embeddings_spec_compliant: bool = False
    saved_searches: list[dict[str, Any]] = Field(default_factory=list)
