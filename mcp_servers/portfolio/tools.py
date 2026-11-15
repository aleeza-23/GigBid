"""Portfolio tool implementations and their schemas.

Kept separate from server.py so the same callables can be unit-tested directly and
registered through FastMCP without a running server. Every tool here is read-only: the
Portfolio server never writes, which is why its annotations say so and why it needs no
idempotency keys.

The schemas are not decoration. A2's lesson applies to the read side too: constraints
declared in Pydantic are enforced at the tool boundary, so a node asking for k=500 is
refused by the schema rather than quietly handed the whole corpus.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from mcp_servers.portfolio import corpus
from mcp_servers.portfolio.search import get_ranker

EvidenceTypeName = Literal["case_study", "proposal", "review", "readme", "code", "cv"]
PersonaName = Literal["ai_ml_engineer", "fullstack_web_developer", "data_analyst"]

# Text budget per item. The Writer needs enough of a case study to quote it; it does not
# need 8 KB of source code, and the token bill for 10 items at full length is real.
DEFAULT_TEXT_LIMIT = 1200

# ----------------------------------------------------------- argument annotations
#
# Declared once here and reused in server.py's registered signatures. This matters more
# than it looks: FastMCP builds the public JSON Schema from the signature of the function
# it registers, so a constraint that lives only on the inner implementation is not
# enforced for any caller. Sharing the aliases keeps the published schema and the
# validated implementation the same thing by construction.

RequirementArg = Annotated[
    str,
    Field(
        min_length=3,
        max_length=2000,
        description=(
            "One requirement in the client's words, for example 'build a dbt model for a "
            "Snowflake warehouse'. Pass requirements one at a time rather than the whole "
            "job post: the corpus describes each project several ways and a single "
            "blended query returns the average of all of them."
        ),
    ),
]
PersonaArg = Annotated[PersonaName, Field(description="Whose portfolio to search.")]
KArg = Annotated[int, Field(ge=1, le=20, description="How many items to return.")]
OutcomeAwareArg = Annotated[
    bool,
    Field(
        description=(
            "Boost evidence from proposals that won or got replies. Useful when choosing "
            "phrasing; switch it off when you want neutral coverage."
        )
    ),
]
TextLimitArg = Annotated[
    int, Field(ge=200, le=8000, description="Max characters of text per item.")
]
CaseIdArg = Annotated[
    str,
    Field(
        min_length=2,
        max_length=64,
        pattern=r"^[A-Za-z0-9_-]+$",
        description="Evidence id, for example 'ai-p01-case', or a project id like 'ai-p01'.",
    ),
]
MinEvidenceArg = Annotated[
    int, Field(ge=0, le=50, description="Only skills with at least this much evidence.")
]


class EvidenceFilters(BaseModel):
    """Narrowing options for search_evidence.

    Separate model rather than loose kwargs so the filter surface is visible in the tool
    schema a host shows the user, and so adding a filter is a schema change CI notices.
    """

    model_config = ConfigDict(extra="forbid")

    types: list[EvidenceTypeName] | None = Field(
        default=None, description="Restrict to these evidence types."
    )
    skills: list[str] | None = Field(
        default=None,
        max_length=20,
        description="Only items whose indexed skills include at least one of these.",
    )
    min_date: str | None = Field(
        default=None,
        pattern=r"^\d{4}-\d{2}-\d{2}$",
        description="ISO date; drop evidence older than this. Use to prefer recent work.",
    )
    client_industry: str | None = Field(
        default=None, max_length=80, description="Case-insensitive industry match."
    )
    outcomes: list[Literal["won", "replied", "ignored", "delivered"]] | None = Field(
        default=None, description="For proposals, keep only these outcomes."
    )
    project_ids: list[str] | None = Field(
        default=None, max_length=50, description="Restrict to specific projects."
    )
    exclude_ids: list[str] | None = Field(
        default=None, max_length=50, description="Evidence ids to leave out."
    )


class EvidenceHit(BaseModel):
    """One retrieved item. The output schema the graph relies on."""

    model_config = ConfigDict(extra="forbid")

    id: str
    citation: str = Field(description="Stable handle a proposal claim points at.")
    type: str
    text: str
    source: str = Field(description="Repo-relative path the text came from.")
    persona: str
    score: float | None = None
    date: str | None = None
    project: str | None = None
    project_id: str | None = None
    skills: list[str] = Field(default_factory=list)
    client_industry: str | None = None
    outcome: str | None = None
    rating: float | None = None
    cited_project_ids: list[str] = Field(
        default_factory=list,
        description="For a past proposal, the projects it cited as evidence.",
    )
    extra: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "What the retrieval variant recorded about this hit: which retrievers found "
            "it, which requirements it covers, whether the corrective loop judged the "
            "evidence sufficient."
        ),
    )
    """Annotations from the configured variant.

    This field has to exist because the model forbids extra keys and `EvidenceItem.to_dict`
    emits `extra` whenever a variant annotated the hit. With the built-in BM25 ranker
    nothing annotates anything, so the omission was invisible until a measured variant was
    configured, at which point every `search_evidence` call failed validation. The
    annotations are also what the requirement-to-evidence matrix in the UI reads, so
    dropping them rather than declaring them would have cost a screen."""


class SearchEvidenceResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    requirement: str
    persona: str
    k: int
    total_candidates: int = Field(description="Items that survived filtering before ranking.")
    hits: list[EvidenceHit]
    ranker: str = Field(description="Which ranking implementation answered this call.")


def search_evidence(
    requirement: RequirementArg,
    persona: PersonaArg,
    k: KArg = 5,
    filters: Annotated[EvidenceFilters | None, Field(description="Optional narrowing.")] = None,
    outcome_aware: OutcomeAwareArg = False,
    text_limit: TextLimitArg = DEFAULT_TEXT_LIMIT,
) -> dict[str, Any]:
    """Find portfolio evidence that supports one requirement.

    Returns items with type, source, date and outcome so the caller can cite them. Every
    claim in a proposal must trace to an id this tool returned.
    """
    items = corpus.load_corpus(persona)
    flt = filters or EvidenceFilters()

    def keep(item: corpus.EvidenceItem) -> bool:
        if flt.types and item.type not in flt.types:
            return False
        if flt.exclude_ids and item.id in flt.exclude_ids:
            return False
        if flt.project_ids and item.project_id not in flt.project_ids:
            return False
        if flt.min_date and (item.date or "") < flt.min_date:
            return False
        if flt.client_industry and (item.client_industry or "").lower() != (
            flt.client_industry.lower()
        ):
            return False
        if flt.skills:
            wanted = {s.lower() for s in flt.skills}
            if not wanted & {s.lower() for s in item.skills}:
                return False
        # Items with no outcome (a README, say) are not proposals and are not what an
        # outcome filter is asking about, so they drop out.
        if flt.outcomes and item.outcome not in set(flt.outcomes):
            return False
        return True

    candidates = [i for i in items if keep(i)]
    ranker = get_ranker()
    hits = ranker(requirement, candidates, k, outcome_aware=outcome_aware)

    return SearchEvidenceResult(
        requirement=requirement,
        persona=persona,
        k=k,
        total_candidates=len(candidates),
        hits=[
            EvidenceHit(**item.to_dict(text_limit=text_limit))  # type: ignore[arg-type]
            for item in hits
        ],
        ranker=getattr(ranker, "__qualname__", repr(ranker)),
    ).model_dump()


class CaseStudy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    citation: str
    project: str | None = None
    project_id: str | None = None
    persona: str
    date: str | None = None
    client_industry: str | None = None
    outcome: str | None = None
    skills: list[str] = Field(default_factory=list)
    text: str
    source: str
    related: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Other evidence for the same project: repo, code, review.",
    )


def get_case_study(
    case_id: CaseIdArg,
    persona: Annotated[
        PersonaName | None,
        Field(description="Narrows the lookup. Omit to search all personas."),
    ] = None,
) -> dict[str, Any]:
    """Load one case study in full, with the other evidence for the same project.

    `search_evidence` truncates text to keep token costs down. Call this when a case study
    looks like the right evidence and the Writer needs the whole thing, including the
    numbers in the results section.
    """
    item: corpus.EvidenceItem | None = None
    if persona:
        candidates = corpus.load_corpus(persona)
        item = next((i for i in candidates if i.id == case_id), None)
        if item is None:
            # Accept a bare project id too; agents conflate the two constantly and
            # failing on that is a tool-design problem, not a model problem.
            item = next(
                (i for i in candidates if i.project_id == case_id and i.type == "case_study"),
                None,
            )
        if item is None:
            raise corpus.PortfolioError(f"no case study {case_id!r} for persona {persona!r}")
    else:
        try:
            item = corpus.find_item_any_persona(case_id)
        except corpus.PortfolioError:
            item = None
            for name in corpus.VALID_PERSONAS:
                found = next(
                    (
                        i
                        for i in corpus.load_corpus(name)
                        if i.project_id == case_id and i.type == "case_study"
                    ),
                    None,
                )
                if found:
                    item = found
                    break
            if item is None:
                raise

    related = [
        i.to_dict(text_limit=400)
        for i in corpus.load_corpus(item.persona)
        if item.project_id and i.project_id == item.project_id and i.id != item.id
    ]

    return CaseStudy(
        id=item.id,
        citation=item.citation,
        project=item.project,
        project_id=item.project_id,
        persona=item.persona,
        date=item.date,
        client_industry=item.client_industry,
        outcome=item.outcome,
        skills=item.skills,
        text=item.text,
        source=item.source,
        related=related,
    ).model_dump()


class SkillRow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    skill: str
    declared: bool = Field(description="Named on the persona's profile.")
    evidence_count: int
    evidence_ids: list[str] = Field(default_factory=list)
    projects: list[str] = Field(default_factory=list)
    won_proposals: int = 0


class SkillsResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    persona: str
    headline: str | None = None
    hourly_rate_usd: float | None = None
    minimum_project_budget_usd: float | None = None
    years_experience: int | None = None
    skills: list[SkillRow]
    unbacked_skills: list[str] = Field(
        description="Declared on the profile but with no evidence behind them. Do not claim these."
    )


def list_skills(
    persona: PersonaArg,
    min_evidence: MinEvidenceArg = 0,
) -> dict[str, Any]:
    """List the persona's skills with the evidence backing each one.

    A skill on a CV and a skill with three delivered projects behind it are different
    claims. `evidence_count` tells them apart, and `unbacked_skills` names the ones the
    Writer must not boast about, because R5 forbids claiming what the portfolio cannot
    support.
    """
    rows = corpus.list_skills(persona)
    profile = corpus.load_profile(persona)
    kept = [r for r in rows if r["evidence_count"] >= min_evidence]

    return SkillsResult(
        persona=persona,
        headline=profile.get("headline"),
        hourly_rate_usd=profile.get("hourly_rate_usd"),
        minimum_project_budget_usd=profile.get("minimum_project_budget_usd"),
        years_experience=profile.get("years_experience"),
        skills=[SkillRow(**r) for r in kept],
        unbacked_skills=[r["skill"] for r in rows if r["declared"] and r["evidence_count"] == 0],
    ).model_dump()
