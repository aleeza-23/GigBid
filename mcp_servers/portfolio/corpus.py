"""Loads the persona portfolio and resolves every evidence item back to its text.

`data/personas/evidence_index.json` is the catalogue: 592 items across three personas,
each with the metadata B1 requires (persona, type, project, skills, date, client industry
and, for proposals, outcome). The index holds no text, only a `path`, because the same
project is deliberately described three ways on disk and the retriever should decide which
description to show.

This module does the boring half of that: turn an index entry into an EvidenceItem with
its text attached, cheaply and repeatedly. It knows nothing about ranking. Scoring lives
in `search.py` so that Part B can replace it with a measured retriever without touching
the tool layer.

Loading is lazy and cached per persona. A tool call that asks about one persona never pays
to parse the other two.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, fields
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

# Overridable so the server can be pointed at a real freelancer's own portfolio, which is
# the bar A1 sets: another freelancer should be able to install this and use it. Defaults
# to the personas shipped with the repo.
PERSONAS_DIR = Path(
    os.getenv("PERSONAS_DIR") or Path(__file__).resolve().parents[2] / "data" / "personas"
)
INDEX_PATH = PERSONAS_DIR / "evidence_index.json"

EvidenceType = Literal["case_study", "proposal", "review", "readme", "code", "cv"]
VALID_PERSONAS = ("ai_ml_engineer", "fullstack_web_developer", "data_analyst")

# Proposal outcomes, worst to best. V4 (outcome-aware ranking) boosts the top of this
# list; the Writer is shown what actually worked rather than what was merely sent.
OUTCOME_RANK = {"ignored": 0, "replied": 1, "won": 2}


class PortfolioError(RuntimeError):
    """An item or persona that does not exist."""


@dataclass
class EvidenceItem:
    """One piece of portfolio evidence, with its text resolved.

    `citation` is the string a proposal claim points at. R5 says no sentence may claim
    experience the portfolio does not support, which only works if every claim carries a
    stable handle back to the thing that supports it.
    """

    id: str
    persona: str
    type: str
    text: str
    source: str
    date: str | None = None
    project: str | None = None
    project_id: str | None = None
    skills: list[str] = field(default_factory=list)
    client_industry: str | None = None
    outcome: str | None = None
    cited_project_ids: list[str] = field(default_factory=list)
    score: float | None = None
    rating: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)
    """Variant annotations: which retriever found it, which requirements it covers,
    whether the corrective loop judged the evidence sufficient. A declared field rather
    than attributes set on the instance, because retrieval pipelines copy these objects
    between stages and a dynamic attribute breaks the copy."""

    @property
    def citation(self) -> str:
        return f"portfolio://{self.type}/{self.id}"

    def clone(self, **changes: Any) -> EvidenceItem:
        """A copy with fields replaced. The safe way to re-score an item mid-pipeline."""
        data = {f.name: getattr(self, f.name) for f in fields(self)}
        data["extra"] = dict(self.extra)
        data.update(changes)
        return EvidenceItem(**data)

    def annotate(self, **notes: Any) -> EvidenceItem:
        """Attach variant metadata without touching the declared fields."""
        self.extra.update(notes)
        return self

    def to_dict(self, include_text: bool = True, text_limit: int | None = None) -> dict[str, Any]:
        out = asdict(self)
        out["citation"] = self.citation
        if not include_text:
            out.pop("text")
        elif text_limit and len(self.text) > text_limit:
            out["text"] = self.text[:text_limit] + f"...[+{len(self.text) - text_limit} chars]"
        if out.get("score") is None:
            out.pop("score", None)
        if out.get("rating") is None:
            out.pop("rating", None)
        if not out.get("extra"):
            out.pop("extra", None)
        return out


def _read_index() -> list[dict[str, Any]]:
    if not INDEX_PATH.exists():
        raise PortfolioError(
            f"evidence index missing at {INDEX_PATH}. Run `make seed` to build the corpus."
        )
    return json.loads(INDEX_PATH.read_text(encoding="utf-8"))


def _strip_frontmatter(raw: str) -> str:
    """Drop the YAML block from a case study or README.

    The metadata is already in the index; leaving it in the text would let a lexical
    retriever match on a frontmatter skill list rather than on what the project says,
    which flatters the baseline for the wrong reason.
    """
    if raw.startswith("---"):
        end = raw.find("\n---", 3)
        if end != -1:
            return raw[end + 4 :].lstrip("\n")
    return raw


@lru_cache(maxsize=8)
def _persona_blobs(persona: str) -> dict[str, dict[str, Any]]:
    """The JSON collections for one persona, keyed by item id.

    proposals.json and reviews.json each hold many items behind a single index `path`, so
    they are loaded once and indexed by id rather than re-read per item.
    """
    base = PERSONAS_DIR / persona
    blobs: dict[str, dict[str, Any]] = {}
    for name in ("proposals.json", "reviews.json"):
        path = base / name
        if not path.exists():
            continue
        rows = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(rows, dict):
            rows = next((v for v in rows.values() if isinstance(v, list)), [])
        for row in rows:
            if isinstance(row, dict) and row.get("id"):
                blobs[row["id"]] = row
    return blobs


def _resolve_text(entry: dict[str, Any]) -> tuple[str, str]:
    """Return (text, human-readable source) for one index entry."""
    persona = entry["persona"]
    rel = entry.get("path") or ""
    base = PERSONAS_DIR / persona

    # proposals.json and reviews.json are collections: find this id inside them.
    if rel.endswith(".json"):
        row = _persona_blobs(persona).get(entry["id"])
        if row is None:
            return "", f"{persona}/{rel}"
        return str(row.get("text") or ""), f"{persona}/{rel}#{entry['id']}"

    path = base / rel
    if not path.exists():
        return "", f"{persona}/{rel}"
    raw = path.read_text(encoding="utf-8", errors="replace")
    if path.suffix in (".md", ".markdown"):
        raw = _strip_frontmatter(raw)
    return raw, f"{persona}/{rel}"


@lru_cache(maxsize=8)
def load_corpus(persona: str) -> tuple[EvidenceItem, ...]:
    """Every evidence item for one persona, text resolved. Cached."""
    if persona not in VALID_PERSONAS:
        raise PortfolioError(f"unknown persona {persona!r}; expected one of {VALID_PERSONAS}")

    items: list[EvidenceItem] = []
    for entry in _read_index():
        if entry.get("persona") != persona:
            continue
        text, source = _resolve_text(entry)
        blob = _persona_blobs(persona).get(entry["id"], {})
        items.append(
            EvidenceItem(
                id=entry["id"],
                persona=persona,
                type=entry.get("type", "unknown"),
                text=text,
                source=source,
                date=entry.get("date"),
                project=entry.get("project"),
                project_id=entry.get("project_id"),
                skills=list(entry.get("skills") or []),
                client_industry=entry.get("client_industry"),
                outcome=entry.get("outcome"),
                cited_project_ids=list(entry.get("cited_project_ids") or []),
                rating=blob.get("rating"),
            )
        )
    if not items:
        raise PortfolioError(f"no evidence indexed for persona {persona!r}")
    return tuple(items)


def get_item(persona: str, item_id: str) -> EvidenceItem:
    """One item by id, or PortfolioError."""
    for item in load_corpus(persona):
        if item.id == item_id:
            return item
    raise PortfolioError(f"no evidence item {item_id!r} for persona {persona!r}")


def find_item_any_persona(item_id: str) -> EvidenceItem:
    """One item by id without knowing its persona, for resource URIs."""
    for persona in VALID_PERSONAS:
        try:
            return get_item(persona, item_id)
        except PortfolioError:
            continue
    raise PortfolioError(f"no evidence item {item_id!r} in any persona")


def load_profile(persona: str) -> dict[str, Any]:
    """The persona's profile: rates, floor, skills, availability."""
    path = PERSONAS_DIR / persona / "profile.json"
    if not path.exists():
        raise PortfolioError(f"no profile for persona {persona!r}")
    return json.loads(path.read_text(encoding="utf-8"))


def list_skills(persona: str) -> list[dict[str, Any]]:
    """Declared skills, each with the evidence that actually backs it.

    A skill on a CV and a skill with three delivered projects behind it are not the same
    claim, so the tool reports both the declared list and the evidence count. The Fit
    Matcher uses the count; a skill with zero backing evidence is one the Compliance Guard
    should stop the Writer from boasting about.
    """
    corpus = load_corpus(persona)
    profile = load_profile(persona)

    counts: dict[str, dict[str, Any]] = {}
    for skill in profile.get("skills") or []:
        counts[skill.lower()] = {
            "skill": skill,
            "declared": True,
            "evidence_count": 0,
            "evidence_ids": [],
            "projects": [],
            "won_proposals": 0,
        }

    # Which skills each project demonstrates, so a proposal's outcome can be credited to
    # the skills it actually argued for. A proposal carries no skills of its own in the
    # index; it cites project ids, and the skills live on those projects.
    skills_by_project: dict[str, set[str]] = {}
    for item in corpus:
        if item.project_id:
            skills_by_project.setdefault(item.project_id, set()).update(item.skills)

    def row_for(skill: str) -> dict[str, Any]:
        return counts.setdefault(
            skill.lower(),
            {
                "skill": skill,
                "declared": False,
                "evidence_count": 0,
                "evidence_ids": [],
                "projects": [],
                "won_proposals": 0,
            },
        )

    for item in corpus:
        for skill in item.skills:
            row = row_for(skill)
            row["evidence_count"] += 1
            if len(row["evidence_ids"]) < 12:
                row["evidence_ids"].append(item.id)
            if item.project and item.project not in row["projects"]:
                row["projects"].append(item.project)

        # Credit a won proposal to every skill of every project it cited. This is what
        # makes "I have won work with Celery" a checkable statement rather than a guess,
        # and it is the read-only ancestor of the outcome memory in C2.
        if item.type == "proposal" and item.outcome == "won":
            for project_id in item.cited_project_ids:
                for skill in skills_by_project.get(project_id, ()):
                    row_for(skill)["won_proposals"] += 1

    return sorted(counts.values(), key=lambda r: (-r["evidence_count"], r["skill"].lower()))
