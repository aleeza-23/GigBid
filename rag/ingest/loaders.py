"""Loaders for every portfolio source B1 names.

PDF CVs, Markdown case studies, repository READMEs and code, client reviews and past
proposals. Each loader returns `RawDocument`s carrying the metadata B1 requires on every
chunk: persona, type, project, skills, date, client industry and, for proposals, outcome.

Two choices worth defending:

**Metadata comes from the document, not from the filename.** Case studies and READMEs
carry YAML frontmatter, so the loader parses it rather than inferring type from a path.
A file moved to a different folder keeps its meaning.

**Code is loaded as code, not as prose.** A Python file is split on definition boundaries
rather than on character counts, so a chunk is a function with its signature intact. A
chunk that starts halfway through a function body cannot support a claim about what the
code does.

GitHub ingestion goes through the official GitHub MCP server rather than the API directly,
so it obeys A5's rule that third-party access is an MCP tool. `github.py` holds that.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Extensions we treat as source code, mapped to the language a chunker should assume.
CODE_SUFFIXES = {
    ".py": "python",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".js": "javascript",
    ".jsx": "javascript",
    ".sql": "sql",
    ".r": "r",
    ".R": "r",
    ".go": "go",
    ".java": "java",
    ".rb": "ruby",
}

_FRONTMATTER = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


@dataclass
class RawDocument:
    """One source document before chunking."""

    id: str
    text: str
    type: str
    source: str
    persona: str | None = None
    project: str | None = None
    project_id: str | None = None
    skills: list[str] = field(default_factory=list)
    date: str | None = None
    client_industry: str | None = None
    outcome: str | None = None
    language: str | None = None
    cited_project_ids: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    def metadata(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "source": self.source,
            "persona": self.persona,
            "project": self.project,
            "project_id": self.project_id,
            "skills": list(self.skills),
            "date": self.date,
            "client_industry": self.client_industry,
            "outcome": self.outcome,
            "language": self.language,
            "cited_project_ids": list(self.cited_project_ids),
        }


def parse_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    """Pull a YAML frontmatter block off the front of a document.

    Falls back to a tiny line parser when PyYAML is unavailable, because losing all
    metadata over a missing optional dependency would silently degrade retrieval.
    """
    match = _FRONTMATTER.match(text)
    if not match:
        return {}, text
    block, body = match.group(1), text[match.end() :]
    try:
        import yaml

        parsed = yaml.safe_load(block) or {}
        if isinstance(parsed, dict):
            return parsed, body
    except Exception:
        pass

    meta: dict[str, Any] = {}
    for line in block.split("\n"):
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        value = value.strip().strip("\"'")
        if value.startswith("[") and value.endswith("]"):
            meta[key.strip()] = [
                v.strip().strip("\"'") for v in value[1:-1].split(",") if v.strip()
            ]
        else:
            meta[key.strip()] = value
    return meta, body


def load_markdown(path: Path, persona: str | None = None) -> RawDocument:
    """A case study, CV or README. Metadata from frontmatter where present."""
    text = path.read_text(encoding="utf-8", errors="replace")
    meta, body = parse_frontmatter(text)

    inferred = "case_study"
    if path.name.lower().startswith("readme"):
        inferred = "readme"
    elif path.stem.lower() == "cv":
        inferred = "cv"

    return RawDocument(
        id=str(meta.get("id") or f"{path.parent.name}-{path.stem}"),
        text=body.strip(),
        type=str(meta.get("type") or inferred),
        source=str(path),
        persona=str(meta.get("persona") or persona or ""),
        project=meta.get("project") or (path.parent.name if inferred == "readme" else None),
        project_id=meta.get("project_id"),
        skills=list(meta.get("skills") or []),
        date=meta.get("date"),
        client_industry=meta.get("client_industry"),
        outcome=meta.get("outcome"),
    )


def load_code(path: Path, persona: str | None = None, project_id: str | None = None) -> RawDocument:
    """A source file. Language recorded so the chunker can split on definitions."""
    # The catalogue keys code and READMEs by project id, e.g. "ai-p01-code". Derive the
    # same shape where a project id is known so enrich() can find the entry.
    item_id = f"{project_id}-code" if project_id else f"{path.parent.name}-{path.stem}-code"
    return RawDocument(
        id=item_id,
        text=path.read_text(encoding="utf-8", errors="replace"),
        type="code",
        source=str(path),
        persona=persona,
        project=path.parent.name,
        project_id=project_id,
        language=CODE_SUFFIXES.get(path.suffix, "text"),
    )


def load_pdf(path: Path, persona: str | None = None) -> RawDocument:
    """A PDF CV. Requires pypdf; raises with a clear message rather than returning empty.

    Silently returning an empty document would make a CV look like it contained nothing,
    and the Fit Matcher would conclude the freelancer has no background.
    """
    try:
        import pypdf
    except ImportError as exc:
        raise RuntimeError(
            "PDF loading needs pypdf: pip install pypdf"
        ) from exc

    reader = pypdf.PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    text = "\n\n".join(pages).strip()
    if not text:
        raise RuntimeError(
            f"{path.name} yielded no text. It is probably a scanned image, which needs OCR "
            "before it can be indexed."
        )
    return RawDocument(
        id=f"{path.stem}-cv",
        text=text,
        type="cv",
        source=str(path),
        persona=persona,
        extra={"pages": len(pages)},
    )


def load_json_collection(
    path: Path, doc_type: str, persona: str | None = None
) -> list[RawDocument]:
    """proposals.json or reviews.json: many documents behind one file."""
    rows = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(rows, dict):
        rows = next((v for v in rows.values() if isinstance(v, list)), [])

    out: list[RawDocument] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        text = str(row.get("text") or "").strip()
        if not text:
            continue
        out.append(
            RawDocument(
                id=str(row.get("id") or f"{path.stem}-{len(out)}"),
                text=text,
                type=str(row.get("type") or doc_type),
                source=f"{path}#{row.get('id')}",
                persona=str(row.get("persona") or persona or ""),
                project=row.get("project"),
                project_id=row.get("project_id"),
                skills=list(row.get("skills") or []),
                date=row.get("date"),
                client_industry=row.get("client_industry"),
                # For a proposal this is the field that matters most: V4 ranks on it and
                # the outcome memory in C2 learns from it.
                outcome=row.get("outcome"),
                cited_project_ids=list(row.get("cited_project_ids") or []),
                extra={"rating": row.get("rating"), "job_category": row.get("job_category")},
            )
        )
    return out


def load_evidence_index(root: Path) -> dict[str, dict[str, Any]]:
    """The corpus catalogue, if present, keyed by item id.

    READMEs and source files carry no frontmatter, so their skills, project id and date
    live only in `evidence_index.json`. Without merging it, 205 of 221 chunks would ship
    with no skill metadata at all, and B1 requires it on every chunk. It also means a
    lexical retriever can match "Celery" against a README whose prose never says it.
    """
    path = root / "evidence_index.json"
    if not path.exists():
        return {}
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return {str(r["id"]): r for r in rows if isinstance(r, dict) and r.get("id")}


def enrich(
    doc: RawDocument,
    catalogue: dict[str, dict[str, Any]],
    by_path: dict[str, dict[str, Any]] | None = None,
    skills_by_project: dict[str, list[str]] | None = None,
) -> RawDocument:
    """Fill gaps in a document's metadata from the catalogue, without overwriting it.

    Matched by relative path first, then by id. Path is the reliable join: a README has no
    frontmatter to carry an id, so its generated id never matches the catalogue, but
    `repos/ticket-router/README.md` does.

    The document's own frontmatter wins where it exists: it is closer to the content, and
    an author editing a case study should not be silently overridden by a stale index.
    """
    entry = None
    if by_path:
        # Normalise separators so a Windows path matches a catalogue written with "/".
        key = str(doc.source).replace("\\", "/")
        entry = next((v for k, v in by_path.items() if key.endswith(k)), None)
    entry = entry or catalogue.get(doc.id)

    if entry:
        doc.id = str(entry.get("id") or doc.id)
        doc.skills = doc.skills or list(entry.get("skills") or [])
        doc.project = doc.project or entry.get("project")
        doc.project_id = doc.project_id or entry.get("project_id")
        doc.date = doc.date or entry.get("date")
        doc.client_industry = doc.client_industry or entry.get("client_industry")
        doc.outcome = doc.outcome or entry.get("outcome")
        doc.cited_project_ids = doc.cited_project_ids or list(
            entry.get("cited_project_ids") or []
        )

    # Proposals and reviews carry no skills of their own. A proposal argues from the
    # projects it cites; a review is about one project. Inheriting those skills is what
    # makes both findable by skill, which is the premise of outcome-aware ranking:
    # "show me phrasing that won work involving Celery".
    if not doc.skills and skills_by_project:
        sources = list(doc.cited_project_ids)
        if doc.project_id and doc.project_id not in sources:
            sources.append(doc.project_id)
        inherited: list[str] = []
        for project_id in sources:
            for skill in skills_by_project.get(project_id, ()):
                if skill not in inherited:
                    inherited.append(skill)
        doc.skills = inherited

    return doc


def load_persona_directory(root: Path, persona: str) -> list[RawDocument]:
    """Every document for one persona, from a directory laid out as the corpus is."""
    base = root / persona
    if not base.exists():
        raise FileNotFoundError(f"no portfolio directory at {base}")

    catalogue = load_evidence_index(root)
    # Path -> entry, for the README and code files that carry no frontmatter id.
    by_path = {
        f"{entry.get('persona')}/{entry.get('path')}": entry
        for entry in catalogue.values()
        if entry.get("path") and not str(entry["path"]).endswith(".json")
    }
    # Project -> its skills, so a proposal can inherit the skills of what it cited.
    skills_by_project: dict[str, list[str]] = {}
    for entry in catalogue.values():
        pid = entry.get("project_id")
        if pid and entry.get("skills"):
            bucket = skills_by_project.setdefault(pid, [])
            for skill in entry["skills"]:
                if skill not in bucket:
                    bucket.append(skill)

    docs: list[RawDocument] = []

    for markdown in sorted((base / "case_studies").glob("*.md")) if (base / "case_studies").exists() else []:
        docs.append(load_markdown(markdown, persona))

    cv_md = base / "cv.md"
    if cv_md.exists():
        docs.append(load_markdown(cv_md, persona))
    for pdf in sorted(base.glob("*.pdf")):
        try:
            docs.append(load_pdf(pdf, persona))
        except RuntimeError as exc:
            # One unreadable CV should not abort ingesting the whole portfolio.
            print(f"  skipped {pdf.name}: {exc}")

    repos = base / "repos"
    if repos.exists():
        for repo in sorted(p for p in repos.iterdir() if p.is_dir()):
            readme = next((p for p in repo.glob("README*")), None)
            if readme:
                docs.append(load_markdown(readme, persona))
            for source in sorted(repo.iterdir()):
                if source.suffix in CODE_SUFFIXES:
                    docs.append(load_code(source, persona))

    for name, doc_type in (("proposals.json", "proposal"), ("reviews.json", "review")):
        path = base / name
        if path.exists():
            docs.extend(load_json_collection(path, doc_type, persona))

    # READMEs and code have no frontmatter; the catalogue is where their skills live.
    return [enrich(doc, catalogue, by_path, skills_by_project) for doc in docs]
