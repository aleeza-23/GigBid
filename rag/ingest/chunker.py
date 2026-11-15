"""Chunking, with the metadata B1 requires carried onto every chunk.

Three strategies, because one size genuinely does not fit this corpus:

* **prose** - split on Markdown headings first, then pack paragraphs up to a token budget
  with a small overlap. Headings are the author's own structure; ignoring them to cut at
  character 800 is how "## Results" gets separated from the numbers under it.
* **code** - split on definition boundaries so a chunk is a whole function with its
  signature. A chunk beginning mid-body cannot support a claim about what the code does.
* **atomic** - proposals and reviews are short and self-contained. Splitting a 150-word
  proposal destroys the thing that makes it useful, which is its shape as a whole.

Every chunk keeps its parent's metadata plus `chunk_index` and `parent_id`, so a claim
citing a chunk can be traced back to the document it came from.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from rag.ingest.loaders import RawDocument

# Roughly four characters per token. Budgets are in tokens because that is what costs
# money downstream; the conversion is approximate on purpose.
DEFAULT_CHUNK_TOKENS = 220
DEFAULT_OVERLAP_TOKENS = 40
CHARS_PER_TOKEN = 4

_HEADING = re.compile(r"^(#{1,4})\s+(.+)$", re.MULTILINE)

# Definition starts, per language. Deliberately simple: a real parser would be better but
# would add a dependency per language for a corpus of short files.
_DEF_PATTERNS = {
    "python": re.compile(r"^(?:async\s+)?(?:def|class)\s+\w+", re.MULTILINE),
    "typescript": re.compile(
        r"^(?:export\s+)?(?:async\s+)?(?:function|class|const|interface|type)\s+\w+",
        re.MULTILINE,
    ),
    "javascript": re.compile(
        r"^(?:export\s+)?(?:async\s+)?(?:function|class|const)\s+\w+", re.MULTILINE
    ),
    "sql": re.compile(r"^(?:with|select|create|insert)\b", re.MULTILINE | re.IGNORECASE),
    "go": re.compile(r"^func\s+\w+", re.MULTILINE),
    "java": re.compile(r"^\s*(?:public|private|protected).*\(", re.MULTILINE),
    "r": re.compile(r"^\w+\s*<-\s*function", re.MULTILINE),
}

ATOMIC_TYPES = {"proposal", "review"}


@dataclass
class Chunk:
    """One indexable unit, with full provenance."""

    id: str
    text: str
    parent_id: str
    chunk_index: int
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
    heading: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def citation(self) -> str:
        return f"portfolio://{self.type}/{self.parent_id}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "parent_id": self.parent_id,
            "chunk_index": self.chunk_index,
            "citation": self.citation,
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
            "heading": self.heading,
        }


def _budget_chars(tokens: int) -> int:
    return tokens * CHARS_PER_TOKEN


def _pack(
    pieces: list[tuple[str | None, str]], max_chars: int, overlap_chars: int
) -> list[tuple[str | None, str]]:
    """Pack (heading, text) pieces into chunks under max_chars, with overlap."""
    out: list[tuple[str | None, str]] = []
    heading: str | None = None
    buffer = ""

    for piece_heading, text in pieces:
        if not text.strip():
            continue
        if piece_heading:
            heading = piece_heading
        if len(buffer) + len(text) + 2 <= max_chars:
            buffer = f"{buffer}\n\n{text}".strip()
            continue
        if buffer:
            out.append((heading, buffer))
            # Carry the tail forward so a sentence spanning a boundary survives in one
            # of the two chunks.
            buffer = buffer[-overlap_chars:] if overlap_chars else ""
        # A single piece larger than the budget is hard-split rather than dropped.
        while len(text) > max_chars:
            out.append((heading, text[:max_chars]))
            text = text[max_chars - overlap_chars :]
        buffer = f"{buffer}\n\n{text}".strip() if buffer else text

    if buffer.strip():
        out.append((heading, buffer))
    return out


def chunk_prose(text: str, max_tokens: int, overlap_tokens: int) -> list[tuple[str | None, str]]:
    """Split on headings, then pack paragraphs."""
    matches = list(_HEADING.finditer(text))
    sections: list[tuple[str | None, str]] = []

    if not matches:
        sections = [(None, para) for para in re.split(r"\n\s*\n", text) if para.strip()]
    else:
        if matches[0].start() > 0:
            preamble = text[: matches[0].start()].strip()
            if preamble:
                sections.append((None, preamble))
        for i, match in enumerate(matches):
            heading = match.group(2).strip()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            body = text[match.end() : end].strip()
            for para in re.split(r"\n\s*\n", body):
                if para.strip():
                    sections.append((heading, para.strip()))

    return _pack(sections, _budget_chars(max_tokens), _budget_chars(overlap_tokens))


def chunk_code(text: str, language: str | None, max_tokens: int) -> list[tuple[str | None, str]]:
    """Split on definition boundaries, keeping each definition whole."""
    pattern = _DEF_PATTERNS.get(language or "")
    if pattern is None:
        return _pack([(None, text)], _budget_chars(max_tokens), 0)

    starts = [m.start() for m in pattern.finditer(text)]
    if not starts:
        return _pack([(None, text)], _budget_chars(max_tokens), 0)

    # Keep the module header (imports, docstring) as its own chunk: it names the libraries
    # used, which is exactly what a lexical retriever needs to match "Celery".
    pieces: list[tuple[str | None, str]] = []
    if starts[0] > 0:
        header = text[: starts[0]].strip()
        if header:
            pieces.append(("module header", header))

    bounds = starts + [len(text)]
    for i in range(len(starts)):
        body = text[bounds[i] : bounds[i + 1]].strip()
        if not body:
            continue
        first_line = body.split("\n", 1)[0][:80]
        # No overlap between definitions: they are already self-contained units.
        if len(body) <= _budget_chars(max_tokens):
            pieces.append((first_line, body))
        else:
            pieces.extend(
                (first_line, body[j : j + _budget_chars(max_tokens)])
                for j in range(0, len(body), _budget_chars(max_tokens))
            )
    return pieces


def chunk_document(
    doc: RawDocument,
    max_tokens: int = DEFAULT_CHUNK_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
) -> list[Chunk]:
    """Chunk one document with the strategy its type calls for."""
    text = doc.text.strip()
    if not text:
        return []

    if doc.type in ATOMIC_TYPES:
        pieces: list[tuple[str | None, str]] = [(None, text)]
    elif doc.type == "code":
        pieces = chunk_code(text, doc.language, max_tokens)
    else:
        pieces = chunk_prose(text, max_tokens, overlap_tokens)

    chunks: list[Chunk] = []
    for index, (heading, body) in enumerate(pieces):
        chunks.append(
            Chunk(
                # Single-chunk documents keep the parent id, so citations stay stable for
                # the proposals and reviews that make up most of the corpus.
                id=doc.id if len(pieces) == 1 else f"{doc.id}#{index}",
                text=body.strip(),
                parent_id=doc.id,
                chunk_index=index,
                type=doc.type,
                source=doc.source,
                persona=doc.persona,
                project=doc.project,
                project_id=doc.project_id,
                skills=list(doc.skills),
                date=doc.date,
                client_industry=doc.client_industry,
                outcome=doc.outcome,
                language=doc.language,
                heading=heading,
                extra=dict(doc.extra),
            )
        )
    return chunks


def chunk_all(
    docs: list[RawDocument],
    max_tokens: int = DEFAULT_CHUNK_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
) -> list[Chunk]:
    out: list[Chunk] = []
    for doc in docs:
        out.extend(chunk_document(doc, max_tokens, overlap_tokens))
    return out
