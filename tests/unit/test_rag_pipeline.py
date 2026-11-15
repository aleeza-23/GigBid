"""Ingestion, chunking and the variant registry (B1, B2).

The load-bearing tests here are the metadata ones. B1 requires metadata on every chunk,
and the first run of the pipeline shipped 205 of 221 chunks with no skills at all, because
READMEs and source files carry no frontmatter and the catalogue keys them by path rather
than by id. That is the kind of gap that quietly costs retrieval quality and shows up
nowhere until an evaluation.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mcp_servers.portfolio.corpus import load_corpus
from rag.ingest.chunker import chunk_document, chunk_prose
from rag.ingest.loaders import RawDocument, load_persona_directory, parse_frontmatter
from rag.ingest.pipeline import ingest, stats
from rag.retriever import get_ranker, list_variants

PERSONA = "ai_ml_engineer"
PERSONAS_DIR = Path(__file__).resolve().parents[2] / "data" / "personas"


# ------------------------------------------------------------------------ loaders


def test_frontmatter_is_parsed_off_markdown() -> None:
    meta, body = parse_frontmatter(
        '---\nid: "ai-p01-case"\ntype: "case_study"\nskills: ["Python", "Celery"]\n---\n\n# Title\n\nBody.'
    )
    assert meta["id"] == "ai-p01-case"
    assert meta["skills"] == ["Python", "Celery"]
    assert body.lstrip().startswith("# Title")
    assert "---" not in body


def test_every_persona_source_type_is_loaded() -> None:
    docs = load_persona_directory(PERSONAS_DIR, PERSONA)
    types = {d.type for d in docs}
    # B1 names each of these explicitly.
    assert {"case_study", "readme", "code", "review", "proposal", "cv"} <= types


def test_every_loaded_document_carries_skill_metadata() -> None:
    """Regression: READMEs, code, proposals and reviews all shipped with no skills.

    READMEs and code have no frontmatter, so they are matched to the catalogue by path.
    Proposals inherit from the projects they cite; reviews from the project they describe.
    """
    docs = load_persona_directory(PERSONAS_DIR, PERSONA)
    without = [d for d in docs if not d.skills]
    assert not without, f"{len(without)} documents have no skills, e.g. {[d.id for d in without[:4]]}"


def test_readmes_and_code_resolve_to_catalogue_ids() -> None:
    """A path-matched README should adopt its catalogue id, not a path-derived one."""
    docs = load_persona_directory(PERSONAS_DIR, PERSONA)
    readmes = [d for d in docs if d.type == "readme"]
    assert readmes
    assert any(d.id.endswith("-readme") for d in readmes), [d.id for d in readmes[:3]]
    assert all(d.project_id for d in readmes), "a README belongs to a project"


# ----------------------------------------------------------------------- chunking


def test_prose_splits_on_headings_not_character_counts() -> None:
    """"## Results" must not be separated from the numbers under it."""
    text = "# Project\n\nIntro para.\n\n## Results\n\nAccuracy rose from 71% to 89%.\n"
    chunks = chunk_prose(text, max_tokens=500, overlap_tokens=0)
    results = [body for heading, body in chunks if heading == "Results"]
    assert results
    assert "89%" in results[0]


def test_code_chunks_keep_whole_definitions() -> None:
    code = (
        "import celery\n\n"
        "def enqueue(task):\n    return celery.send(task)\n\n"
        "def dequeue():\n    return celery.get()\n"
    )
    doc = RawDocument(id="x-code", text=code, type="code", source="x.py", language="python")
    chunks = chunk_document(doc, max_tokens=500)
    bodies = [c.text for c in chunks]

    # The module header is its own chunk: it names the library, which is what a lexical
    # retriever matches "Celery" against.
    assert any("import celery" in b for b in bodies)
    enqueue = next(b for b in bodies if "def enqueue" in b)
    assert "return celery.send(task)" in enqueue, "a definition must not be split mid-body"


def test_short_documents_are_not_split() -> None:
    """A 150-word proposal loses its shape if chunked."""
    doc = RawDocument(
        id="ai-prop-001",
        text="Your post asks for NLP work; I built exactly that. " * 4,
        type="proposal",
        source="proposals.json",
        outcome="won",
    )
    chunks = chunk_document(doc, max_tokens=50)
    assert len(chunks) == 1
    assert chunks[0].id == "ai-prop-001", "a single chunk keeps the parent id so citations hold"


def test_chunks_carry_provenance_back_to_their_document() -> None:
    doc = RawDocument(
        id="ai-p01-case",
        text="# A\n\n" + ("word " * 400) + "\n\n## B\n\n" + ("word " * 400),
        type="case_study",
        source="case_studies/ai-p01.md",
        project_id="ai-p01",
        skills=["Python"],
    )
    chunks = chunk_document(doc, max_tokens=120)
    assert len(chunks) > 1
    for index, chunk in enumerate(chunks):
        assert chunk.parent_id == "ai-p01-case"
        assert chunk.chunk_index == index
        assert chunk.citation == "portfolio://case_study/ai-p01-case"
        assert chunk.project_id == "ai-p01"
        assert chunk.skills == ["Python"]


# ---------------------------------------------------------------------- pipeline


def test_reingest_is_incremental(tmp_path) -> None:
    """B1's incremental re-indexing: an unchanged corpus must re-embed nothing."""
    first = ingest(PERSONA, index_dir=tmp_path)
    assert first["added"] > 0
    assert first["reused"] == 0

    second = ingest(PERSONA, index_dir=tmp_path)
    assert second["added"] == 0
    assert second["changed"] == 0
    assert second["reused"] == first["added"]
    assert second["chunks"] == first["chunks"]
    assert second["incremental"] is True


def test_force_rebuilds_everything(tmp_path) -> None:
    ingest(PERSONA, index_dir=tmp_path)
    forced = ingest(PERSONA, index_dir=tmp_path, force=True)
    assert forced["reused"] == 0


def test_stats_counts_outcomes_only_for_proposals(tmp_path) -> None:
    """Case studies carry "delivered", which is not a proposal outcome."""
    ingest(PERSONA, index_dir=tmp_path)
    report = stats(PERSONA, index_dir=tmp_path)
    assert set(report["proposal_outcomes"]) <= {"won", "replied", "ignored"}
    assert report["chunks_missing_skills"] == 0


# ---------------------------------------------------------------------- variants


def test_the_required_variant_groups_are_all_implemented() -> None:
    """B2: V0 mandatory, plus every group marked required."""
    groups = {spec["group"] for spec in list_variants()}
    ids = {spec["id"] for spec in list_variants()}

    assert "V0" in ids, "V0 is mandatory"
    assert "baseline" in groups
    assert "query" in groups, "V1 or V7 required"
    assert "retrieval" in groups, "V2 and V3 required"
    assert "agentic" in groups, "V8 or V9 required"
    assert len(ids) >= 5, f"V0 plus four more required, found {sorted(ids)}"


@pytest.mark.parametrize("variant_id", ["V0", "V1", "V2", "V3", "V8"])
def test_each_variant_returns_scored_items(variant_id: str) -> None:
    """The contract the Portfolio server's swap point relies on."""
    items = load_corpus(PERSONA)
    ranker = get_ranker(variant_id)
    assert ranker is not None, f"{variant_id} did not resolve"

    hits = ranker("fine-tune a transformer for text classification", items, 4)
    assert hits, f"{variant_id} returned nothing"
    assert len(hits) <= 4
    for hit in hits:
        assert hit.score is not None, "a ranker must set a score"
        assert hit.id and hit.citation.startswith("portfolio://")


def test_variants_can_be_rescored_without_losing_annotations() -> None:
    """Regression: V3 crashed copying a V2 result, because annotations were attributes.

    V2 marks which retriever found each item; V3 then re-scores those items. With the
    annotation set as an instance attribute rather than a declared field, the dataclass
    copy raised TypeError and V3 was dead on arrival.
    """
    items = load_corpus(PERSONA)
    hits = get_ranker("V3")("build a Celery task queue", items, 3)
    assert hits
    assert "reranker" in hits[0].extra
    assert "found_by" in hits[0].extra, "V2's provenance should survive V3's re-scoring"


def test_the_corrective_loop_reports_whether_evidence_is_sufficient() -> None:
    """V8's refusal is what lets the graph skip rather than claim (B2, agentic group)."""
    from rag.variants.v8_corrective import retrieve_with_report

    items = load_corpus(PERSONA)

    strong = retrieve_with_report("fine-tune a transformer for text classification", items, 4)
    assert strong["sufficient"] is True
    assert strong["should_skip"] is False

    absurd = retrieve_with_report("operate a commercial fishing trawler in Alaska", items, 4)
    assert absurd["sufficient"] is False, absurd["reason"]
    assert absurd["should_skip"] is True
    assert absurd["rounds"] >= 2, "a weak retrieval should trigger at least one re-query"
