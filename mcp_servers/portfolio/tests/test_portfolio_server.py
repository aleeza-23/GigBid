"""The Portfolio server, exercised through a real in-memory MCP client (A7).

These tests go through the MCP protocol rather than calling the Python functions, so they
cover what calling the functions cannot: that tools are actually registered, that their
schemas validate input, that annotations are advertised, and that resources and prompts
resolve. A server whose functions work but whose schema is malformed is broken for every
host, and only a protocol-level test notices.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from mcp.shared.memory import create_connected_server_and_client_session

from mcp_servers.portfolio import corpus
from mcp_servers.portfolio.server import mcp

PERSONA = "ai_ml_engineer"


async def _session():
    return create_connected_server_and_client_session(mcp._mcp_server)


def _payload(result: Any) -> dict[str, Any]:
    """Pull the structured result out of a tool call."""
    if getattr(result, "structuredContent", None):
        return result.structuredContent
    text = result.content[0].text
    return json.loads(text)


@pytest.mark.asyncio
async def test_the_three_required_tools_are_registered() -> None:
    async with await _session() as session:
        tools = {t.name: t for t in (await session.list_tools()).tools}

    for name in ("search_evidence", "get_case_study", "list_skills"):
        assert name in tools, f"A1 requires {name}"

    # The description is the tool's real interface: it is what the model reads when
    # choosing. An empty one is a bug even though nothing crashes.
    for name, tool in tools.items():
        assert tool.description and len(tool.description) > 40, f"{name} needs a real description"


@pytest.mark.asyncio
async def test_every_tool_declares_itself_read_only() -> None:
    """A1 says the Portfolio server is read-only, so every annotation must say so."""
    async with await _session() as session:
        tools = (await session.list_tools()).tools

    for tool in tools:
        ann = tool.annotations
        assert ann is not None, f"{tool.name} has no annotations"
        assert ann.readOnlyHint is True, f"{tool.name} must be read-only"
        assert ann.destructiveHint is False, f"{tool.name} must not be destructive"


@pytest.mark.asyncio
async def test_search_returns_citable_evidence() -> None:
    async with await _session() as session:
        result = await session.call_tool(
            "search_evidence",
            {"requirement": "fine-tune a transformer for text classification", "persona": PERSONA, "k": 5},
        )
    data = _payload(result)
    assert data["hits"], "a core requirement should find evidence"
    assert len(data["hits"]) <= 5

    for hit in data["hits"]:
        # R5 hangs on this: a claim must be traceable to the thing that supports it.
        assert hit["citation"].startswith("portfolio://")
        assert hit["id"] and hit["type"] and hit["source"]
        assert hit["text"].strip(), "evidence with no text cannot support a claim"


@pytest.mark.asyncio
async def test_the_schema_rejects_bad_arguments() -> None:
    """Constraints in the schema, not just in the docstring."""
    async with await _session() as session:
        # k above the declared maximum.
        result = await session.call_tool(
            "search_evidence", {"requirement": "python", "persona": PERSONA, "k": 500}
        )
        assert result.isError, "k=500 should be refused by the schema"

        # A persona that does not exist.
        result = await session.call_tool(
            "search_evidence", {"requirement": "python", "persona": "astronaut"}
        )
        assert result.isError, "an unknown persona should be refused"

        # Too short to be a requirement.
        result = await session.call_tool(
            "search_evidence", {"requirement": "a", "persona": PERSONA}
        )
        assert result.isError, "a one-character requirement should be refused"


@pytest.mark.asyncio
async def test_filters_actually_narrow() -> None:
    async with await _session() as session:
        result = await session.call_tool(
            "search_evidence",
            {
                "requirement": "machine learning pipeline",
                "persona": PERSONA,
                "k": 10,
                "filters": {"types": ["case_study"]},
            },
        )
    data = _payload(result)
    assert data["hits"]
    assert {h["type"] for h in data["hits"]} == {"case_study"}


@pytest.mark.asyncio
async def test_outcome_filter_keeps_only_won_proposals() -> None:
    async with await _session() as session:
        result = await session.call_tool(
            "search_evidence",
            {
                "requirement": "recommendation system",
                "persona": PERSONA,
                "k": 10,
                "filters": {"types": ["proposal"], "outcomes": ["won"]},
            },
        )
    data = _payload(result)
    assert all(h["outcome"] == "won" for h in data["hits"])


@pytest.mark.asyncio
async def test_case_study_resolves_by_evidence_id_and_project_id() -> None:
    """Agents conflate the two constantly; accepting both is tool design, not leniency."""
    async with await _session() as session:
        by_evidence = _payload(await session.call_tool("get_case_study", {"case_id": "ai-p01-case"}))
        by_project = _payload(await session.call_tool("get_case_study", {"case_id": "ai-p01"}))

    assert by_evidence["id"] == by_project["id"] == "ai-p01-case"
    assert by_evidence["text"].strip()
    # The same project described several ways is the point of this corpus (V5).
    assert {r["type"] for r in by_evidence["related"]} >= {"readme", "code"}


@pytest.mark.asyncio
async def test_a_missing_case_study_is_an_error_not_an_empty_answer() -> None:
    async with await _session() as session:
        result = await session.call_tool("get_case_study", {"case_id": "does-not-exist"})
    assert result.isError


@pytest.mark.asyncio
async def test_list_skills_separates_declared_from_evidenced() -> None:
    async with await _session() as session:
        data = _payload(await session.call_tool("list_skills", {"persona": PERSONA}))

    assert data["skills"]
    assert "unbacked_skills" in data, "R5 needs the list of skills with nothing behind them"
    assert data["hourly_rate_usd"], "the Pricing Strategist needs the floor"

    top = data["skills"][0]
    assert top["evidence_count"] > 0
    assert top["evidence_ids"], "a skill claim must name its evidence"


@pytest.mark.asyncio
async def test_min_evidence_filters_weak_skills() -> None:
    async with await _session() as session:
        loose = _payload(await session.call_tool("list_skills", {"persona": PERSONA, "min_evidence": 0}))
        strict = _payload(await session.call_tool("list_skills", {"persona": PERSONA, "min_evidence": 5}))

    assert len(strict["skills"]) < len(loose["skills"])
    assert all(s["evidence_count"] >= 5 for s in strict["skills"])


@pytest.mark.asyncio
async def test_required_resource_templates_exist_and_resolve() -> None:
    async with await _session() as session:
        templates = {t.uriTemplate for t in (await session.list_resource_templates()).resourceTemplates}

        assert "portfolio://case/{case_id}" in templates, "A1 requires portfolio://case/{id}"
        assert "portfolio://proposal/{proposal_id}" in templates, (
            "A1 requires portfolio://proposal/{id}"
        )

        contents = (await session.read_resource("portfolio://case/ai-p01-case")).contents
        body = contents[0].text
        assert "ai-p01-case" in body
        assert "Support ticket triage" in body


@pytest.mark.asyncio
async def test_proposal_resource_states_the_outcome() -> None:
    """A past proposal without its outcome is a template waiting to be copied badly."""
    ignored = next(
        i
        for i in corpus.load_corpus(PERSONA)
        if i.type == "proposal" and i.outcome == "ignored"
    )
    async with await _session() as session:
        body = (await session.read_resource(f"portfolio://proposal/{ignored.id}")).contents[0].text

    assert "ignored" in body.lower()
    assert "do not copy" in body.lower()


@pytest.mark.asyncio
async def test_prompts_are_registered_and_render() -> None:
    async with await _session() as session:
        prompts = {p.name for p in (await session.list_prompts()).prompts}
        assert "answer_screening_question" in prompts, "A1 names this one explicitly"
        assert len(prompts) >= 4

        rendered = await session.get_prompt(
            "answer_screening_question",
            {"question": "Have you deployed a RAG system in production?", "persona": PERSONA},
        )
    text = rendered.messages[0].content.text
    assert "RAG system in production" in text
    # Installable in a host with no Compliance Guard behind it, so the grounding rule has
    # to travel inside the prompt.
    assert "search_evidence" in text
    assert "portfolio does not support" in text


@pytest.mark.asyncio
async def test_health_reports_all_three_personas() -> None:
    async with await _session() as session:
        data = _payload(await session.call_tool("portfolio_health", {}))

    assert data["status"] == "ok"
    for persona in corpus.VALID_PERSONAS:
        assert data["personas"][persona]["items"] > 0


def test_bm25_prefers_exact_library_names_over_prose() -> None:
    """Why the baseline is BM25 and not cosine (V2's argument, before V2 exists).

    'Celery' must beat a project that merely talks about background processing. A dense
    embedding blurs the two; that blurring is the thing Part B has to measure.
    """
    from mcp_servers.portfolio.tools import search_evidence

    hits = search_evidence("Celery", PERSONA, k=5)["hits"]
    assert hits, "an exact library name should retrieve something"
    assert any("Celery" in (h["skills"] or []) for h in hits[:3])
