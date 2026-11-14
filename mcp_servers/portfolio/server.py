"""Portfolio MCP server (A1): read-only access to a freelancer's evidence.

Built on the official MCP Python SDK. Three tools, five resource templates and five
prompts, all read-only, which is why every annotation here says readOnlyHint=True and
destructiveHint=False and why there are no idempotency keys: nothing on this server can
change anything.

Run it:

    python -m mcp_servers.portfolio.server              # stdio, for Claude Desktop
    python -m mcp_servers.portfolio.server --http       # Streamable HTTP on :8701
    npx @modelcontextprotocol/inspector python -m mcp_servers.portfolio.server

A1 asks that this be good enough for another freelancer to plug into their own host. It
is: it depends on nothing in GigBid, reads only `data/personas/`, and the install command
is in mcp_servers/portfolio/README.md.
"""

from __future__ import annotations

import argparse
import os
import time
from typing import Any

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations

from mcp_servers._common.audit import AuditLog
from mcp_servers._common.trace import get_trace_id
from mcp_servers.portfolio import prompts as prompt_templates
from mcp_servers.portfolio import resources as resource_templates
from mcp_servers.portfolio import tools as portfolio_tools
from mcp_servers.portfolio.corpus import PortfolioError

SERVER_NAME = "gigbid-portfolio"
INSTRUCTIONS = """Read-only access to a freelancer's portfolio: case studies, repository
READMEs and code, client reviews, a CV, and past proposals labelled with whether they won.

Search one requirement at a time. The corpus describes each project several ways on
purpose, so a query blending five requirements returns the average of all of them and
nothing precisely. Every item comes back with a `citation` handle; a claim in a proposal
must carry the citation of the evidence that supports it, and anything the portfolio does
not support must not be claimed at all."""

audit = AuditLog(server=SERVER_NAME)
mcp = FastMCP(SERVER_NAME, instructions=INSTRUCTIONS)

# Read-only throughout. openWorldHint=False because the answer comes from a fixed local
# corpus: the same call returns the same thing until the user adds work, which matters to
# a host deciding whether a result is cacheable.
READ_ONLY = ToolAnnotations(
    readOnlyHint=True,
    destructiveHint=False,
    idempotentHint=True,
    openWorldHint=False,
)


def _audited(tool_name: str, fn, **kwargs: Any) -> Any:
    """Run a tool, record it, and turn internal errors into useful messages.

    A PortfolioError is the model's mistake (a bad id, an unknown persona) and its text is
    actionable, so it is surfaced. Anything else is our bug and the detail goes to the
    audit log, not to the model, which cannot act on a traceback anyway.
    """
    started = time.perf_counter()
    try:
        result = fn(**kwargs)
    except PortfolioError as exc:
        audit.record(
            tool_name,
            kwargs,
            status="rejected",
            error=str(exc),
            duration_ms=(time.perf_counter() - started) * 1000,
        )
        raise ValueError(str(exc)) from exc
    except Exception as exc:
        audit.record(
            tool_name,
            kwargs,
            status="error",
            error=f"{type(exc).__name__}: {exc}",
            duration_ms=(time.perf_counter() - started) * 1000,
        )
        raise ValueError(f"portfolio server failed handling {tool_name}") from exc

    audit.record(
        tool_name,
        kwargs,
        result=_summarise(tool_name, result),
        duration_ms=(time.perf_counter() - started) * 1000,
    )
    return result


def _summarise(tool_name: str, result: Any) -> Any:
    """What to put in the audit log. Full case-study text would swamp it."""
    if not isinstance(result, dict):
        return result
    if tool_name == "search_evidence":
        return {
            "total_candidates": result.get("total_candidates"),
            "ranker": result.get("ranker"),
            "hit_ids": [h.get("id") for h in result.get("hits", [])],
            "scores": [h.get("score") for h in result.get("hits", [])],
        }
    if tool_name == "get_case_study":
        return {"id": result.get("id"), "chars": len(result.get("text") or "")}
    if tool_name == "list_skills":
        return {
            "skills": len(result.get("skills", [])),
            "unbacked": result.get("unbacked_skills"),
        }
    return result


# ----------------------------------------------------------------------- tools
# Docstrings and parameter descriptions are the tool's real interface: they are what the
# model reads when choosing. R&D question A measures exactly this, and the "best set with
# rewritten descriptions" arm of that study is these three with sharper guidance.


@mcp.tool(annotations=READ_ONLY, structured_output=True)
def search_evidence(
    requirement: portfolio_tools.RequirementArg,
    persona: portfolio_tools.PersonaArg,
    k: portfolio_tools.KArg = 5,
    filters: dict[str, Any] | None = None,
    outcome_aware: portfolio_tools.OutcomeAwareArg = False,
    text_limit: portfolio_tools.TextLimitArg = portfolio_tools.DEFAULT_TEXT_LIMIT,
) -> dict[str, Any]:
    """Find portfolio evidence supporting ONE requirement.

    Pass a single requirement in the client's own words, not the whole job post. Returns
    evidence items with type, source, date and outcome, each with a `citation` to quote in
    a proposal. Filter by type, skill, date, industry or proposal outcome. Set
    outcome_aware=true to prefer phrasing from proposals that won.
    """
    parsed = portfolio_tools.EvidenceFilters(**filters) if filters else None
    return _audited(
        "search_evidence",
        portfolio_tools.search_evidence,
        requirement=requirement,
        persona=persona,
        k=k,
        filters=parsed,
        outcome_aware=outcome_aware,
        text_limit=text_limit,
    )


@mcp.tool(annotations=READ_ONLY, structured_output=True)
def get_case_study(
    case_id: portfolio_tools.CaseIdArg,
    persona: portfolio_tools.PersonaName | None = None,
) -> dict[str, Any]:
    """Load one case study in full, plus the repo, code and review for the same project.

    Use after search_evidence points at a case study and you need the whole thing,
    including the numbers in its results section. Accepts an evidence id ('ai-p01-case')
    or a project id ('ai-p01').
    """
    return _audited(
        "get_case_study", portfolio_tools.get_case_study, case_id=case_id, persona=persona
    )


@mcp.tool(annotations=READ_ONLY, structured_output=True)
def list_skills(
    persona: portfolio_tools.PersonaArg,
    min_evidence: portfolio_tools.MinEvidenceArg = 0,
) -> dict[str, Any]:
    """List the persona's skills and how much evidence backs each one.

    Returns evidence_count and won_proposals per skill, plus `unbacked_skills`: skills
    declared on the profile with nothing behind them. Never claim an unbacked skill.
    Call this before writing a proposal to check what you are allowed to assert.
    """
    return _audited(
        "list_skills", portfolio_tools.list_skills, persona=persona, min_evidence=min_evidence
    )


# ------------------------------------------------------------------- resources

for _uri, _spec in resource_templates.TEMPLATES.items():
    mcp.resource(_uri, name=_spec["name"], description=_spec["description"], mime_type="text/markdown")(
        _spec["fn"]
    )


# --------------------------------------------------------------------- prompts

for _name, _spec in prompt_templates.TEMPLATES.items():
    mcp.prompt(name=_name, title=_spec["title"], description=_spec["description"])(_spec["fn"])


# ---------------------------------------------------------------------- health


@mcp.tool(annotations=READ_ONLY, structured_output=True)
def portfolio_health() -> dict[str, Any]:
    """Check the corpus is loadable and report what is indexed.

    A6 requires a clear degraded mode. The UI calls this to decide whether to show the
    Portfolio server as available before a run starts rather than failing mid-graph.
    """
    from mcp_servers.portfolio.corpus import VALID_PERSONAS, load_corpus

    counts: dict[str, Any] = {}
    ok = True
    for persona in VALID_PERSONAS:
        try:
            items = load_corpus(persona)
            counts[persona] = {
                "items": len(items),
                "types": sorted({i.type for i in items}),
                "with_text": sum(1 for i in items if i.text.strip()),
            }
        except PortfolioError as exc:
            ok = False
            counts[persona] = {"error": str(exc)}
    return {"status": "ok" if ok else "degraded", "trace_id": get_trace_id(), "personas": counts}


def main() -> None:
    parser = argparse.ArgumentParser(description="GigBid Portfolio MCP server (read-only)")
    parser.add_argument(
        "--http",
        action="store_true",
        help="Serve Streamable HTTP instead of stdio (for the web app and remote hosts).",
    )
    parser.add_argument("--port", type=int, default=int(os.getenv("PORTFOLIO_MCP_PORT", "8701")))
    parser.add_argument("--host", default=os.getenv("PORTFOLIO_MCP_HOST", "127.0.0.1"))
    args = parser.parse_args()

    if args.http:
        mcp.settings.host = args.host
        mcp.settings.port = args.port
        mcp.run(transport="streamable-http")
    else:
        mcp.run()


if __name__ == "__main__":
    main()
