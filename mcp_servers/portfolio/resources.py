"""Resource templates, so a host can load a full portfolio item by URI.

A1 asks for `portfolio://case/{id}` and `portfolio://proposal/{id}`. Resources differ from
tools in intent: a tool is an action the model decides to take, a resource is content the
*host* attaches to context, often because the user picked it. Claude Desktop or Cursor can
list these and let a freelancer drop a case study into a chat without the model having to
search for it first.

Three extra templates are here because they answer questions a freelancer actually asks of
their own portfolio, which is the bar A1 sets for the server being useful outside GigBid.
"""

from __future__ import annotations

import json
from typing import Any

from mcp_servers.portfolio import corpus


def _header(item: corpus.EvidenceItem) -> str:
    """A readable provenance block above the text.

    A resource gets pasted into a model's context with no surrounding schema, so the
    metadata has to travel inside the content or it is lost.
    """
    bits = [f"# {item.project or item.id}", ""]
    meta = {
        "citation": item.citation,
        "type": item.type,
        "persona": item.persona,
        "date": item.date,
        "client_industry": item.client_industry,
        "outcome": item.outcome,
        "skills": ", ".join(item.skills) or None,
        "source": item.source,
    }
    for key, value in meta.items():
        if value:
            bits.append(f"- **{key}**: {value}")
    bits.append("")
    bits.append("---")
    bits.append("")
    return "\n".join(bits)


def case_resource(case_id: str) -> str:
    """portfolio://case/{case_id} - one case study in full."""
    item = corpus.find_item_any_persona(case_id)
    return _header(item) + item.text


def proposal_resource(proposal_id: str) -> str:
    """portfolio://proposal/{proposal_id} - one past proposal and how it landed."""
    item = corpus.find_item_any_persona(proposal_id)
    note = ""
    if item.outcome:
        # The outcome is the point of a past proposal. Saying it plainly stops a model
        # treating an ignored proposal as a template worth copying.
        verdict = {
            "won": "This proposal won the job.",
            "replied": "This proposal got a reply but no contract.",
            "ignored": "This proposal was ignored. Do not copy its approach.",
        }.get(item.outcome, f"Outcome: {item.outcome}")
        note = f"\n\n> **Outcome:** {verdict}\n"
    cited = ""
    if item.cited_project_ids:
        cited = "\n\n> **Evidence cited:** " + ", ".join(item.cited_project_ids) + "\n"
    return _header(item) + item.text + note + cited


def profile_resource(persona: str) -> str:
    """portfolio://profile/{persona} - rates, floor, availability, declared skills."""
    profile = corpus.load_profile(persona)
    return json.dumps(profile, indent=2, ensure_ascii=False)


def project_resource(project_id: str) -> str:
    """portfolio://project/{project_id} - every description of one project at once.

    The corpus deliberately describes the same project as a case study, a README, code and
    a client review. V5 exists because any one of those alone can look irrelevant. This
    resource is the human-facing version of that argument.
    """
    for persona in corpus.VALID_PERSONAS:
        items = [i for i in corpus.load_corpus(persona) if i.project_id == project_id]
        if not items:
            continue
        order = {"case_study": 0, "readme": 1, "code": 2, "review": 3, "proposal": 4}
        items.sort(key=lambda i: order.get(i.type, 9))
        parts = [f"# Project {project_id} ({persona})", ""]
        for item in items:
            parts.append(f"## {item.type}: {item.id}")
            parts.append(f"*source: {item.source}* | *citation: {item.citation}*")
            parts.append("")
            parts.append(item.text.strip())
            parts.append("")
        return "\n".join(parts)
    raise corpus.PortfolioError(f"no project {project_id!r} in any persona")


def wins_resource(persona: str) -> str:
    """portfolio://wins/{persona} - the proposals that won, newest first.

    What actually worked, for the Writer to learn phrasing from. This is the read-only
    ancestor of the outcome memory in C2.
    """
    items = [
        i
        for i in corpus.load_corpus(persona)
        if i.type == "proposal" and i.outcome == "won"
    ]
    items.sort(key=lambda i: i.date or "", reverse=True)
    if not items:
        return f"No won proposals recorded for {persona}."
    parts = [f"# Won proposals - {persona} ({len(items)})", ""]
    for item in items[:40]:
        parts.append(f"## {item.id} ({item.date})")
        parts.append(f"*cites: {', '.join(item.cited_project_ids) or 'nothing'}*")
        parts.append("")
        parts.append(item.text.strip())
        parts.append("")
    return "\n".join(parts)


TEMPLATES: dict[str, dict[str, Any]] = {
    "portfolio://case/{case_id}": {
        "fn": case_resource,
        "name": "case_study",
        "description": "One case study in full, with provenance metadata.",
    },
    "portfolio://proposal/{proposal_id}": {
        "fn": proposal_resource,
        "name": "past_proposal",
        "description": "One past proposal, its cited evidence and whether it won.",
    },
    "portfolio://project/{project_id}": {
        "fn": project_resource,
        "name": "project_all_views",
        "description": "Every description of one project: case study, README, code, review.",
    },
    "portfolio://profile/{persona}": {
        "fn": profile_resource,
        "name": "profile",
        "description": "Rates, minimum budget, availability and declared skills.",
    },
    "portfolio://wins/{persona}": {
        "fn": wins_resource,
        "name": "won_proposals",
        "description": "Past proposals that won, to learn phrasing that worked.",
    },
}
