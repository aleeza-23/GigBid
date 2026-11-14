"""Schema snapshot tests: CI fails when a tool signature changes unintentionally (A7).

Every tool across the three servers is serialised to `snapshots/<server>.json` with its
name, input schema, annotations and description. A change to any of those changes what
every agent and every host sees, so it should be a deliberate act with a diff in the
commit, not something that slips out with an unrelated refactor.

Descriptions are included on purpose. R&D question A measures how much tool descriptions
change agent behaviour, so a silent reword is a silent change to the experiment.

Updating after an intended change:

    make snapshot-update      (or: UPDATE_SNAPSHOTS=1 pytest tests/contract)

Review the diff before committing it. That review is the entire point.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest
from mcp.shared.memory import create_connected_server_and_client_session

SNAPSHOT_DIR = Path(__file__).parent / "snapshots"
UPDATE = os.getenv("UPDATE_SNAPSHOTS", "").lower() in ("1", "true", "yes")

SERVERS = ("portfolio", "marketplace", "workspace")


def _server_module(name: str):
    if name == "portfolio":
        from mcp_servers.portfolio import server
    elif name == "marketplace":
        from mcp_servers.marketplace import server
    else:
        from mcp_servers.workspace import server
    return server


async def _capture(name: str) -> dict[str, Any]:
    """The full public surface of one server, in a stable, diffable shape."""
    module = _server_module(name)
    async with create_connected_server_and_client_session(module.mcp._mcp_server) as session:
        tools = (await session.list_tools()).tools

        try:
            templates = sorted(
                t.uriTemplate for t in (await session.list_resource_templates()).resourceTemplates
            )
        except Exception:
            templates = []
        try:
            prompts = sorted(p.name for p in (await session.list_prompts()).prompts)
        except Exception:
            prompts = []

    return {
        "server": module.SERVER_NAME,
        "tools": {
            tool.name: {
                "description": (tool.description or "").strip(),
                "input_schema": tool.inputSchema,
                "annotations": (
                    tool.annotations.model_dump(exclude_none=True) if tool.annotations else None
                ),
            }
            for tool in sorted(tools, key=lambda t: t.name)
        },
        "resource_templates": templates,
        "prompts": prompts,
    }


@pytest.mark.parametrize("name", SERVERS)
@pytest.mark.asyncio
async def test_tool_schemas_match_the_committed_snapshot(name: str) -> None:
    current = await _capture(name)
    path = SNAPSHOT_DIR / f"{name}.json"

    if UPDATE or not path.exists():
        SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(current, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if not UPDATE:
            pytest.skip(f"wrote the initial snapshot for {name}; commit it")
        return

    expected = json.loads(path.read_text(encoding="utf-8"))
    if current == expected:
        return

    # A bare "dicts differ" is useless in CI, so say exactly what moved.
    added = sorted(set(current["tools"]) - set(expected["tools"]))
    removed = sorted(set(expected["tools"]) - set(current["tools"]))
    changed = [
        tool
        for tool in sorted(set(current["tools"]) & set(expected["tools"]))
        if current["tools"][tool] != expected["tools"][tool]
    ]

    detail: list[str] = [f"{name} server's MCP surface changed."]
    if added:
        detail.append(f"  tools added:   {added}")
    if removed:
        detail.append(f"  tools removed: {removed}  <-- this breaks every caller")
    for tool in changed:
        before, after = expected["tools"][tool], current["tools"][tool]
        if before["annotations"] != after["annotations"]:
            detail.append(
                f"  {tool}: annotations {before['annotations']} -> {after['annotations']}"
            )
        if before["input_schema"] != after["input_schema"]:
            before_props = set(before["input_schema"].get("properties") or {})
            after_props = set(after["input_schema"].get("properties") or {})
            detail.append(
                f"  {tool}: schema changed"
                + (f", params added {sorted(after_props - before_props)}" if after_props - before_props else "")
                + (
                    f", params removed {sorted(before_props - after_props)}"
                    if before_props - after_props
                    else ""
                )
            )
        if before["description"] != after["description"]:
            detail.append(f"  {tool}: description reworded (affects tool selection)")
    if current["resource_templates"] != expected["resource_templates"]:
        detail.append(
            f"  resource templates: {expected['resource_templates']} -> "
            f"{current['resource_templates']}"
        )
    if current["prompts"] != expected["prompts"]:
        detail.append(f"  prompts: {expected['prompts']} -> {current['prompts']}")

    detail.append("")
    detail.append("If this was intentional: make snapshot-update, then review the diff.")
    pytest.fail("\n".join(detail))


@pytest.mark.parametrize("name", SERVERS)
@pytest.mark.asyncio
async def test_every_tool_is_documented_and_annotated(name: str) -> None:
    """Invariants that hold regardless of what the snapshot happens to contain.

    A tool with no annotations gives a host no basis for deciding whether to confirm it,
    and a tool with a one-line description is one a model will misuse. Both are the "wraps
    functions with no schemas" failure the rubric puts at the bottom of the scale.
    """
    captured = await _capture(name)
    for tool, spec in captured["tools"].items():
        assert spec["annotations"], f"{tool} has no annotations"
        for hint in ("readOnlyHint", "destructiveHint", "idempotentHint"):
            assert hint in spec["annotations"], f"{tool} does not declare {hint}"
        assert len(spec["description"]) >= 60, f"{tool}'s description is too thin to guide a model"
        assert spec["input_schema"]["type"] == "object"


@pytest.mark.asyncio
async def test_only_the_marketplace_can_write() -> None:
    """The write surface of the whole system, asserted in one place.

    Part A's safety argument rests on a small, known set of tools that can reach the
    outside world. This test is where that set is pinned: a new write tool anywhere fails
    here and has to be added deliberately.
    """
    writers: dict[str, list[str]] = {}
    for name in SERVERS:
        captured = await _capture(name)
        writers[name] = sorted(
            tool
            for tool, spec in captured["tools"].items()
            if spec["annotations"].get("readOnlyHint") is not True
        )

    assert writers["portfolio"] == [], "the portfolio server must be read-only (A1)"
    assert writers["marketplace"] == ["send_message", "submit_proposal"], (
        f"unexpected marketplace write surface: {writers['marketplace']}"
    )
    assert writers["workspace"] == [
        "create_email_draft",
        "create_interview_hold",
        "release_interview_hold",
    ], f"unexpected workspace write surface: {writers['workspace']}"
