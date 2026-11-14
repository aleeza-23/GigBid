"""The Workspace server through an in-memory MCP client (A7).

The load-bearing test here is the last one: this server has no tool that sends anything.
A3 allows a local mock, and the mock's job is to be honest about that boundary rather than
to pretend to be Gmail.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

import pytest
from mcp.shared.memory import create_connected_server_and_client_session

from mcp_servers.workspace import server as ws
from mcp_servers.workspace.calendar_mock import LocalWorkspace


@pytest.fixture(autouse=True)
def _fresh_backend(tmp_path):
    ws.set_backend_for_tests(LocalWorkspace(tmp_path / "workspace.sqlite3"))
    yield
    ws.set_backend_for_tests(None)


async def _session():
    return create_connected_server_and_client_session(ws.mcp._mcp_server)


def _payload(result: Any) -> dict[str, Any]:
    if getattr(result, "structuredContent", None):
        return result.structuredContent
    return json.loads(result.content[0].text)


@pytest.mark.asyncio
async def test_the_three_required_capabilities_exist() -> None:
    async with await _session() as session:
        names = {t.name for t in (await session.list_tools()).tools}

    assert "free_interview_slots" in names, "A3: free calendar slots"
    assert "create_interview_hold" in names, "A3: a tentative interview hold"
    assert "create_email_draft" in names, "A3: an email draft"


@pytest.mark.asyncio
async def test_slots_are_weekdays_inside_working_hours() -> None:
    async with await _session() as session:
        data = _payload(await session.call_tool("free_interview_slots", {"days_ahead": 14, "limit": 5}))

    assert data["slots"], "a fresh calendar should have room"
    for slot in data["slots"]:
        start = datetime.fromisoformat(slot["start"])
        assert start.weekday() < 5, "no weekend interviews"
        assert 9 <= start.hour < 18, "inside working hours"
        assert start > datetime.now(start.tzinfo), "never offer a slot in the past"


@pytest.mark.asyncio
async def test_a_hold_removes_that_slot_from_later_offers() -> None:
    async with await _session() as session:
        first = _payload(await session.call_tool("free_interview_slots", {"limit": 3}))
        target = first["slots"][0]["start"]

        hold = _payload(
            await session.call_tool("create_interview_hold", {"start": target, "title": "Client call"})
        )
        assert hold["status"] == "tentative"

        after = _payload(await session.call_tool("free_interview_slots", {"limit": 3}))
        assert target not in [s["start"] for s in after["slots"]]

        listed = _payload(await session.call_tool("list_interview_holds", {}))
        assert listed["total"] == 1


@pytest.mark.asyncio
async def test_a_clashing_hold_is_refused() -> None:
    """Offering a slot twice is the failure this prevents."""
    async with await _session() as session:
        slots = _payload(await session.call_tool("free_interview_slots", {"limit": 2}))
        target = slots["slots"][0]["start"]

        await session.call_tool("create_interview_hold", {"start": target})
        clash = await session.call_tool("create_interview_hold", {"start": target})

    assert clash.isError
    assert "clash" in clash.content[0].text.lower()


@pytest.mark.asyncio
async def test_releasing_a_hold_frees_the_slot_and_is_idempotent() -> None:
    async with await _session() as session:
        slots = _payload(await session.call_tool("free_interview_slots", {"limit": 2}))
        hold = _payload(
            await session.call_tool("create_interview_hold", {"start": slots["slots"][0]["start"]})
        )

        first = _payload(await session.call_tool("release_interview_hold", {"hold_id": hold["hold_id"]}))
        second = _payload(await session.call_tool("release_interview_hold", {"hold_id": hold["hold_id"]}))

    assert first["released"] is True
    # Releasing twice is a success reporting "nothing to do", not an error.
    assert second["released"] is False


@pytest.mark.asyncio
async def test_a_bad_start_time_is_a_clear_error() -> None:
    async with await _session() as session:
        result = await session.call_tool("create_interview_hold", {"start": "next tuesday-ish"})
    assert result.isError
    assert "ISO 8601" in result.content[0].text


@pytest.mark.asyncio
async def test_email_drafts_are_created_unsent() -> None:
    async with await _session() as session:
        data = _payload(
            await session.call_tool(
                "create_email_draft",
                {
                    "to": "client@example.com",
                    "subject": "Following up on the pipeline project",
                    "body": "Thanks for the reply. Tuesday at 2pm works for me.",
                    "thread_ref": "thr-001",
                },
            )
        )

    assert data["status"] == "draft"
    assert data["sent"] is False
    assert data["thread_ref"] == "thr-001"


@pytest.mark.asyncio
async def test_an_invalid_address_is_refused() -> None:
    async with await _session() as session:
        result = await session.call_tool(
            "create_email_draft", {"to": "not-an-address", "subject": "x", "body": "y"}
        )
    assert result.isError


@pytest.mark.asyncio
async def test_this_server_has_no_way_to_send_anything() -> None:
    """R4 as an absence rather than a check.

    A capability that does not exist cannot be reached by a confused model or a prompt
    injection. If someone later adds a send tool, this test fails and makes them justify
    it.
    """
    async with await _session() as session:
        names = {t.name for t in (await session.list_tools()).tools}

    forbidden = {n for n in names if "send" in n.lower() or n.startswith("post_")}
    assert not forbidden, f"the workspace server must not be able to send: found {forbidden}"

    health = None
    async with await _session() as session:
        health = _payload(await session.call_tool("workspace_health", {}))
    assert health["sends_email"] is False


def test_holds_survive_a_restart(tmp_path) -> None:
    """C6's durability argument, one layer down.

    A tentative hold kept in memory would vanish on the worker restart the assignment
    asks us to demonstrate, and the agent would then offer a slot it had already promised.
    """
    path = tmp_path / "persist.sqlite3"
    first = LocalWorkspace(path)
    slot = first.free_slots(7, 30, 1)[0]
    created = first.create_hold(slot.start, 30, "Interview", "")
    first.close()

    reopened = LocalWorkspace(path)
    assert [h["hold_id"] for h in reopened.list_holds()] == [created["hold_id"]]
    reopened.close()


def test_slots_are_spread_across_days_not_stacked_on_one_morning() -> None:
    """A slate of three options on one Tuesday morning is a worse offer than three days."""
    workspace = LocalWorkspace(":memory:")
    slots = workspace.free_slots(days_ahead=10, duration_minutes=30, limit=4)
    days = {s.start.date() for s in slots}
    assert len(days) == len(slots), "one slot per day"
