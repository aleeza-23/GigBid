"""The bid graph end to end against GigSim in-process (C1, C4, C5, C6).

Run with the model forced offline. That is deliberate: these tests check routing,
reducers, loop bounds, interrupts and exactly-once submission, none of which should
depend on what a model felt like returning. The rule-driven nodes (risk, coverage,
pricing, claim verification) still run for real, and those are the ones Part B measured.

What each group proves:

* **parallelism** - all three analysts report, and the fan-out is a real fork;
* **routing** - a scam never reaches the Writer, a blocked draft never reaches a human;
* **the interrupt** - the graph stops before spending anything, with a payload a human
  can decide from;
* **exactly once** - approve, crash mid-submit, resume: one proposal, one spend.
"""

from __future__ import annotations

import os
from typing import Any

import httpx
import pytest

# Must be set before agent.llm is imported anywhere, or the cached client keeps a key.
os.environ["GIGBID_FORCE_OFFLINE"] = "1"

from langgraph.types import Command

from agent.checkpointer import checkpointer
from agent.graphs.bid_graph import (
    PARALLEL_ANALYSTS,
    build_bid_graph,
    dispatch,
    initial_state,
    route_after_compliance,
)
from agent.handoffs import ComplianceVerdict, Critique
from agent.nodes.critic import route_after_critique
from agent.nodes.decision import route_after_decision
from gigsim.config import settings
from gigsim.rate_limit import bucket
from gigsim.server import app
from gigsim.store import store
from mcp_servers._common.idempotency import IdempotencyStore
from mcp_servers.marketplace import server as market
from mcp_servers.marketplace.adapters.gigsim import GigSimAdapter

PERSONA = "ai_ml_engineer"


@pytest.fixture(autouse=True)
def _wire(tmp_path, monkeypatch):
    """GigSim in-process, a fresh idempotency store, and the model offline."""
    store.reset()
    bucket.reset()
    monkeypatch.setattr(settings, "error_rate", 0.0)
    monkeypatch.setattr(settings, "rate_limit_per_min", 10_000)
    monkeypatch.setenv("GIGBID_FORCE_OFFLINE", "1")

    # Per-test store: a shared one would let an earlier test's key make a later
    # submission look like a replay, hiding a real double-spend.
    market.idempotency = IdempotencyStore(tmp_path / "idem.sqlite3")

    adapter = GigSimAdapter(
        base_url="http://sim", token=settings.api_token, transport=httpx.ASGITransport(app=app)
    )
    market._port = adapter

    import agent.llm as llm_module

    llm_module._default = None
    yield
    market._port = None
    llm_module._default = None


def _best_job() -> dict[str, Any]:
    """An open job the system should actually bid on.

    Skill overlap alone is not enough. The highest-overlap job in the seed data is a
    fixed-price post below the persona's $200 minimum project budget, which the graph
    correctly skips — so a test picking on overlap alone never reaches the Writer and
    looks like a graph bug rather than a working economic guard.
    """
    from mcp_servers.portfolio import corpus
    from mcp_servers.portfolio import tools as portfolio

    own = {row["skill"].lower() for row in portfolio.list_skills(PERSONA)["skills"]}
    profile = corpus.load_profile(PERSONA)
    rate_floor = float(profile.get("hourly_rate_usd") or 0)
    project_floor = float(profile.get("minimum_project_budget_usd") or 0)

    def viable(job: dict[str, Any]) -> bool:
        client = store.clients.get(job.get("client_id") or "", {})
        if client.get("is_scam_account"):
            return False
        if job.get("is_hourly"):
            return float(job.get("hourly_high") or 0) >= rate_floor
        return float(job.get("budget") or 0) >= project_floor

    def overlap(job: dict[str, Any]) -> int:
        return len(own & {str(s).lower() for s in (job.get("skills") or [])})

    candidates = [j for j in store.jobs.values() if viable(j) and overlap(j) >= 3]
    if not candidates:
        pytest.skip("no economically viable, well-matched job in the seed data")
    return max(candidates, key=overlap)


def _a_scam_job() -> dict[str, Any]:
    for job in store.jobs.values():
        client = store.clients.get(job.get("client_id") or "", {})
        if client.get("is_scam_account"):
            return job
    pytest.skip("no scam job in the seed data")


def _nodes(state: dict[str, Any]) -> list[str]:
    return [entry["node"] for entry in state.get("execution_trace") or []]


# ------------------------------------------------------------------ structure


def test_the_fan_out_dispatches_all_three_analysts() -> None:
    """R3: the three analyses run in parallel, not in sequence."""
    sends = dispatch({"status": "running"})
    assert sorted(send.node for send in sends) == sorted(PARALLEL_ANALYSTS)
    assert len(sends) == 3


def test_a_failed_thread_skips_the_fan_out() -> None:
    sends = dispatch({"status": "needs_attention"})
    assert [send.node for send in sends] == ["join"]


def test_routing_sends_only_a_bid_to_the_writer() -> None:
    from agent.handoffs import BidDecision

    for verdict in ("skip", "watch"):
        assert (
            route_after_decision({"decision": BidDecision(decision=verdict), "status": "running"})
            == "halt"
        )
    assert (
        route_after_decision({"decision": BidDecision(decision="bid"), "status": "running"})
        == "bid"
    )


def test_a_blocked_draft_never_reaches_a_human() -> None:
    blocked = ComplianceVerdict(status="blocked", violations=["invented a credential"])
    assert route_after_compliance({"compliance_verdict": blocked}) == "halt"
    assert route_after_compliance({"compliance_verdict": ComplianceVerdict(status="pass")}) == "approval"


def test_the_critic_loop_is_bounded_by_the_router() -> None:
    """C4: the Critic may ask forever; the router says no after two.

    The bound lives here rather than in the prompt, because a prompt asking a model to
    restrain itself is a request and a router that counts is a guarantee.
    """
    revise = Critique(verdict="revise", reasons=["generic opening"])
    assert route_after_critique({"critique": revise, "revision_count": 1}) == "rewrite"
    assert route_after_critique({"critique": revise, "revision_count": 2}) == "rewrite"
    # Third request refused: the draft goes forward carrying the unresolved critique.
    assert route_after_critique({"critique": revise, "revision_count": 3}) == "compliance"
    assert route_after_critique({"critique": Critique(verdict="approve"), "revision_count": 0}) == "compliance"


def test_both_graphs_compile_and_expose_their_shape() -> None:
    """C1 asks for both graphs rendered from code, not drawn by hand."""
    from agent.graphs.bid_graph import render_mermaid as bid_diagram
    from agent.graphs.conversation_graph import render_mermaid as conversation_diagram

    bid = bid_diagram()
    assert "fan_out" in bid and "approval" in bid and "submit" in bid
    # The loop edge back to the writer must be visible in the rendered graph.
    assert "critic" in bid and "writer" in bid
    assert "classify_intent" in conversation_diagram()


# -------------------------------------------------------------------- behaviour


@pytest.mark.asyncio
async def test_all_three_analysts_report_before_the_decision() -> None:
    job = _best_job()
    with checkpointer(backend="memory") as saver:
        graph = build_bid_graph(checkpointer=saver)
        config = {"configurable": {"thread_id": "parallel"}}
        await graph.ainvoke(initial_state(job, PERSONA, freelancer_id=PERSONA), config)
        snapshot = await graph.aget_state(config)

    completed = set(snapshot.values.get("analyses_complete") or [])
    assert completed == set(PARALLEL_ANALYSTS), f"only {completed} reported"

    visited = _nodes(snapshot.values)
    assert visited.index("join") > max(visited.index(name) for name in PARALLEL_ANALYSTS)
    # The reducer worked: three concurrent branches each left trace entries.
    assert len([n for n in visited if n in PARALLEL_ANALYSTS]) == 3


@pytest.mark.asyncio
async def test_a_scam_job_is_skipped_and_never_reaches_the_writer() -> None:
    """The money test: a scam must not consume Writer tokens or reach a human."""
    job = _a_scam_job()
    with checkpointer(backend="memory") as saver:
        graph = build_bid_graph(checkpointer=saver)
        config = {"configurable": {"thread_id": "scam"}}
        result = await graph.ainvoke(initial_state(job, PERSONA, freelancer_id=PERSONA), config)

    assert result["decision"].decision == "skip"
    assert result["decision"].deciding_factor in ("risk", "free_work")
    assert result["status"] == "skipped"
    visited = _nodes(result)
    assert "writer" not in visited
    assert "approval" not in visited
    assert result["decision"].reasons, "a skip must tell the user why"


@pytest.mark.asyncio
async def test_the_graph_interrupts_before_spending_anything() -> None:
    """R4 and C5: nothing leaves the system without a human."""
    job = _best_job()
    freelancer = next(iter(store.connects))
    before = store.connects_balance(freelancer)["balance"]

    with checkpointer(backend="memory") as saver:
        graph = build_bid_graph(checkpointer=saver)
        config = {"configurable": {"thread_id": "interrupt"}}
        await graph.ainvoke(initial_state(job, PERSONA, freelancer_id=freelancer), config)
        snapshot = await graph.aget_state(config)

    assert snapshot.interrupts, f"expected an interrupt; stopped at {_nodes(snapshot.values)}"
    assert store.connects_balance(freelancer)["balance"] == before, "nothing may be spent yet"
    assert not store.proposals, "no proposal may exist before approval"

    payload = snapshot.interrupts[0].value
    # C5 lists what the approval payload must carry.
    for key in ("draft", "claims_with_evidence", "risk", "fit", "price", "dry_run_preview"):
        assert key in payload, f"the approval payload is missing {key}"
    assert payload["price"]["rationale"], "the price must explain itself"
    assert "cannot be undone" in payload["warning"]


@pytest.mark.asyncio
async def test_approving_submits_exactly_once() -> None:
    job = _best_job()
    freelancer = next(iter(store.connects))
    cost = job["connects_cost"]
    before = store.connects_balance(freelancer)["balance"]

    with checkpointer(backend="memory") as saver:
        graph = build_bid_graph(checkpointer=saver)
        config = {"configurable": {"thread_id": "approve"}}
        await graph.ainvoke(initial_state(job, PERSONA, freelancer_id=freelancer), config)
        result = await graph.ainvoke(Command(resume={"action": "approve"}), config)

    assert result["status"] == "submitted"
    assert result["submission"]["proposal_id"]
    assert result["idempotency_key"], "the key must be recorded in state for a retry"
    spent = before - store.connects_balance(freelancer)["balance"]
    assert spent == cost, f"expected one spend of {cost}, got {spent}"
    assert len(store.proposals) == 1


@pytest.mark.asyncio
async def test_rejecting_sends_nothing() -> None:
    job = _best_job()
    freelancer = next(iter(store.connects))
    before = store.connects_balance(freelancer)["balance"]

    with checkpointer(backend="memory") as saver:
        graph = build_bid_graph(checkpointer=saver)
        config = {"configurable": {"thread_id": "reject"}}
        await graph.ainvoke(initial_state(job, PERSONA, freelancer_id=freelancer), config)
        result = await graph.ainvoke(
            Command(resume={"action": "reject", "reason": "rate too low"}), config
        )

    assert result["status"] == "rejected"
    assert store.connects_balance(freelancer)["balance"] == before
    assert not store.proposals


@pytest.mark.asyncio
async def test_an_edit_is_recorded_for_style_memory_and_sent_instead() -> None:
    """C5: edits are diffed, saved for style memory, and re-checked by Compliance."""
    job = _best_job()
    freelancer = next(iter(store.connects))
    edited = (
        "You need a transcription pipeline that keeps up with your volume. I built one "
        "that handled four thousand items a month and cut turnaround by a third."
    )

    with checkpointer(backend="memory") as saver:
        graph = build_bid_graph(checkpointer=saver)
        config = {"configurable": {"thread_id": "edit"}}
        await graph.ainvoke(
            initial_state(job, PERSONA, freelancer_id=freelancer, user_id="aleeza"), config
        )
        result = await graph.ainvoke(
            Command(resume={"action": "edit", "cover_letter": edited}), config
        )

    assert result["status"] == "submitted"
    assert result["proposal_draft"].cover_letter == edited
    assert result["user_edits"], "the diff must be recorded for style memory"
    assert store.proposals
    submitted = next(iter(store.proposals.values()))
    assert submitted["cover_letter"] == edited, "the human's text is what gets sent"


@pytest.mark.asyncio
async def test_an_edit_that_breaks_policy_is_not_sent() -> None:
    """A human can reintroduce a phone number as easily as a model can."""
    job = _best_job()
    freelancer = next(iter(store.connects))

    with checkpointer(backend="memory") as saver:
        graph = build_bid_graph(checkpointer=saver)
        config = {"configurable": {"thread_id": "bad-edit"}}
        await graph.ainvoke(initial_state(job, PERSONA, freelancer_id=freelancer), config)
        result = await graph.ainvoke(
            Command(
                resume={
                    "action": "edit",
                    "cover_letter": "Great fit for this work. Email me at me@example.com "
                    "and we can sort out payment directly.",
                }
            ),
            config,
        )

    assert result["status"] == "needs_attention"
    assert not store.proposals, "policy-breaking text must not reach the marketplace"
    assert any("policy" in str(e.get("error", "")).lower() for e in result.get("errors") or [])


@pytest.mark.asyncio
async def test_a_pending_approval_survives_rebuilding_the_graph() -> None:
    """C6, as far as an in-memory checkpointer can show it.

    The graph object is thrown away and rebuilt against the same checkpointer, which is
    what an API restart looks like to the thread. A true cross-process restart needs the
    SQLite or Postgres checkpointer; the test below skips when neither is installed.
    """
    job = _best_job()
    freelancer = next(iter(store.connects))

    with checkpointer(backend="memory") as saver:
        config = {"configurable": {"thread_id": "restart"}}
        first = build_bid_graph(checkpointer=saver)
        await first.ainvoke(initial_state(job, PERSONA, freelancer_id=freelancer), config)
        assert (await first.aget_state(config)).interrupts

        del first
        # A new graph object, the same persisted thread.
        second = build_bid_graph(checkpointer=saver)
        snapshot = await second.aget_state(config)
        assert snapshot.interrupts, "the pending approval was lost"

        result = await second.ainvoke(Command(resume={"action": "approve"}), config)

    assert result["status"] == "submitted"
    assert len(store.proposals) == 1


@pytest.mark.asyncio
async def test_a_crash_between_claim_and_write_blocks_the_retry() -> None:
    """The half of exactly-once a checkpointer alone cannot give you.

    The key is claimed before the marketplace call. Kill the process in between and the
    claim is left in flight, so the retry is refused rather than allowed to spend again.
    Refusing costs a human looking at a board; proceeding costs real Connects.
    """
    job = _best_job()
    freelancer = next(iter(store.connects))

    with checkpointer(backend="memory") as saver:
        graph = build_bid_graph(checkpointer=saver)
        config = {"configurable": {"thread_id": "crash"}}
        await graph.ainvoke(initial_state(job, PERSONA, freelancer_id=freelancer), config)
        snapshot = await graph.aget_state(config)
        assert snapshot.interrupts

        # Simulate a process that claimed the key and died before the write landed.
        key = f"proposal-{job['job_id']}-crashed"
        market.idempotency.claim(key, tool="submit_proposal", user_id="aleeza")
        await graph.aupdate_state(config, {"idempotency_key": key})

        result = await graph.ainvoke(Command(resume={"action": "approve"}), config)

    assert not store.proposals, "an in-flight key must not be allowed to spend again"
    assert result["status"] == "needs_attention"


@pytest.mark.asyncio
async def test_a_thread_over_budget_stops_visibly() -> None:
    """C4: exceeding a budget moves the thread to needs_attention, not into a loop."""
    job = _best_job()
    with checkpointer(backend="memory") as saver:
        graph = build_bid_graph(checkpointer=saver)
        config = {"configurable": {"thread_id": "budget"}}
        state = initial_state(job, PERSONA, freelancer_id=PERSONA)
        state["tokens_used"] = 999_999
        result = await graph.ainvoke(state, config)

    assert result["status"] == "needs_attention"
    assert result["budget_exceeded"]
    assert "writer" not in _nodes(result)


@pytest.mark.asyncio
async def test_the_trace_records_every_node_with_its_prompt_version() -> None:
    """E3 and viva question 4: a trace has to be enough to find the node that erred."""
    job = _best_job()
    with checkpointer(backend="memory") as saver:
        graph = build_bid_graph(checkpointer=saver)
        config = {"configurable": {"thread_id": "trace"}}
        await graph.ainvoke(initial_state(job, PERSONA, freelancer_id=PERSONA), config)
        snapshot = await graph.aget_state(config)

    entries = snapshot.values.get("execution_trace") or []
    assert entries
    for entry in entries:
        assert entry["node"] and entry["kind"] and entry["ts"]

    # Model-backed nodes record which prompt version produced their output.
    model_entries = [e for e in entries if e["kind"] == "model" and e.get("prompt")]
    assert model_entries, "no node recorded its prompt version"
    assert all("@v" in e["prompt"] for e in model_entries)


@pytest.mark.asyncio
async def test_a_dead_marketplace_degrades_rather_than_crashing() -> None:
    """A6: the UI must be able to say what is down."""

    def always_down(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"detail": "simulated outage"})

    job = _best_job()
    market._port = GigSimAdapter(
        base_url="http://sim",
        token=settings.api_token,
        transport=httpx.MockTransport(always_down),
        max_attempts=1,
    )

    with checkpointer(backend="memory") as saver:
        graph = build_bid_graph(checkpointer=saver)
        config = {"configurable": {"thread_id": "degraded"}}
        # The job is already in state, so the Scout does not need the marketplace; the
        # thread should still complete its analysis and report what it could not reach.
        result = await graph.ainvoke(initial_state(job, PERSONA, freelancer_id=PERSONA), config)

    assert result.get("degraded"), "an unreachable marketplace must be reported"
    assert result["status"] != "submitted"
