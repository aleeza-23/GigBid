"""The bid graph: one thread per job (C1).

Hand-written StateGraph, as R3 requires. The shape:

    START -> scout -> route_member -> fan_out ==> [analyst | risk_assessor | fit_matcher]
                                              |
                                        (all three join)
                                              v
                                          pricing
                                              v
                                          decision ----skip/watch/halt----> END
                                              | bid
                                              v
                                           writer <--------------+
                                              v                  | at most 2 revisions
                                          verifier                |
                                              v                  |
                                           critic ---revise------+
                                              | approve
                                              v
                                         compliance ---blocked---> END (needs attention)
                                              | pass/fixed
                                              v
                                      approval  <-- interrupt() stops here
                                              | approved
                                              v
                                           submit -> END

Three things about this that are easy to get wrong and worth stating:

**The fan-out is real.** `fan_out` returns a list of `Send` objects, so LangGraph runs the
three analysts concurrently rather than in sequence. That is why every state field they
touch declares a reducer: three nodes returning `execution_trace` into a plain field is a
lost update.

**Pricing runs after the join, not in parallel.** It needs the client history the Risk
Assessor fetched and the domain the Analyst inferred. Putting it in the fan-out would make
it race them.

**The critique loop is bounded by the router, not the prompt.** `route_after_critique`
counts revisions and stops at two. A prompt asking the Critic to be reasonable is a
request; a router that counts is a guarantee.
"""

from __future__ import annotations

import time
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from agent.nodes.agency import route_member
from agent.nodes.analyst import analyst
from agent.nodes.approval import approval, submit
from agent.nodes.compliance import compliance
from agent.nodes.critic import critic, route_after_critique
from agent.nodes.decision import decision, route_after_decision
from agent.nodes.fit_matcher import fit_matcher
from agent.nodes.pricing import pricing
from agent.nodes.risk_assessor import risk_assessor
from agent.nodes.scout import scout
from agent.nodes.verifier import verifier
from agent.nodes.writer import writer
from agent.state import GigBidState, trace
from mcp_servers._common.trace import new_trace_id

# The three analysts that run concurrently (R3).
PARALLEL_ANALYSTS = ("analyst", "risk_assessor", "fit_matcher")


async def fan_out(state: GigBidState, config: RunnableConfig | None = None) -> dict[str, Any]:
    """Start the parallel analysis branches.

    A node rather than three edges from `scout`, because the `Send` list is what makes the
    branching explicit in the trace and lets a future autonomy setting drop a branch
    without rewiring the graph.
    """
    return {
        "execution_trace": trace(
            "fan_out", "decision", result_summary={"branches": list(PARALLEL_ANALYSTS)}
        )
    }


def dispatch(state: GigBidState) -> list[Send]:
    """The conditional edge that actually forks the graph."""
    if state.get("status") in ("needs_attention", "dead_letter"):
        return [Send("join", state)]
    return [Send(name, state) for name in PARALLEL_ANALYSTS]


async def join(state: GigBidState, config: RunnableConfig | None = None) -> dict[str, Any]:
    """Barrier after the parallel branches.

    Reports which branches arrived rather than assuming all three did. A branch that
    failed leaves its report missing, and the decision node is written to treat a missing
    fit report as no coverage rather than as zero coverage.
    """
    done = sorted(set(state.get("analyses_complete") or []))
    missing = [name for name in PARALLEL_ANALYSTS if name not in done]
    return {
        "degraded": {"analysis_incomplete": missing} if missing else {},
        "execution_trace": trace(
            "join", "decision", result_summary={"completed": done, "missing": missing}
        ),
    }


async def halt(state: GigBidState, config: RunnableConfig | None = None) -> dict[str, Any]:
    """Terminal node for skip, watch and needs-attention.

    A node rather than an edge to END so the thread records *why* it stopped and stays
    legible on the pipeline board. A thread that ends with no explanation is a thread
    nobody can act on.
    """
    verdict = state.get("decision")
    status = state.get("status") or "skipped"
    if status == "running":
        status = "skipped" if getattr(verdict, "decision", "") == "skip" else "needs_attention"
    return {
        "status": status,
        "execution_trace": trace(
            "halt",
            "decision",
            result_summary={
                "status": status,
                "decision": getattr(verdict, "decision", None),
                "reasons": getattr(verdict, "reasons", [])[:3],
                "budget_exceeded": state.get("budget_exceeded"),
            },
        ),
    }


def route_after_compliance(state: GigBidState) -> str:
    """A blocked draft never reaches a human for approval."""
    verdict = state.get("compliance_verdict")
    if getattr(verdict, "status", "pass") == "blocked":
        return "halt"
    if state.get("status") in ("needs_attention", "dead_letter"):
        return "halt"
    return "approval"


def route_after_approval(state: GigBidState) -> str:
    """Only an approval reaches the submit node."""
    return "submit" if state.get("status") == "approved" else "halt"


def build_bid_graph(checkpointer: Any = None, store: Any = None) -> Any:
    """Compile the bid graph.

    `checkpointer` is what makes the interrupt durable: without one, `interrupt()` raises
    because there is nowhere to persist the pending thread. Tests pass an InMemorySaver;
    the deployed stack passes Postgres.
    """
    graph = StateGraph(GigBidState)

    graph.add_node("scout", scout)
    # X1: picks which team member bids. A pass-through unless agency mode is enabled.
    graph.add_node("route_member", route_member)
    graph.add_node("fan_out", fan_out)
    graph.add_node("analyst", analyst)
    graph.add_node("risk_assessor", risk_assessor)
    graph.add_node("fit_matcher", fit_matcher)
    graph.add_node("join", join)
    graph.add_node("pricing", pricing)
    graph.add_node("decision", decision)
    graph.add_node("writer", writer)
    graph.add_node("verifier", verifier)
    graph.add_node("critic", critic)
    graph.add_node("compliance", compliance)
    graph.add_node("approval", approval)
    graph.add_node("submit", submit)
    graph.add_node("halt", halt)

    graph.add_edge(START, "scout")
    # Routing runs before the fan-out, not inside it: every downstream node reads
    # `persona`, so choosing the member in a parallel branch would race the branches that
    # retrieve from that member's portfolio.
    graph.add_edge("scout", "route_member")
    graph.add_edge("route_member", "fan_out")

    # The fork. Each analyst returns to `join`, which waits for all of them.
    graph.add_conditional_edges("fan_out", dispatch, [*PARALLEL_ANALYSTS, "join"])
    for name in PARALLEL_ANALYSTS:
        graph.add_edge(name, "join")

    graph.add_edge("join", "pricing")
    graph.add_edge("pricing", "decision")

    graph.add_conditional_edges(
        "decision", route_after_decision, {"bid": "writer", "halt": "halt"}
    )

    graph.add_edge("writer", "verifier")
    graph.add_edge("verifier", "critic")
    # The bounded loop (C4).
    graph.add_conditional_edges(
        "critic", route_after_critique, {"rewrite": "writer", "compliance": "compliance"}
    )

    graph.add_conditional_edges(
        "compliance", route_after_compliance, {"approval": "approval", "halt": "halt"}
    )
    graph.add_conditional_edges(
        "approval", route_after_approval, {"submit": "submit", "halt": "halt"}
    )

    graph.add_edge("submit", END)
    graph.add_edge("halt", END)

    return graph.compile(checkpointer=checkpointer, store=store)


def initial_state(
    job: dict[str, Any],
    persona: str,
    *,
    freelancer_id: str | None = None,
    user_id: str = "default",
    client: dict[str, Any] | None = None,
) -> GigBidState:
    """Seed state for one job. The worker calls this."""
    return {
        "job_id": str(job.get("job_id") or ""),
        "job": dict(job),
        "client": dict(client or {}),
        "persona": persona,
        "freelancer_id": freelancer_id or persona,
        "user_id": user_id,
        "trace_id": new_trace_id(),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": "running",
        "steps_used": 0,
        "tokens_used": 0,
        "revision_count": 0,
    }


def render_mermaid() -> str:
    """The graph as Mermaid, for docs/ARCHITECTURE.md.

    Rendered from the compiled graph rather than drawn by hand, so the diagram in the
    report cannot drift from the code it documents.
    """
    compiled = build_bid_graph()
    try:
        return compiled.get_graph().draw_mermaid()
    except Exception as exc:  # pragma: no cover
        return f"(could not render: {exc})"
