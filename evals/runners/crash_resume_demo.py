"""C6: kill the process while a proposal waits for approval, restart, approve once.

This is the demonstration C6 asks for and viva question 2 tests, run across **real process
boundaries** rather than simulated inside one. Nothing here is faked: each phase is a
separate Python process, and the only things that survive between them are the three
durable stores.

    make crash-resume

What it does:

1. **phase 1** (one process) runs the bid graph to the approval interrupt against a
   running GigSim, writing the thread to a SQLite checkpoint. Then the process exits
   without approving. That is the crash.
2. **phase 2** (a new process) opens the same checkpoint, finds the pending approval still
   there, approves it, and submits.
3. **phase 3** (a third process) resumes the *same already-completed* thread again and
   re-submits with the same idempotency key, which is what a retry or a double-click
   looks like.

Then it asks GigSim how many proposals exist. The answer must be one.

Three stores have to be durable for this to work, and losing any one breaks it:

| Store | Holds | Without it |
|---|---|---|
| checkpoint (SQLite/Postgres) | the paused thread | the pending approval is lost on restart |
| idempotency (SQLite) | which writes completed | the retry in phase 3 spends again |
| GigSim | the proposal itself | nothing to count |

GigSim runs as a real HTTP server here for the same reason: an in-process simulator would
reset with each phase, and then "exactly once" would be a statement about one process's
memory rather than about the system.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "evals" / "results"
STATE_DIR = ROOT / "data" / "audit" / "crash_demo"

GIGSIM_URL = os.getenv("GIGSIM_BASE_URL", "http://127.0.0.1:8081")
THREAD_ID = "crash-resume-demo"
PERSONA = "ai_ml_engineer"


def _env_for_phase() -> dict[str, str]:
    """The durable paths every phase must agree on."""
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    return {
        **os.environ,
        "CHECKPOINTER": "sqlite",
        "CHECKPOINT_DB": str(STATE_DIR / "checkpoints.sqlite3"),
        "MCP_IDEMPOTENCY_DB": str(STATE_DIR / "idempotency.sqlite3"),
        "GIGSIM_BASE_URL": GIGSIM_URL,
        # The graph's routing and interrupts are what is under test, not a model's output.
        "GIGBID_FORCE_OFFLINE": "1",
        "PYTHONWARNINGS": "ignore",
    }


def _client() -> httpx.Client:
    token = os.getenv("GIGSIM_API_TOKEN", "gigsim-dev-token")
    return httpx.Client(
        base_url=f"{GIGSIM_URL}/api/v1",
        headers={"Authorization": f"Bearer {token}"},
        timeout=20,
    )


def count_proposals(job_id: str) -> list[dict[str, Any]]:
    with _client() as client:
        response = client.get("/proposals")
        response.raise_for_status()
        rows = response.json().get("proposals") or []
    return [p for p in rows if p.get("job_id") == job_id]


def connects(freelancer_id: str) -> int:
    with _client() as client:
        response = client.get(f"/freelancers/{freelancer_id}/connects")
        response.raise_for_status()
        return int(response.json().get("balance") or 0)


def pick_job() -> dict[str, Any]:
    """A job the graph will actually bid on: good overlap and economically viable."""
    from mcp_servers.portfolio import corpus
    from mcp_servers.portfolio import tools as portfolio

    own = {row["skill"].lower() for row in portfolio.list_skills(PERSONA)["skills"]}
    profile = corpus.load_profile(PERSONA)
    rate_floor = float(profile.get("hourly_rate_usd") or 0)
    project_floor = float(profile.get("minimum_project_budget_usd") or 0)

    with _client() as client:
        response = client.get("/jobs", params={"limit": 100})
        response.raise_for_status()
        jobs = response.json().get("jobs") or []

    def viable(job: dict[str, Any]) -> bool:
        if job.get("is_hourly"):
            return float(job.get("hourly_high") or 0) >= rate_floor
        return float(job.get("budget") or 0) >= project_floor

    def overlap(job: dict[str, Any]) -> int:
        return len(own & {str(s).lower() for s in (job.get("skills") or [])})

    candidates = [j for j in jobs if viable(j) and overlap(j) >= 3]
    if not candidates:
        raise SystemExit("no economically viable, well-matched job available in GigSim")
    return max(candidates, key=overlap)


# --------------------------------------------------------------------- phases


async def phase_start(job_id: str) -> dict[str, Any]:
    """Run to the approval interrupt, then exit without approving."""
    from agent.checkpointer import acheckpointer, describe
    from agent.graphs.bid_graph import build_bid_graph, initial_state

    persistence = describe()
    if not persistence["survives_restart"]:
        raise SystemExit(
            f"checkpointer is {persistence['checkpointer']!r}, which does not survive a "
            "restart. This demo proves nothing without a durable one."
        )

    with _client() as client:
        job_response = client.get(f"/jobs/{job_id}")
        if job_response.status_code == 401:
            raise SystemExit(
                "GigSim refused the token (401). The server does not read .env, so it uses "
                "its built-in default; a placeholder GIGSIM_API_TOKEN in .env makes every "
                "client call fail while the server looks healthy. Set "
                "GIGSIM_API_TOKEN=gigsim-dev-token in .env, or export the same value "
                "before starting the server."
            )
        job_response.raise_for_status()
        job = job_response.json()
        history_response = client.get(f"/clients/{job['client_id']}/history")
        history_response.raise_for_status()
        history = history_response.json()

    async with acheckpointer() as saver:
        graph = build_bid_graph(checkpointer=saver)
        config = {"configurable": {"thread_id": THREAD_ID}}
        await graph.ainvoke(
            initial_state(job, PERSONA, freelancer_id=PERSONA, user_id="aleeza", client=history),
            config,
        )
        snapshot = await graph.aget_state(config)
        pending = bool(snapshot.interrupts)

    return {
        "phase": "start",
        "job_id": job_id,
        "awaiting_approval": pending,
        "checkpointer": persistence["checkpointer"],
        "nodes": [e["node"] for e in (snapshot.values.get("execution_trace") or [])],
    }


async def phase_approve() -> dict[str, Any]:
    """A fresh process: find the pending approval and approve it."""
    from langgraph.types import Command

    from agent.checkpointer import acheckpointer
    from agent.graphs.bid_graph import build_bid_graph

    async with acheckpointer() as saver:
        graph = build_bid_graph(checkpointer=saver)
        config = {"configurable": {"thread_id": THREAD_ID}}

        snapshot = await graph.aget_state(config)
        if not snapshot.interrupts:
            raise SystemExit(
                "no pending approval found after restart. The checkpoint did not survive, "
                "which is the exact failure C6 is about."
            )

        result = await graph.ainvoke(Command(resume={"action": "approve"}), config)

    submission = result.get("submission") or {}
    return {
        "phase": "approve",
        "found_pending_approval": True,
        "status": result.get("status"),
        "proposal_id": submission.get("proposal_id"),
        "connects_spent": submission.get("connects_spent"),
        "idempotency_key": result.get("idempotency_key"),
        "idempotent_replay": submission.get("idempotent_replay"),
    }


async def phase_retry() -> dict[str, Any]:
    """A third process re-submits the same approval. Nothing may be spent again."""
    from agent.checkpointer import acheckpointer
    from agent.graphs.bid_graph import build_bid_graph
    from agent.nodes.approval import submit

    async with acheckpointer() as saver:
        graph = build_bid_graph(checkpointer=saver)
        config = {"configurable": {"thread_id": THREAD_ID}}
        snapshot = await graph.aget_state(config)
        state = dict(snapshot.values)

    # Call the submit node directly with the stored state and its original key, which is
    # what a worker retrying a completed job does.

    from mcp_servers.marketplace import server as market
    from mcp_servers.marketplace.adapters.gigsim import GigSimAdapter

    market._port = GigSimAdapter(base_url=GIGSIM_URL, token=os.getenv("GIGSIM_API_TOKEN", "gigsim-dev-token"))
    state["status"] = "approved"
    result = await submit(state)
    await market._port.close()

    submission = result.get("submission") or {}
    return {
        "phase": "retry",
        "reused_key": state.get("idempotency_key"),
        "status": result.get("status"),
        "proposal_id": submission.get("proposal_id"),
        "idempotent_replay": submission.get("idempotent_replay"),
    }


# ------------------------------------------------------------------ orchestrator


def run_phase(phase: str, job_id: str = "") -> dict[str, Any]:
    """Run one phase in its own process and return its JSON result."""
    command = [sys.executable, "-m", "evals.runners.crash_resume_demo", "--phase", phase]
    if job_id:
        command += ["--job", job_id]
    # check=False on purpose: a phase that fails should be reported with its stderr
    # below, not raise a CalledProcessError that hides the output explaining why.
    completed = subprocess.run(
        command,
        cwd=str(ROOT),
        env=_env_for_phase(),
        capture_output=True,
        text=True,
        check=False,
    )
    for line in completed.stdout.splitlines():
        if line.startswith("RESULT:"):
            return json.loads(line[len("RESULT:") :])
    raise SystemExit(
        f"phase {phase} produced no result (exit {completed.returncode}).\n"
        f"stdout:\n{completed.stdout[-1500:]}\nstderr:\n{completed.stderr[-1500:]}"
    )


def orchestrate() -> None:
    # A fresh checkpoint and idempotency store, or a previous run's completed thread would
    # make phase 2 find nothing pending.
    if STATE_DIR.exists():
        for path in STATE_DIR.glob("*.sqlite3*"):
            path.unlink(missing_ok=True)
    STATE_DIR.mkdir(parents=True, exist_ok=True)

    try:
        with _client() as client:
            client.get("/jobs", params={"limit": 1}).raise_for_status()
    except Exception as exc:
        raise SystemExit(
            f"GigSim is not reachable at {GIGSIM_URL} ({type(exc).__name__}).\n"
            "This demo needs it running as a real server, because an in-process simulator "
            "would reset between phases and 'exactly once' would mean nothing.\n"
            "Start it with:  make gigsim"
        ) from exc

    job = pick_job()
    job_id = job["job_id"]
    before = connects(PERSONA)

    print("C6: crash and resume, across three separate processes")
    print(f"job        {job_id}  {job['title'][:52]}")
    print(f"connects   {before} before")
    print(f"state      {STATE_DIR}")
    print()

    print("phase 1  run to the approval interrupt, then exit without approving")
    start = run_phase("start", job_id)
    print(f"         awaiting_approval={start['awaiting_approval']} "
          f"checkpointer={start['checkpointer']}")
    print(f"         nodes: {' -> '.join(start['nodes'][-5:])}")
    mid = count_proposals(job_id)
    print(f"         proposals in GigSim: {len(mid)}  (must be 0)")
    print("         *** process exited: this is the crash ***")
    print()

    print("phase 2  NEW process, same checkpoint: find the pending approval and approve")
    approve = run_phase("approve")
    print(f"         found_pending_approval={approve['found_pending_approval']} "
          f"status={approve['status']}")
    print(f"         proposal={approve['proposal_id']} "
          f"connects_spent={approve['connects_spent']}")
    after_approve = count_proposals(job_id)
    print(f"         proposals in GigSim: {len(after_approve)}  (must be 1)")
    print()

    print("phase 3  THIRD process re-submits the same approval (a retry or double-click)")
    retry = run_phase("retry")
    print(f"         reused key, idempotent_replay={retry['idempotent_replay']} "
          f"proposal={retry['proposal_id']}")
    final = count_proposals(job_id)
    spent = before - connects(PERSONA)
    print(f"         proposals in GigSim: {len(final)}  (must still be 1)")
    print()

    checks = {
        "paused before spending anything": len(mid) == 0,
        "pending approval survived the restart": approve["found_pending_approval"],
        "submitted after approval": len(after_approve) == 1,
        "retry did not create a second proposal": len(final) == 1,
        "retry reported itself as a replay": bool(retry["idempotent_replay"]),
        "connects spent exactly once": spent == int(job.get("connects_cost") or 0),
        "the same key was reused, not regenerated": retry["reused_key"]
        == approve["idempotency_key"],
    }

    print("=" * 72)
    for label, passed in checks.items():
        print(f"  {'PASS' if passed else 'FAIL'}  {label}")
    print("=" * 72)
    verdict = all(checks.values())
    print(f"\n{'exactly once, across three processes' if verdict else 'DEMONSTRATION FAILED'}")
    print(f"connects spent: {spent} (job costs {job.get('connects_cost')})")

    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / "c6_crash_resume.json"
    out.write_text(
        json.dumps(
            {
                "demonstrates": "C6 exactly-once across a real process restart",
                "job_id": job_id,
                "connects_cost": job.get("connects_cost"),
                "connects_spent": spent,
                "proposals_created": len(final),
                "phases": {"start": start, "approve": approve, "retry": retry},
                "checks": checks,
                "passed": verdict,
                "ran_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nwritten: {out}")
    if not verdict:
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="C6 crash-and-resume demonstration")
    parser.add_argument("--phase", choices=["start", "approve", "retry"], default=None)
    parser.add_argument("--job", default="")
    args = parser.parse_args()

    if args.phase is None:
        orchestrate()
        return

    if args.phase == "start":
        result = asyncio.run(phase_start(args.job))
    elif args.phase == "approve":
        result = asyncio.run(phase_approve())
    else:
        result = asyncio.run(phase_retry())

    # The orchestrator reads this line out of the child process's stdout.
    print("RESULT:" + json.dumps(result))


if __name__ == "__main__":
    main()
