"""The worker process (E1).

Two runners behind one set of tasks:

* **asyncio** (default) — an in-process scheduler. No broker, nothing to install, and it
  is what `make worker` runs. Correct for one worker, which is what a single freelancer's
  copilot needs.
* **arq** (`--runner arq`) — Redis-backed, for more than one worker or a separate
  schedule. Imported lazily, so the default path needs no Redis.

The schedule is not a cron expression. Each saved search carries its own interval and the
ledger records when it last ran, so a worker that was down for an hour does one catch-up
sweep rather than the twelve it missed. That matters more in live mode than in GigSim:
twelve sweeps in a burst against someone else's marketplace is exactly the traffic pattern
their automation rules are about.

    make worker                 # run forever
    make worker-once            # one pass, then exit. what CI runs
    python -m worker.scheduler --dry-run
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import signal
import time
from typing import Any, ClassVar

from worker.tasks import (
    Ledger,
    SavedSearch,
    default_searches,
    poll_outcomes,
    scout_sweep,
)

# How often the loop wakes to check whether anything is due. Not the sweep interval: a
# short tick with long intervals means a search that becomes due is picked up promptly
# without polling the marketplace.
TICK_SEC = float(os.getenv("WORKER_TICK_SEC", "15"))

# Outcome polling advances the simulation, so it runs on its own cadence rather than per
# search. In live mode this would instead read real proposal statuses.
OUTCOME_INTERVAL_SEC = float(os.getenv("WORKER_OUTCOME_INTERVAL_SEC", "600"))


def log(event: str, **fields: Any) -> None:
    """One JSON line per event, so the worker log is greppable and parseable.

    Deliberately not the `logging` module: this process has one job and its output is a
    stream of structured events. A trace id is attached where there is one, so a worker
    line can be correlated with the API and MCP audit rows for the same run.
    """
    from mcp_servers._common.trace import get_trace_id

    print(
        json.dumps(
            {
                "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "event": event,
                "trace_id": get_trace_id(),
                **fields,
            },
            default=str,
        ),
        flush=True,
    )


def due(search: SavedSearch, ledger: Ledger) -> bool:
    last = ledger.last_run(search.search_id)
    return last is None or (time.time() - last) >= search.interval_sec


async def run_once(
    searches: list[SavedSearch],
    ledger: Ledger,
    *,
    dry_run: bool = False,
    force: bool = False,
    poll: bool = True,
) -> dict[str, Any]:
    """One pass over every search that is due. Returns a summary.

    Used by `--once` and by CI, which wants a deterministic single pass rather than a
    daemon.
    """
    from mcp_servers._common.trace import set_trace_id, set_user_id

    results: list[dict[str, Any]] = []
    for search in searches:
        if not force and not due(search, ledger):
            continue
        # A fresh trace id per sweep, so everything one sweep caused shares an id.
        set_trace_id(None)
        set_user_id(search.user_id)
        log("sweep_start", search_id=search.search_id, persona=search.persona)
        result = await scout_sweep(search, ledger, dry_run=dry_run)
        log("sweep_done", **result)
        results.append(result)

    outcomes: dict[str, Any] = {}
    if poll and not dry_run and searches:
        set_trace_id(None)
        outcomes = await poll_outcomes(searches[0].user_id, searches[0].persona)
        log("outcomes_polled", **outcomes)

    return {"sweeps": results, "outcomes": outcomes}


async def run_forever(searches: list[SavedSearch], ledger: Ledger) -> None:
    """The daemon. Ticks, runs what is due, and shuts down cleanly on a signal."""
    stopping = asyncio.Event()

    def request_stop(*_: Any) -> None:
        # A sweep in flight finishes rather than being killed mid-write. The ledger is
        # durable either way, but a half-written thread is a worse thing to resume.
        log("shutdown_requested")
        stopping.set()

    with contextlib.suppress(NotImplementedError, AttributeError):
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, request_stop)

    log(
        "worker_started",
        searches=[s.search_id for s in searches],
        tick_sec=TICK_SEC,
        intervals={s.search_id: s.interval_sec for s in searches},
    )

    last_poll = 0.0
    while not stopping.is_set():
        try:
            poll_now = (time.time() - last_poll) >= OUTCOME_INTERVAL_SEC
            await run_once(searches, ledger, poll=poll_now)
            if poll_now:
                last_poll = time.time()
        except Exception as exc:
            # A failing sweep must not kill the worker; the next tick tries again.
            log("sweep_error", error=f"{type(exc).__name__}: {exc}")

        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(stopping.wait(), timeout=TICK_SEC)

    log("worker_stopped", **ledger.stats())


# ----------------------------------------------------------------------- arq


def arq_settings() -> Any:
    """Arq's WorkerSettings, for the Redis-backed runner.

    Built lazily so the default asyncio path needs neither arq nor Redis installed.
    """
    try:
        from arq.connections import RedisSettings
    except ImportError as exc:
        raise SystemExit(
            "the arq runner needs: pip install arq\n"
            "The default asyncio runner needs nothing and is what `make worker` uses."
        ) from exc

    ledger = Ledger()
    searches = default_searches()

    async def sweep_task(ctx: dict[str, Any], search_id: str) -> dict[str, Any]:
        search = next((s for s in searches if s.search_id == search_id), None)
        if search is None:
            return {"error": f"unknown search {search_id}"}
        return await scout_sweep(search, ledger)

    async def outcomes_task(ctx: dict[str, Any]) -> dict[str, Any]:
        return await poll_outcomes(searches[0].user_id, searches[0].persona)

    from arq import cron

    class WorkerSettings:
        # ClassVar: arq reads these off the class, they are not instance state.
        functions: ClassVar[list] = [sweep_task, outcomes_task]
        redis_settings = RedisSettings.from_dsn(
            os.getenv("REDIS_URL", "redis://localhost:6379/0")
        )
        # Arq's cron is minute-granular, which is why the asyncio runner keeps the
        # per-search intervals: they are the authoritative schedule.
        cron_jobs: ClassVar[list] = [
            cron(outcomes_task, minute={0, 10, 20, 30, 40, 50}),
        ]
        max_jobs = 4
        job_timeout = 300

    return WorkerSettings


def main() -> None:
    parser = argparse.ArgumentParser(description="GigBid worker (E1)")
    parser.add_argument("--runner", choices=["asyncio", "arq"], default="asyncio")
    parser.add_argument("--once", action="store_true", help="one pass, then exit")
    parser.add_argument("--force", action="store_true", help="ignore the schedule")
    parser.add_argument(
        "--dry-run", action="store_true", help="report what would be queued, queue nothing"
    )
    parser.add_argument("--stats", action="store_true", help="print the ledger and exit")
    args = parser.parse_args()

    ledger = Ledger()

    if args.stats:
        print(json.dumps(ledger.stats(), indent=2, default=str))
        return

    if args.runner == "arq":
        from arq import run_worker

        run_worker(arq_settings())  # type: ignore[arg-type]
        return

    searches = default_searches()
    if args.once or args.dry_run:
        result = asyncio.run(
            run_once(searches, ledger, dry_run=args.dry_run, force=args.force or args.dry_run)
        )
        print(json.dumps(result, indent=2, default=str))
        return

    asyncio.run(run_forever(searches, ledger))


if __name__ == "__main__":
    main()
