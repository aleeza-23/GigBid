"""Background work (E1): scouting saved searches, running bid threads, polling outcomes.

None of this belongs in a web request. A bid thread makes several model calls and can take
a minute; a scouting sweep hits the marketplace dozens of times. Both would blow any
proxy's timeout and neither needs a browser attached.

**Idempotency is the whole design.** E1 requires that a worker restart never
double-analyses or double-submits, and there are three separate guards because they fail
differently:

1. **Thread ids are deterministic** (`bid:<user>:<job>`). Re-running a job resumes its
   existing thread rather than creating a second one, so a restart mid-sweep costs a
   replay, not a duplicate.
2. **A seen-jobs ledger** stops the Scout re-queueing a job it already queued, including
   across restarts, which is what makes near-duplicate posts cheap rather than expensive.
3. **The idempotency key** stops the marketplace write landing twice, which is the only
   one of the three that involves money.

The worker never approves anything. It runs threads *up to* the interrupt and stops. A
scheduled job that could approve its own proposals would make the human gate decorative.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

LEDGER_PATH = Path(
    __file__
).resolve().parents[1] / "data" / "audit" / "worker_ledger.sqlite3"


class Ledger:
    """What the worker has already done, durable across restarts.

    SQLite rather than a set in memory: the point is surviving the restart. A worker that
    forgets its ledger re-queues every job in the feed on every boot, and the user wakes
    up to two hundred duplicate threads.
    """

    def __init__(self, path: Path | str | None = None) -> None:
        self.path = Path(path or LEDGER_PATH)
        if str(self.path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS seen_jobs (
                user_id TEXT NOT NULL,
                job_id  TEXT NOT NULL,
                queued_at REAL NOT NULL,
                outcome TEXT,
                content_key TEXT,
                PRIMARY KEY (user_id, job_id)
            );
            CREATE INDEX IF NOT EXISTS seen_jobs_content
                ON seen_jobs (user_id, content_key);
            CREATE TABLE IF NOT EXISTS sweeps (
                search_id TEXT PRIMARY KEY,
                last_run  REAL NOT NULL,
                found     INTEGER DEFAULT 0,
                queued    INTEGER DEFAULT 0
            );
            """
        )
        self._conn.commit()

    def seen(self, user_id: str, job_id: str) -> bool:
        with self._lock:
            row = self._conn.execute(
                "SELECT 1 FROM seen_jobs WHERE user_id=? AND job_id=?", (user_id, job_id)
            ).fetchone()
        return row is not None

    def seen_content(self, user_id: str, content_key: str) -> str | None:
        """The job id we already queued with this content, if any.

        The Scout is required to de-duplicate, and an exact-id check does not: this corpus
        contains reposts of the same job under a new id, and a sweep queued both
        `upw-…ad` and `upw-…ad-repost`, spending two sets of model calls to produce two
        identical proposals for the same client.
        """
        if not content_key:
            return None
        with self._lock:
            row = self._conn.execute(
                "SELECT job_id FROM seen_jobs WHERE user_id=? AND content_key=? LIMIT 1",
                (user_id, content_key),
            ).fetchone()
        return row[0] if row else None

    def mark_seen(
        self,
        user_id: str,
        job_id: str,
        outcome: str | None = None,
        content_key: str | None = None,
    ) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO seen_jobs (user_id, job_id, queued_at, outcome, content_key) "
                "VALUES (?,?,?,?,?) ON CONFLICT(user_id, job_id) "
                "DO UPDATE SET outcome=COALESCE(excluded.outcome, seen_jobs.outcome), "
                "content_key=COALESCE(excluded.content_key, seen_jobs.content_key)",
                (user_id, job_id, time.time(), outcome, content_key),
            )
            self._conn.commit()

    def record_sweep(self, search_id: str, found: int, queued: int) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO sweeps (search_id, last_run, found, queued) VALUES (?,?,?,?) "
                "ON CONFLICT(search_id) DO UPDATE SET last_run=excluded.last_run, "
                "found=excluded.found, queued=excluded.queued",
                (search_id, time.time(), found, queued),
            )
            self._conn.commit()

    def last_run(self, search_id: str) -> float | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT last_run FROM sweeps WHERE search_id=?", (search_id,)
            ).fetchone()
        return row[0] if row else None

    def stats(self) -> dict[str, Any]:
        with self._lock:
            jobs = self._conn.execute("SELECT COUNT(*) FROM seen_jobs").fetchone()[0]
            sweeps = self._conn.execute(
                "SELECT search_id, last_run, found, queued FROM sweeps"
            ).fetchall()
        return {
            "jobs_seen": jobs,
            "sweeps": [
                {"search_id": s[0], "last_run": s[1], "found": s[2], "queued": s[3]}
                for s in sweeps
            ],
        }

    def close(self) -> None:
        with self._lock:
            self._conn.close()


def normalise_title(title: str) -> str:
    cleaned = re.sub(r"[^a-z0-9 ]+", " ", str(title or "").lower())
    return re.sub(r"\s+", " ", cleaned).strip()


def content_key(job: dict[str, Any]) -> str:
    """A fingerprint for near-duplicate detection: normalised title plus the client.

    **Not** a hash of the description. The reposts in this corpus prepend a sentence
    ("Reposting because the previous freelancer stopped responding.") and then repeat the
    body verbatim — token overlap 0.974 — so any hash anchored at the start of the
    description sees a completely different post. A first attempt hashed title plus the
    opening 400 characters and caught none of them.

    The same client posting the same title is what a human recognises as a repost, and it
    is an exact lookup rather than a scan. `similar_description` below is the second line
    of defence for a repost that was also retitled.

    Derived from the post, never from a `duplicate_of` field: a real marketplace does not
    label its own reposts.
    """
    return hashlib.sha256(
        f"{normalise_title(job.get('title') or '')}|{job.get('client_id') or ''}".encode()
    ).hexdigest()[:16]


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", str(text or "").lower()))


def similar_description(left: dict[str, Any], right: dict[str, Any], floor: float = 0.85) -> bool:
    """Jaccard overlap on description tokens, for a repost that was also retitled.

    Token sets rather than character windows, so inserting or removing a sentence barely
    moves the score. 0.85 is deliberately high: two genuinely different jobs in the same
    niche share plenty of vocabulary, and a false positive here means silently skipping a
    real job, which is worse than analysing one twice.
    """
    a = _tokens(left.get("description"))
    b = _tokens(right.get("description"))
    if not a or not b:
        return False
    return len(a & b) / len(a | b) >= floor


@dataclass
class SavedSearch:
    """One saved search, with its own schedule.

    `interval_sec` is per search because the right cadence depends on the marketplace, not
    on the system. GigSim is local and free; live Upwork is somebody else's infrastructure
    and its interval is justified in docs/RUNBOOK.md.
    """

    search_id: str
    user_id: str
    persona: str
    filters: dict[str, Any] = field(default_factory=dict)
    interval_sec: float = 300.0
    max_queue_per_sweep: int = 5
    min_skill_overlap: int = 2


def default_searches() -> list[SavedSearch]:
    """The searches the demo runs. In production these come from the settings screen."""
    return [
        SavedSearch(
            search_id="ai-ml-hourly",
            user_id="aleeza",
            persona="ai_ml_engineer",
            filters={"limit": 40},
            interval_sec=float(300),
        )
    ]


async def scout_sweep(
    search: SavedSearch,
    ledger: Ledger,
    *,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Find new jobs for one saved search and queue the promising ones.

    Two filters before anything is queued, and both exist to avoid spending tokens:

    * the ledger, so a job is never queued twice;
    * a skill-overlap floor, because running the full graph on a job with no overlap
      produces a correct skip at the cost of several thousand tokens. The cheap signal
      does that for free.
    """
    from mcp_servers.marketplace import server as market
    from mcp_servers.portfolio import tools as portfolio

    try:
        page = await market.search_jobs(**search.filters)
    except Exception as exc:
        return {
            "search_id": search.search_id,
            "error": f"{type(exc).__name__}: {exc}",
            "found": 0,
            "queued": [],
        }

    try:
        own = {row["skill"].lower() for row in portfolio.list_skills(search.persona)["skills"]}
    except Exception:
        own = set()

    jobs = page.get("jobs") or []
    candidates: list[tuple[int, dict[str, Any]]] = []
    duplicates = 0
    # Within one page too, not only against the ledger: a sweep can return a post and its
    # own repost in the same response.
    seen_this_sweep: dict[str, str] = {}

    for job in jobs:
        job_id = str(job.get("job_id") or "")
        if not job_id or ledger.seen(search.user_id, job_id):
            continue

        key = content_key(job)
        prior = seen_this_sweep.get(key) or ledger.seen_content(search.user_id, key)

        # Second line of defence: a repost that was retitled has a different key, so
        # compare descriptions against what this sweep already accepted from the same
        # client.
        if not prior:
            for accepted_overlap, accepted in candidates:
                if accepted.get("client_id") != job.get("client_id"):
                    continue
                if similar_description(accepted, job):
                    prior = str(accepted.get("job_id"))
                    break

        if prior and prior != job_id:
            duplicates += 1
            ledger.mark_seen(
                search.user_id, job_id, outcome=f"duplicate_of:{prior}", content_key=key
            )
            continue

        overlap = len(own & {str(s).lower() for s in (job.get("skills") or [])})
        if overlap < search.min_skill_overlap:
            # Recorded as seen so the next sweep does not reconsider it. A job that does
            # not match today will not match tomorrow.
            ledger.mark_seen(
                search.user_id, job_id, outcome="below_overlap_floor", content_key=key
            )
            continue

        seen_this_sweep[key] = job_id
        candidates.append((overlap, job))

    candidates.sort(key=lambda pair: -pair[0])
    selected = candidates[: search.max_queue_per_sweep]

    queued: list[str] = []
    for overlap, job in selected:
        job_id = str(job["job_id"])
        key = content_key(job)
        if dry_run:
            queued.append(job_id)
            continue
        try:
            await run_bid_thread(job, search.user_id, search.persona)
            ledger.mark_seen(search.user_id, job_id, outcome="queued", content_key=key)
            queued.append(job_id)
        except Exception as exc:
            # One bad job must not abort the sweep.
            ledger.mark_seen(
                search.user_id,
                job_id,
                outcome=f"error:{type(exc).__name__}",
                content_key=key,
            )

    ledger.record_sweep(search.search_id, len(jobs), len(queued))
    return {
        "search_id": search.search_id,
        "found": len(jobs),
        "new_candidates": len(candidates),
        "queued": queued,
        "skipped_as_duplicate": duplicates,
        "skipped_below_overlap": len(jobs) - len(candidates) - duplicates,
    }


async def run_bid_thread(
    job: dict[str, Any], user_id: str, persona: str
) -> dict[str, Any]:
    """Run one job through the bid graph, up to the approval interrupt.

    The thread id is deterministic, so re-running this for the same job resumes rather
    than duplicating. The worker stops at the interrupt and never resumes it: approval is
    a human's to give, and a scheduled job that could approve its own output would make
    the gate meaningless.
    """
    from agent.checkpointer import acheckpointer
    from agent.graphs.bid_graph import build_bid_graph, initial_state
    from agent.state import summarise

    thread_id = f"bid:{user_id}:{job.get('job_id')}"
    config = {"configurable": {"thread_id": thread_id}}

    async with acheckpointer() as saver:
        graph = build_bid_graph(checkpointer=saver)

        existing = await graph.aget_state(config)
        if existing and existing.values:
            # Already run. Report where it got to; do not re-run and do not resume an
            # interrupt.
            return {
                "thread_id": thread_id,
                "already_ran": True,
                "awaiting_approval": bool(existing.interrupts),
                **summarise(dict(existing.values)),
            }

        await graph.ainvoke(
            initial_state(job, persona, freelancer_id=persona, user_id=user_id), config
        )
        snapshot = await graph.aget_state(config)

    values = dict(snapshot.values or {})
    return {
        "thread_id": thread_id,
        "already_ran": False,
        "awaiting_approval": bool(snapshot.interrupts),
        **summarise(values),
    }


async def poll_outcomes(user_id: str, persona: str) -> dict[str, Any]:
    """Advance the simulation and record what clients did, for outcome memory.

    `/admin/tick` is GigSim's own decision step: it scores pending proposals against each
    client's hidden preferences. That is where reply and hire rate come from, rather than
    from a model grading its own work.

    Outcomes go into memory as *observations*. Promotion needs five samples in a category
    and a reply rate clearly off the baseline, so one tick teaches nothing by design.
    """
    from agent.marketplace_port import MarketplaceError
    from mcp_servers.marketplace import server as market

    port = await market._marketplace()
    ticked: dict[str, Any] = {}
    try:
        ticked = await port._request("POST", "/admin/tick")
    except MarketplaceError as exc:
        return {"error": str(exc), "recorded": 0}
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}", "recorded": 0}

    recorded = 0
    try:
        from agent.checkpointer import store as build_store
        from agent.memory.store import Memory

        memory = Memory(build_store(), user_id)
        payload = await port._request(
            "GET", "/proposals", params={"freelancer_id": persona, "limit": 200}
        )
        for proposal in payload.get("proposals") or []:
            status = str(proposal.get("status") or "")
            if status in ("draft", "submitted"):
                continue
            memory.observe_outcome(
                str(proposal.get("job_id") or ""),
                str(proposal.get("job_category") or "uncategorised"),
                status,
                proposal_id=proposal.get("proposal_id"),
            )
            recorded += 1
        promoted = memory.promote()
    except Exception as exc:
        return {"tick": ticked, "recorded": recorded, "error": f"memory: {exc}"}

    return {
        "tick": {"decided": ticked.get("decided"), "outcomes": ticked.get("outcomes")},
        "recorded": recorded,
        "promoted": {
            "style": len(promoted.get("style") or []),
            "outcome": len(promoted.get("outcome") or []),
        },
    }


def summarise_result(result: dict[str, Any]) -> str:
    """One line per task, for the worker log."""
    return json.dumps(result, default=str)[:400]
