"""Seed loading and mutable simulator state.

The store owns the line between what the marketplace shows an agent and what it keeps to
itself. Clients have hidden preferences and a scam flag; jobs carry the id of the post
they duplicate. None of that crosses the API boundary, so an agent that wants to know
whether a post is a repost has to work it out. That is the point of the Scout.
"""

from __future__ import annotations

import json
import threading
import uuid
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from gigsim.config import Settings, settings

# Fields the marketplace keeps to itself. Everything else on a record is public.
PRIVATE_CLIENT_FIELDS = frozenset({"hidden_preferences", "is_scam_account"})
PRIVATE_JOB_FIELDS = frozenset({"duplicate_of"})

PROPOSAL_STATES = ("submitted", "viewed", "replied", "interview", "hired", "declined")


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def public_job(job: dict[str, Any]) -> dict[str, Any]:
    """A job as an agent may see it."""
    return {k: v for k, v in job.items() if k not in PRIVATE_JOB_FIELDS}


def public_client(client: dict[str, Any]) -> dict[str, Any]:
    """A client profile as an agent may see it."""
    return {k: v for k, v in client.items() if k not in PRIVATE_CLIENT_FIELDS}


class Store:
    """In-memory marketplace state.

    Seed data is read once and never mutated. Everything the agent creates (proposals,
    messages, Connects spending, the audit trail) lives in the mutable half and can be
    wiped with reset(), so an evaluation run always starts from the same place.
    """

    def __init__(self, cfg: Settings | None = None) -> None:
        self.cfg = cfg or settings
        self._lock = threading.RLock()
        self._load_seed()
        self.reset()

    # ---------------------------------------------------------------- seed data

    def _load_seed(self) -> None:
        d = self.cfg.seed_dir
        self.jobs: dict[str, dict] = {j["job_id"]: j for j in _load(d / "jobs_stream.json")}
        self.historical: list[dict] = _load(d / "historical_jobs.json")
        self.clients: dict[str, dict] = {c["client_id"]: c for c in _load(d / "clients.json")}
        self.freelancers: dict[str, dict] = {
            f["freelancer_id"]: f for f in _load(d / "freelancer_accounts.json")
        }
        self.seed_threads: list[dict] = []
        for name in ("conversations_ai.json", "conversations_web.json", "conversations_data.json"):
            path = d / name
            if path.exists():
                self.seed_threads.extend(_load(path))

        derived = self.cfg.derived_dir
        self.price_bands: dict = {}
        self.rate_benchmarks: dict = {}
        if (derived / "price_bands.json").exists():
            self.price_bands = _load(derived / "price_bands.json")
        if (derived / "rate_benchmarks.json").exists():
            self.rate_benchmarks = _load(derived / "rate_benchmarks.json")

    # ------------------------------------------------------------ mutable state

    def reset(self) -> None:
        """Wipe everything the agent has done. Seed data is untouched."""
        with self._lock:
            self.proposals: dict[str, dict] = {}
            self.threads: dict[str, dict] = {}
            self.events: list[dict] = []
            self.idempotency: dict[str, dict] = {}
            self.connects: dict[str, dict] = {
                fid: {
                    "freelancer_id": fid,
                    "balance": f["connects_balance"],
                    "weekly_budget": f["weekly_connects_budget"],
                    "spent_this_week": 0,
                }
                for fid, f in self.freelancers.items()
            }
            for t in self.seed_threads:
                self.threads[t["thread_id"]] = {
                    "thread_id": t["thread_id"],
                    "job_id": t["job_id"],
                    "client_id": t["client_id"],
                    "freelancer_id": t["persona"],
                    "proposal_id": None,
                    "messages": [dict(m) for m in t["messages"]],
                    "data_origin": t.get("data_origin", "synthetic"),
                }

    # ------------------------------------------------------------------- reads

    def search_jobs(
        self,
        q: str | None = None,
        skills: Iterable[str] | None = None,
        category: str | None = None,
        country: str | None = None,
        is_hourly: bool | None = None,
        min_hourly: float | None = None,
        max_hourly: float | None = None,
        posted_after: str | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> tuple[list[dict], int]:
        rows = list(self.jobs.values())
        if q:
            needle = q.lower()
            rows = [
                j
                for j in rows
                if needle in j["title"].lower() or needle in (j.get("description") or "").lower()
            ]
        if skills:
            wanted = {s.lower() for s in skills}
            rows = [j for j in rows if wanted & {s.lower() for s in j.get("skills") or []}]
        if category:
            rows = [j for j in rows if (j.get("category") or "").lower() == category.lower()]
        if country:
            rows = [j for j in rows if (j.get("country") or "").lower() == country.lower()]
        if is_hourly is not None:
            rows = [j for j in rows if bool(j.get("is_hourly")) is is_hourly]
        if min_hourly is not None:
            rows = [j for j in rows if (j.get("hourly_high") or 0) >= min_hourly]
        if max_hourly is not None:
            rows = [j for j in rows if (j.get("hourly_low") or 0) <= max_hourly]
        if posted_after:
            rows = [j for j in rows if (j.get("published_date") or "") >= posted_after]

        rows.sort(key=lambda j: j.get("published_date") or "", reverse=True)
        total = len(rows)
        page = rows[offset : offset + limit]
        return [public_job(j) for j in page], total

    def get_job(self, job_id: str) -> dict | None:
        job = self.jobs.get(job_id)
        return public_job(job) if job else None

    def get_client(self, client_id: str) -> dict | None:
        client = self.clients.get(client_id)
        return public_client(client) if client else None

    def client_history(self, client_id: str) -> dict | None:
        """Public history: what this client has posted and how they behave."""
        client = self.clients.get(client_id)
        if client is None:
            return None
        open_jobs = [public_job(j) for j in self.jobs.values() if j.get("client_id") == client_id]
        past = [j for j in self.historical if j.get("client_id") == client_id]
        return {
            "client": public_client(client),
            "open_jobs": open_jobs,
            "past_job_count": len(past),
            "past_jobs": [public_job(j) for j in past[:20]],
            "reviews_from_freelancers": client.get("reviews_from_freelancers", []),
        }

    def rate_stats(self, category: str | None = None, domain: str | None = None) -> dict:
        """Market rate statistics for the Pricing Strategist."""
        bands = self.price_bands or {}
        out: dict[str, Any] = {"benchmarks": self.rate_benchmarks}
        if domain:
            out["domain"] = (bands.get("by_domain") or {}).get(domain)
        if category:
            out["category"] = (bands.get("by_category") or {}).get(category)
        if not domain and not category:
            out["by_domain"] = bands.get("by_domain", {})
        return out

    def connects_balance(self, freelancer_id: str) -> dict | None:
        return self.connects.get(freelancer_id)

    # ------------------------------------------------------------------ writes

    def log(self, kind: str, **payload: Any) -> None:
        self.events.append({"ts": _now(), "kind": kind, **payload})

    def spend_connects(self, freelancer_id: str, amount: int) -> tuple[bool, str]:
        """Deduct Connects. Returns (ok, reason)."""
        acct = self.connects.get(freelancer_id)
        if acct is None:
            return False, "unknown_freelancer"
        if not self.cfg.connects_enforced:
            return True, "not_enforced"
        if acct["balance"] < amount:
            return False, "insufficient_connects"
        if acct["spent_this_week"] + amount > acct["weekly_budget"]:
            return False, "weekly_budget_exceeded"
        acct["balance"] -= amount
        acct["spent_this_week"] += amount
        return True, "ok"

    def new_id(self, prefix: str) -> str:
        return f"{prefix}-{uuid.uuid4().hex[:12]}"

    def ensure_thread(
        self,
        job_id: str,
        client_id: str,
        freelancer_id: str,
        proposal_id: str | None = None,
    ) -> dict:
        for t in self.threads.values():
            if t["job_id"] == job_id and t["freelancer_id"] == freelancer_id:
                return t
        thread = {
            "thread_id": self.new_id("thr"),
            "job_id": job_id,
            "client_id": client_id,
            "freelancer_id": freelancer_id,
            "proposal_id": proposal_id,
            "messages": [],
            "data_origin": "synthetic",
        }
        self.threads[thread["thread_id"]] = thread
        return thread


store = Store()
