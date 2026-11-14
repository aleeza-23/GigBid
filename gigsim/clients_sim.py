"""Simulated clients.

Every client carries preferences the agent never sees: the skills they actually care
about, what they will pay, whether a stock opening annoys them, how long a proposal they
will read. This module turns a submitted proposal into an outcome by scoring it against
those preferences, which is what makes reply rate and hire rate a real end metric rather
than an LLM judge grading its own homework.

The default engine is deterministic: the same proposal text, the same client and the same
seed always produce the same outcome, so an evaluation run is reproducible. Set
GIGSIM_CLIENT_ENGINE=llm to hand the decision to a model instead.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any

from gigsim.config import settings

# Openings a client who dislikes boilerplate will recognise immediately.
GENERIC_OPENINGS = (
    "dear hiring manager",
    "dear sir",
    "dear madam",
    "to whom it may concern",
    "i hope this message finds you well",
    "i am writing to express",
    "i came across your job",
    "i am the perfect fit",
    "i am excited to apply",
    "greetings of the day",
    "i am a passionate",
    "with great interest",
)

NUMBER_PATTERN = re.compile(r"\b\d+(\.\d+)?\s?(%|percent|x|k|hours|weeks|days|users|ms|rps)?\b")

# What a scam client says when a real freelancer engages with them.
SCAM_REPLIES = (
    (
        "Great. Before we proceed, please complete a short unpaid sample task so we can "
        "assess your skills."
    ),
    "Please add me on Telegram so we can discuss payment outside the platform.",
    "We can start immediately if you cover the initial software licence cost yourself.",
)


@dataclass
class Decision:
    """What a client decided and why. The reason is for your failure analysis, not the agent."""

    outcome: str  # ignored | viewed | replied | interview | hired
    score: float
    reasons: list[str]
    reply_text: str | None = None


def _words(text: str) -> int:
    return len(text.split())


def _jitter(*parts: str) -> float:
    """A stable pseudo-random number in [0, 1) derived from the inputs and the seed."""
    key = "|".join([str(settings.seed), *parts]).encode("utf-8")
    digest = hashlib.sha256(key).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


def score_proposal(client: dict[str, Any], proposal: dict[str, Any]) -> Decision:
    """Score a proposal against one client's hidden preferences.

    client    the full seed record, including hidden_preferences and is_scam_account
    proposal  needs cover_letter, bid_rate and job_id
    """
    prefs = client.get("hidden_preferences") or {}
    letter = (proposal.get("cover_letter") or "").strip()
    lower = letter.lower()
    reasons: list[str] = []
    score = 0.5

    must_have = [s for s in prefs.get("must_have_skills", []) if s]
    if must_have:
        hit = [s for s in must_have if s.lower() in lower]
        coverage = len(hit) / len(must_have)
        score += 0.30 * coverage - 0.15 * (1 - coverage)
        reasons.append(f"must_have_skills {len(hit)}/{len(must_have)} named")

    ceiling = prefs.get("budget_ceiling_usd")
    rate = proposal.get("bid_rate")
    if ceiling is not None and rate is not None:
        if rate > ceiling * 1.25:
            score -= 0.45
            reasons.append(f"rate {rate} far above ceiling {ceiling:.2f}")
        elif rate > ceiling:
            score -= 0.20
            reasons.append(f"rate {rate} above ceiling {ceiling:.2f}")
        else:
            score += 0.10
            reasons.append(f"rate {rate} within ceiling {ceiling:.2f}")

    if prefs.get("dislikes_generic_opening"):
        opening = lower[:160]
        if any(p in opening for p in GENERIC_OPENINGS):
            score -= 0.30
            reasons.append("generic opening")
        else:
            score += 0.05

    if prefs.get("wants_quantified_results"):
        if len(NUMBER_PATTERN.findall(letter)) >= 2:
            score += 0.12
            reasons.append("quantified results present")
        else:
            score -= 0.12
            reasons.append("no quantified results")

    max_words = prefs.get("max_words")
    if max_words and _words(letter) > max_words:
        score -= 0.18
        reasons.append(f"{_words(letter)} words over limit {max_words}")

    if not letter:
        score = 0.0
        reasons.append("empty cover letter")

    score = max(0.0, min(1.0, score))
    # Jitter is keyed on what the proposal says, never on its id: ids are fresh per
    # submission, and an outcome that moves between runs would make evaluation useless.
    fingerprint = (client["client_id"], str(proposal.get("job_id", "")), letter)
    noise = (_jitter(*fingerprint) - 0.5) * 0.12
    effective = max(0.0, min(1.0, score + noise))

    if client.get("is_scam_account"):
        # Scam accounts answer anyone; they just never hire.
        idx = int(_jitter("scam", *fingerprint) * len(SCAM_REPLIES))
        return Decision(
            outcome="replied",
            score=effective,
            reasons=[*reasons, "scam account: replies but never hires"],
            reply_text=SCAM_REPLIES[idx],
        )

    if effective >= 0.80:
        outcome, reply = "hired", "We would like to move forward. Can you start this week?"
    elif effective >= 0.68:
        outcome, reply = "interview", "This looks promising. Do you have time for a short call?"
    elif effective >= 0.52:
        outcome, reply = "replied", "Thanks for applying. A couple of questions before we decide."
    elif effective >= 0.35:
        outcome, reply = "viewed", None
    else:
        outcome, reply = "ignored", None

    return Decision(outcome=outcome, score=effective, reasons=reasons, reply_text=reply)


def reply_to_message(client: dict[str, Any], thread: dict[str, Any], text: str) -> str:
    """A client's answer to a freelancer message in an open thread."""
    if client.get("is_scam_account"):
        idx = int(_jitter("scam-msg", thread["thread_id"], text[:40]) * len(SCAM_REPLIES))
        return SCAM_REPLIES[idx]

    lower = text.lower()
    if any(w in lower for w in ("call", "interview", "meet", "slot", "availability")):
        return "Tuesday or Thursday afternoon works for us. Send a calendar invite."
    if any(w in lower for w in ("rate", "price", "budget", "cost")):
        ceiling = (client.get("hidden_preferences") or {}).get("budget_ceiling_usd")
        if ceiling:
            return f"Our budget tops out near ${ceiling:.0f}/hr. Can you work within that?"
        return "Send your rate and we will compare it with the other applicants."
    if "?" in text:
        return "Good question. The scope is what the post describes, nothing more for now."
    return "Noted, thanks. We are reviewing applications this week."


def decide(client: dict[str, Any], proposal: dict[str, Any]) -> Decision:
    """Entry point. Honours GIGSIM_CLIENT_ENGINE."""
    if settings.client_engine == "llm":
        from gigsim.llm_client import llm_decide  # imported lazily: needs a model

        return llm_decide(client, proposal)
    return score_proposal(client, proposal)
