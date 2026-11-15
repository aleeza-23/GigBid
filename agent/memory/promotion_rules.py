"""Promotion rules: what is allowed into long-term memory, and why (C2, R7).

R7 says outcomes and user edits update long-term memory **only through an explicit
promotion rule**. The rule exists because the obvious alternative is worse: writing every
user edit straight to memory means one rushed edit becomes a permanent instruction, and
the agent starts obeying a preference the user never held.

So promotion requires evidence of a *pattern*:

* **Style memory** - a rule promotes after the same edit pattern appears in
  `min_occurrences` separate approvals (default 3). One user cutting one opening sentence
  is a mood; three is a preference.
* **Outcome memory** - a category promotes after `min_samples` proposals with a known
  outcome (default 5). Below that the reply rate is noise: one win in one attempt is not
  a 100% category.

Raw tool output is never eligible. The only inputs are a diff of what the human changed
and an outcome the marketplace reported, both of which are facts about the user's own
behaviour rather than something a model produced.

Everything promoted carries `evidence_count`, `promoted_at` and the rule that admitted it,
so the memory viewer in the UI can show *why* GigBid believes something and the user can
delete it. Viva question 6 asks exactly that.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Literal

MemoryKind = Literal["style", "outcome"]

# Thresholds live here rather than in config because they are the rule itself, not a
# tuning knob; changing them changes what the system is allowed to learn.
MIN_STYLE_OCCURRENCES = 3
MIN_OUTCOME_SAMPLES = 5
# A category whose reply rate is this far from the overall average is worth remembering.
OUTCOME_SIGNIFICANCE = 0.10


@dataclass
class Candidate:
    """An observation that might become memory, with the evidence behind it."""

    kind: MemoryKind
    key: str
    value: Any
    evidence_count: int = 1
    evidence: list[str] = field(default_factory=list)
    rule: str = ""

    def to_record(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "value": self.value,
            "evidence_count": self.evidence_count,
            "evidence": self.evidence[:10],
            "promoted_by": self.rule,
            "promoted_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }


# ------------------------------------------------------------------ style memory

# Patterns in a user's edits that map onto a durable, statable preference. Each is a
# specific, checkable observation rather than a vague "tone" judgement, because a memory
# the Writer cannot act on is clutter.
_STYLE_SIGNALS: list[tuple[str, str, Any]] = [
    ("cut_first_sentence", "always cut the opening sentence", None),
    ("removed_passionate", "never say 'passionate'", "passionate"),
    ("removed_excited", "never say 'excited' or 'thrilled'", "excited"),
    ("removed_proven_track_record", "never say 'proven track record'", "proven track record"),
    ("removed_came_across", "never open with 'I came across your post'", "came across"),
    ("shortened", "prefers a shorter proposal", None),
    ("added_question", "always end with a question to the client", None),
    ("removed_exclamation", "no exclamation marks", "!"),
]

_BANNED_WORDS = {
    "passionate": "removed_passionate",
    "excited": "removed_excited",
    "thrilled": "removed_excited",
    "proven track record": "removed_proven_track_record",
    "came across": "removed_came_across",
}


def diff_edit(original: str, edited: str) -> dict[str, Any]:
    """Describe what a human changed, as checkable signals rather than prose.

    This is the only thing style memory ever learns from, so it reports facts: a phrase
    the user deleted, a length reduction, a question added. A model's opinion about the
    user's "voice" is not admissible, because it cannot be verified later.
    """
    original_sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", original) if s.strip()]
    edited_sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", edited) if s.strip()]

    signals: list[str] = []

    if (
        original_sentences
        and edited_sentences
        and original_sentences[0] != edited_sentences[0]
        and original_sentences[0] not in edited
    ):
        signals.append("cut_first_sentence")

    lowered_original, lowered_edited = original.lower(), edited.lower()
    for phrase, signal in _BANNED_WORDS.items():
        if phrase in lowered_original and phrase not in lowered_edited:
            signals.append(signal)

    if original.count("!") > edited.count("!"):
        signals.append("removed_exclamation")

    # A 15% cut is an intentional trim; 3% is a typo fix.
    if len(edited) < len(original) * 0.85:
        signals.append("shortened")

    if "?" in edited and "?" not in original:
        signals.append("added_question")

    removed_phrases = [
        s for s in original_sentences if s not in edited_sentences and len(s) > 25
    ]

    return {
        "signals": signals,
        "chars_before": len(original),
        "chars_after": len(edited),
        "removed_sentences": removed_phrases[:5],
        "sentences_before": len(original_sentences),
        "sentences_after": len(edited_sentences),
    }


def should_promote_style_edit(edit_count: int, consistent_pattern: bool) -> bool:
    """The scaffold's signature, kept so existing callers keep working.

    Prefer `style_candidates`, which decides per signal rather than for an edit as a
    whole; one edit usually carries several signals of differing strength.
    """
    return edit_count >= MIN_STYLE_OCCURRENCES and consistent_pattern


def style_candidates(
    edit_history: list[dict[str, Any]], min_occurrences: int = MIN_STYLE_OCCURRENCES
) -> list[Candidate]:
    """Signals seen often enough across separate edits to become a style rule.

    Counted across *approvals*, not occurrences within one draft: a user deleting
    "passionate" three times from one proposal is one preference expressed once.
    """
    counts: dict[str, int] = {}
    evidence: dict[str, list[str]] = {}

    for edit in edit_history or []:
        for signal in set(edit.get("signals") or []):
            counts[signal] = counts.get(signal, 0) + 1
            evidence.setdefault(signal, []).append(str(edit.get("job_id") or "unknown"))

    statements = {name: text for name, text, _ in _STYLE_SIGNALS}
    out: list[Candidate] = []
    for signal, count in sorted(counts.items(), key=lambda pair: -pair[1]):
        if count < min_occurrences:
            continue
        out.append(
            Candidate(
                kind="style",
                key=signal,
                value=statements.get(signal, signal),
                evidence_count=count,
                evidence=evidence.get(signal, []),
                rule=f"style: signal seen in {count} separate approvals (>= {min_occurrences})",
            )
        )
    return out


# ---------------------------------------------------------------- outcome memory


def outcome_candidates(
    outcomes: list[dict[str, Any]],
    min_samples: int = MIN_OUTCOME_SAMPLES,
    significance: float = OUTCOME_SIGNIFICANCE,
) -> list[Candidate]:
    """Job categories whose reply rate differs enough from the average to be worth knowing.

    Two guards against learning noise:

    * at least `min_samples` proposals in the category, so one win does not create a
      "100% reply rate" category;
    * a reply rate at least `significance` away from the overall average, so a category
      that behaves like everything else is not stored as if it were a finding.
    """
    by_category: dict[str, list[dict[str, Any]]] = {}
    for row in outcomes or []:
        category = str(row.get("category") or "uncategorised")
        by_category.setdefault(category, []).append(row)

    replied_overall = sum(
        1 for r in outcomes or [] if r.get("outcome") in ("replied", "interview", "hired")
    )
    total = len(outcomes or [])
    if not total:
        return []
    baseline = replied_overall / total

    out: list[Candidate] = []
    for category, rows in by_category.items():
        if len(rows) < min_samples:
            continue
        replied = sum(
            1 for r in rows if r.get("outcome") in ("replied", "interview", "hired")
        )
        won = sum(1 for r in rows if r.get("outcome") == "hired")
        rate = replied / len(rows)
        if abs(rate - baseline) < significance:
            continue

        direction = "above" if rate > baseline else "below"
        out.append(
            Candidate(
                kind="outcome",
                key=category,
                value={
                    "reply_rate": round(rate, 3),
                    "hire_rate": round(won / len(rows), 3),
                    "samples": len(rows),
                    "baseline_reply_rate": round(baseline, 3),
                    "note": (
                        f"{category}: {rate:.0%} reply rate, {direction} the {baseline:.0%} "
                        f"average over {len(rows)} proposals"
                    ),
                },
                evidence_count=len(rows),
                evidence=[str(r.get("job_id") or "") for r in rows],
                rule=(
                    f"outcome: {len(rows)} samples (>= {min_samples}) and "
                    f"{abs(rate - baseline):.0%} from baseline (>= {significance:.0%})"
                ),
            )
        )
    return sorted(out, key=lambda c: -c.evidence_count)


def explain_rules() -> list[dict[str, Any]]:
    """The rules in a form the memory viewer can render.

    Viva question 6 asks what style memory learned and which rule let it in, so the rules
    have to be presentable rather than buried in this docstring.
    """
    return [
        {
            "kind": "style",
            "rule": f"an edit signal must appear in at least {MIN_STYLE_OCCURRENCES} "
            "separate approvals before it becomes a rule",
            "why": "one edit is a mood; three is a preference",
            "inputs": "a diff of what the human changed, never model output",
        },
        {
            "kind": "outcome",
            "rule": f"a category needs at least {MIN_OUTCOME_SAMPLES} proposals with known "
            f"outcomes and a reply rate at least {OUTCOME_SIGNIFICANCE:.0%} from the average",
            "why": "below that, a reply rate is noise and one win looks like a trend",
            "inputs": "outcomes the marketplace reported, never model output",
        },
    ]
