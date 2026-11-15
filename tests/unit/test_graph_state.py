"""State reducers, serialisation and the memory promotion rules (C2, C6, R7).

These are the parts of Part C that are pure functions, so they are tested without a graph
run. Several encode bugs that were found by running the graph:

* the fan-out lost trace entries until every concurrently written field declared a reducer;
* typed hand-offs could not be read back out of a checkpoint until each class was named in
  the serde allowlist, which LangGraph is going to start enforcing;
* promotion rules silently ran on at most ten observations, because `store.search`
  defaults to `limit=10`.
"""

from __future__ import annotations

from typing import Annotated, get_args, get_origin, get_type_hints

import pytest

from agent import serde
from agent.handoffs import BidDecision, FitReport, RiskReport
from agent.memory import promotion_rules as rules
from agent.memory.store import Memory
from agent.state import (
    ConversationState,
    GigBidState,
    dedupe_evidence,
    keep_first,
    merge_dicts,
    summarise,
    take_max,
    trace,
)

# Written concurrently by the three parallel analysts, so each must declare a reducer.
CONCURRENT_FIELDS = (
    "evidence",
    "analyses_complete",
    "execution_trace",
    "errors",
    "degraded",
    "tokens_used",
    "steps_used",
)


def _reducer_fields(state_type: type) -> set[str]:
    hints = get_type_hints(state_type, include_extras=True)
    return {
        name
        for name, hint in hints.items()
        if get_origin(hint) is Annotated and len(get_args(hint)) > 1
    }


def test_every_concurrently_written_field_declares_a_reducer() -> None:
    """The bug this prevents: three nodes writing `execution_trace` kept one entry.

    Without a reducer LangGraph cannot know whether a second write extends or replaces
    the first, so it replaces, and two of the three analysts' traces vanish.
    """
    declared = _reducer_fields(GigBidState)
    for field in CONCURRENT_FIELDS:
        assert field in declared, (
            f"{field} is written by parallel branches but has no reducer, so writes "
            "from two of the three analysts would be lost"
        )


def test_conversation_state_also_declares_its_reducers() -> None:
    declared = _reducer_fields(ConversationState)
    for field in ("execution_trace", "errors", "risk_flags", "degraded", "tokens_used"):
        assert field in declared


def test_keep_first_protects_the_idempotency_key() -> None:
    """A resumed thread must not mint a new key: that is how Connects get spent twice."""
    assert keep_first("key-from-approval", "key-from-retry") == "key-from-approval"
    # An absent value is replaced, so the first real write wins.
    assert keep_first(None, "key-from-approval") == "key-from-approval"
    assert keep_first("", "key-from-approval") == "key-from-approval"


def test_take_max_stops_counters_going_backwards() -> None:
    """Two branches reporting step counts must not let the budget reset."""
    assert take_max(7, 3) == 7
    assert take_max(None, 4) == 4
    assert take_max(0, 0) == 0


def test_merge_dicts_combines_degraded_reports_from_several_branches() -> None:
    left = {"marketplace": "503"}
    right = {"portfolio": "corpus missing"}
    assert merge_dicts(left, right) == {"marketplace": "503", "portfolio": "corpus missing"}


def test_trace_returns_a_list_so_entries_accumulate() -> None:
    """A node returning a bare dict would replace the whole trace with one entry."""
    entry = trace("writer", "model", tokens=120)
    assert isinstance(entry, list)
    assert entry[0]["node"] == "writer"
    assert entry[0]["kind"] == "model"
    assert entry[0]["tokens"] == 120
    assert entry[0]["ts"]


def test_evidence_is_deduplicated_on_read_keeping_the_best_score() -> None:
    """Evidence concatenates across branches, so the same item arrives several times."""
    merged = dedupe_evidence(
        [
            {"id": "ai-p01-case", "score": 0.4},
            {"id": "ai-p01-case", "score": 0.9},
            {"id": "ai-p02-readme", "score": 0.5},
            {"id": "", "score": 1.0},
        ]
    )
    assert [item["id"] for item in merged] == ["ai-p01-case", "ai-p02-readme"]
    assert merged[0]["score"] == 0.9, "the higher-scoring copy should survive"


def test_summarise_omits_the_execution_trace() -> None:
    """The pipeline board lists many threads; shipping every tool call would be megabytes."""
    state: GigBidState = {
        "job_id": "upw-1",
        "status": "awaiting_approval",
        "decision": BidDecision(decision="bid", reasons=["good coverage"]),
        "fit_report": FitReport(score=0.8, coverage=0.8),
        "risk_report": RiskReport(score=0.1, explanation="clean"),
        "execution_trace": trace("scout", "tool_call") + trace("writer", "model"),
    }
    view = summarise(state)
    assert "execution_trace" not in view
    assert view["trace_entries"] == 2
    assert view["decision"] == "bid"
    assert view["fit_coverage"] == 0.8


# ------------------------------------------------------------------ serialisation


def test_every_handoff_class_is_allowed_out_of_a_checkpoint() -> None:
    """C3 meets C6: typed hand-offs are checkpointed, so each class must be allowlisted.

    LangGraph currently warns and will later refuse. The day it refuses, an unlisted class
    stops every pending approval in the database from loading.
    """
    from agent import handoffs

    required = {
        "JobSpec",
        "RiskReport",
        "FitReport",
        "PriceQuote",
        "BidDecision",
        "ProposalDraft",
        "Critique",
        "ComplianceVerdict",
    }
    allowed = {name for _, name in serde.ALLOWED_MSGPACK_MODULES}
    assert required <= allowed, f"not allowlisted: {sorted(required - allowed)}"
    # And every allowlisted name must actually exist, so a rename cannot leave a stale entry.
    for name in allowed:
        assert hasattr(handoffs, name), f"{name} is allowlisted but not in agent.handoffs"


def test_the_serializer_is_an_allowlist_not_a_blanket_permit() -> None:
    assert serde.describe()["pickle_fallback"] is False
    assert serde.ALLOWED_MSGPACK_MODULES, "an empty allowlist would permit nothing"
    assert all(len(pair) == 2 for pair in serde.ALLOWED_MSGPACK_MODULES), (
        "entries are (module, class) pairs; naming the module alone allows nothing"
    )


def test_handoffs_survive_a_round_trip_through_the_serializer() -> None:
    serializer = serde.serializer()
    original = FitReport(score=0.75, coverage=0.75, matched_skills=["Python"])
    kind, blob = serializer.dumps_typed(original)
    restored = serializer.loads_typed((kind, blob))
    assert isinstance(restored, FitReport)
    assert restored.coverage == 0.75
    assert restored.matched_skills == ["Python"]


# -------------------------------------------------------------- promotion rules


class _DictStore:
    """A store without pagination, to prove `_all` copes with either kind."""

    def __init__(self) -> None:
        self.data: dict[tuple, dict] = {}

    def put(self, namespace, key, value):
        self.data[(namespace, key)] = value

    def search(self, namespace_prefix, **_):
        class Item:
            def __init__(self, value):
                self.value = value

        return [Item(v) for (ns, _), v in self.data.items() if ns == namespace_prefix]

    def delete(self, namespace, key):
        self.data.pop((namespace, key), None)


def test_a_style_signal_needs_three_separate_approvals() -> None:
    """R7: one edit is a mood, three is a preference."""
    memory = Memory(_DictStore(), "u")
    original = "I am passionate about ML. I built a classifier that hit 89%."
    edited = "I built a classifier that hit 89%."

    for job in ("j1", "j2"):
        memory.observe_edit(job, original, edited)
    assert memory.promote()["style"] == [], "two occurrences must not promote"

    memory.observe_edit("j3", original, edited)
    promoted = {record["key"] for record in memory.promote()["style"]}
    assert "removed_passionate" in promoted
    assert "cut_first_sentence" in promoted


def test_promoted_rules_carry_the_rule_that_admitted_them() -> None:
    """Viva question 6: show the promotion rule that let an item in."""
    memory = Memory(_DictStore(), "u")
    for job in ("a", "b", "c"):
        memory.observe_edit(job, "I am excited! Built X.", "Built X.")
    record = memory.promote()["style"][0]
    assert record["evidence_count"] >= 3
    assert record["promoted_by"].startswith("style:")
    assert record["evidence"], "the jobs behind the rule must be listed"
    assert record["promoted_at"]


def test_outcome_memory_needs_enough_samples_and_a_real_difference() -> None:
    memory = Memory(_DictStore(), "u")
    # Four samples is below the threshold, however lopsided.
    for i in range(4):
        memory.observe_outcome(f"a{i}", "Deep Learning", "hired")
    assert memory.promote()["outcome"] == []

    # Six in each of two categories, with clearly different rates.
    memory = Memory(_DictStore(), "u")
    for i in range(6):
        memory.observe_outcome(f"d{i}", "Deep Learning", "hired" if i < 4 else "ignored")
    for i in range(6):
        memory.observe_outcome(f"w{i}", "WordPress", "ignored")

    promoted = {record["key"]: record for record in memory.promote()["outcome"]}
    assert set(promoted) == {"Deep Learning", "WordPress"}
    assert promoted["Deep Learning"]["value"]["reply_rate"] > promoted["WordPress"]["value"]["reply_rate"]
    # Baseline over all twelve observations, not a truncated page of ten.
    assert promoted["Deep Learning"]["value"]["baseline_reply_rate"] == pytest.approx(0.333, abs=0.01)


def test_promotion_reads_every_observation_not_the_first_page() -> None:
    """Regression: `store.search` defaults to limit=10, which skewed every baseline."""
    from langgraph.store.memory import InMemoryStore

    memory = Memory(InMemoryStore(), "u")
    for i in range(25):
        memory.observe_outcome(f"j{i}", "NLP" if i % 2 else "Vision", "hired" if i < 5 else "ignored")
    observations = memory._all("outcome_observations")
    assert len(observations) == 25, f"only {len(observations)} of 25 observations were read"


def test_a_user_can_delete_what_was_learned() -> None:
    """Viva question 6 asks for exactly this, from the UI."""
    memory = Memory(_DictStore(), "u")
    for job in ("a", "b", "c"):
        memory.observe_edit(job, "I am passionate! Built X.", "Built X.")
    memory.promote()
    assert any(r["key"] == "removed_passionate" for r in memory.style_rules())

    assert memory.forget("style", "removed_passionate") is True
    assert not any(r["key"] == "removed_passionate" for r in memory.style_rules())

    # The evidence is still there, so a later promotion would re-derive it. Deleting the
    # observations too is the stronger "this was never my preference".
    assert memory.promote()["style"]
    memory.forget_observations("style")
    memory.store.data.clear() if hasattr(memory.store, "data") else None


def test_only_promoted_rules_reach_the_writer() -> None:
    """An observation below the threshold must not influence drafting."""
    memory = Memory(_DictStore(), "u")
    memory.observe_edit("j1", "I am passionate about ML.", "ML work.")
    assert memory.style_instructions() == [], "one observation is not a style rule"


def test_the_promotion_rules_are_explainable() -> None:
    explained = rules.explain_rules()
    assert len(explained) == 2
    for entry in explained:
        assert entry["rule"] and entry["why"] and entry["inputs"]
        # R7: raw tool output must never be an input to memory.
        assert "model output" in entry["inputs"] or "never" in entry["inputs"]


def test_diff_edit_reports_checkable_signals_not_opinions() -> None:
    diff = rules.diff_edit(
        "I am passionate about this! I built a classifier that hit 89% accuracy. "
        "I also enjoy long walks.",
        "I built a classifier that hit 89% accuracy. What does success look like?",
    )
    assert "cut_first_sentence" in diff["signals"]
    assert "removed_passionate" in diff["signals"]
    assert "removed_exclamation" in diff["signals"]
    assert "added_question" in diff["signals"]
    assert diff["chars_after"] < diff["chars_before"]
