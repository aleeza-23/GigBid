"""Checkpoint serialisation for the typed hand-offs (C3 meets C6).

C3 wants agents exchanging Pydantic objects. C6 wants those objects surviving a crash in a
Postgres checkpoint. Those two requirements meet here.

LangGraph's default serializer will deserialise an unregistered class but warns that it
will stop doing so:

    Deserializing unregistered type agent.handoffs.FitReport from checkpoint.
    This will be blocked in a future version.

Left alone, that is a durability bug waiting for a dependency bump: the day it becomes an
error, every pending approval in the database stops loading. So the hand-off module is
declared explicitly, which both silences the warning now and pins what is allowed back out
of a checkpoint.

Declaring an allowlist rather than passing `True` is the point. `True` permits any type,
which means a checkpoint can reconstruct arbitrary classes from whatever is importable —
fine in a test, not something to run against a shared database.
"""

from __future__ import annotations

from typing import Any

# The allowlist is (module, class) pairs, not module names: LangGraph checks the exact
# type, so naming the module alone allows nothing.
#
# Every hand-off in C3's list. All are plain Pydantic models holding data, with no
# __init__ side effects, which is what makes them safe to reconstruct from a checkpoint.
HANDOFF_CLASSES: tuple[str, ...] = (
    "JobSpec",
    "RiskReport",
    "FitReport",
    "PriceQuote",
    "BidDecision",
    "ProposalDraft",
    "Critique",
    "ComplianceVerdict",
)

ALLOWED_MSGPACK_MODULES: tuple[tuple[str, str], ...] = tuple(
    ("agent.handoffs", name) for name in HANDOFF_CLASSES
)


def serializer() -> Any:
    """The serializer every checkpointer is built with, durable or not.

    Deliberately strict. Passing `True` would permit any importable type back out of a
    checkpoint, which is fine in a test and not something to point at a shared database.
    Adding a hand-off means adding it to HANDOFF_CLASSES, and the test in
    tests/unit/test_graph_state.py fails if the two lists drift.
    """
    from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

    return JsonPlusSerializer(allowed_msgpack_modules=ALLOWED_MSGPACK_MODULES)


def describe() -> dict[str, Any]:
    return {
        "allowed_msgpack_modules": [f"{m}.{c}" for m, c in ALLOWED_MSGPACK_MODULES],
        "pickle_fallback": False,
        "why": (
            "typed hand-offs are checkpointed, so each class must be explicitly allowed "
            "back out; an allowlist beats permitting every importable type"
        ),
    }
