"""The phone-number rule has to tell a phone number from a large figure.

A proposal's whole value is its numbers: rows processed, latency cut, revenue moved. A
redaction rule that treats any long run of digits as off-platform contact blocks exactly
the sentences worth sending. The e2e suite found this the hard way: an edited draft
containing a millisecond timestamp was refused as "no off-platform contact: removed
'1794681917296'".
"""

from __future__ import annotations

import pytest

from agent.nodes.compliance import _REDACTIONS


def flagged(text: str) -> bool:
    return any(pattern.search(text) for _, pattern, _ in _REDACTIONS)


@pytest.mark.parametrize(
    "text",
    [
        "Call me on 03001234567 to discuss.",
        "My number is +44 7700 900123.",
        "Reach me at 555-123-4567 any time.",
        "whatsapp 03121234567",
        "Contact: someone@example.com",
        "Let's move this to Telegram: @handle",
    ],
)
def test_contact_details_are_redacted(text: str):
    assert flagged(text), f"should have been caught: {text}"


@pytest.mark.parametrize(
    "text",
    [
        "We processed 1794681917296 records last quarter.",
        "Revenue grew to 12500000 in 2024.",
        "The model scored 0.892 on 1250 samples.",
        "I cut latency from 1200 ms to 180 ms.",
        "I trained on 1.2 million rows across 250000 sessions.",
        "Edited by the freelancer at 1794681917296.",
    ],
)
def test_ordinary_figures_survive(text: str):
    assert not flagged(text), f"false positive on: {text}"


def test_a_detected_violation_that_is_still_in_the_text_blocks():
    """The gap the change-request rehearsal found.

    A new policy rule was added to the pack, the guard quoted it by number, and the
    offending sentence went through: blocking required the model to label its own verdict
    "blocked" as well as listing the violation. Remediation decides it now.
    """
    from agent.nodes.compliance import is_blocked

    draft = "I personally wrote every word myself with no tools. I built the pipeline."
    violations = ["I personally wrote every word myself with no tools."]

    assert is_blocked(violations, draft, {"status": "fixed"}) is True


def test_a_violation_that_was_removed_does_not_block():
    from agent.nodes.compliance import is_blocked

    cleaned = "I built the pipeline."
    violations = ["I personally wrote every word myself with no tools."]

    assert is_blocked(violations, cleaned, {"status": "fixed"}) is False


def test_no_violations_never_blocks():
    from agent.nodes.compliance import is_blocked

    assert is_blocked([], "any text at all", {"status": "pass"}) is False


def test_the_model_can_still_block_on_its_own_say_so():
    """Belt and braces: a model that says 'blocked' is believed even if its quote drifted."""
    from agent.nodes.compliance import is_blocked

    assert is_blocked(["something it paraphrased"], "unrelated draft", {"status": "blocked"}) is True
