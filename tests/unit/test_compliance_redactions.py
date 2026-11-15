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
