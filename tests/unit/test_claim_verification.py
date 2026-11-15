"""Claim verification (B3), the enforcement point for R5.

Each test here corresponds to a way a proposal can lie. Three of them were real bugs
found by running the module on a deliberately mixed draft: a citation handle contributing
a phantom number, a fabricated credential classified as conversational filler, and a
dangling citation never checked because classification returned early.
"""

from __future__ import annotations

from rag.verify import (
    classify,
    split_claims,
    strip_unsupported,
    verify_claim,
    verify_proposal,
)

EVIDENCE = [
    {
        "id": "ai-p01-case",
        "type": "case_study",
        "project": "Support ticket triage classifier",
        "skills": ["Machine Learning", "Python", "scikit-learn"],
        "client_industry": "SaaS",
        "text": (
            "The support team tagged about 4,000 tickets a month by hand and urgent "
            "tickets waited. I built a TF-IDF and logistic-regression baseline, then "
            "fine-tuned a small transformer and served it from asynchronous workers. "
            "Routing accuracy rose from 71% to 89% and first-response time fell by 35%."
        ),
    },
    {
        "id": "ai-p02-readme",
        "type": "readme",
        "project": "policy-qa",
        "skills": ["LangChain", "pgvector"],
        "client_industry": "Legal",
        "text": "Retrieval-augmented question answering over policy documents using pgvector.",
    },
]


def test_a_grounded_claim_with_matching_numbers_is_supported() -> None:
    report = verify_proposal(
        "I built a TF-IDF and logistic-regression baseline, then fine-tuned a small "
        "transformer, and routing accuracy rose from 71% to 89%.",
        EVIDENCE,
    )
    assert report["supported"] == 1
    assert report["unsupported"] == 0
    assert report["unsupported_rate"] == 0.0


def test_a_fabricated_number_is_caught() -> None:
    """The sharpest check available, and it needs no model.

    The surrounding words all appear in the evidence; only the figures are invented, which
    is exactly how a plausible fabrication reads.
    """
    report = verify_proposal(
        "I fine-tuned a transformer for ticket triage and routing accuracy rose from "
        "71% to 99.8%.",
        EVIDENCE,
    )
    assert report["unsupported"] == 1
    claim = next(c for c in report["claims"] if c["verdict"] == "unsupported")
    assert "99.8%" in claim["unmatched_numbers"]


def test_a_claim_about_work_that_is_not_in_the_portfolio_is_unsupported() -> None:
    report = verify_proposal(
        "I built a real-time bidding engine handling 2 million requests per second for "
        "an ad exchange.",
        EVIDENCE,
    )
    assert report["unsupported"] == 1


def test_a_citation_handle_does_not_contribute_phantom_numbers() -> None:
    """Regression: `ai-p01-case` used to yield the number 01, failing a true claim."""
    report = verify_proposal(
        "I fine-tuned a small transformer and served it from asynchronous workers "
        "(portfolio://case_study/ai-p01-case).",
        EVIDENCE,
    )
    assert report["unsupported"] == 0, report["claims"]


def test_a_dangling_citation_is_always_checked() -> None:
    """Regression: a citation to evidence that was never retrieved must never pass.

    The sentence is a credential rather than an experience claim, so the old classifier
    marked it not-a-claim and skipped the citation check entirely.
    """
    report = verify_proposal(
        "I hold a PhD in machine learning from Stanford "
        "(portfolio://case_study/ai-p99-does-not-exist).",
        EVIDENCE,
    )
    assert report["unsupported"] == 1
    claim = next(c for c in report["claims"] if c["verdict"] == "unsupported")
    assert "never retrieved" in claim["reason"]


def test_a_fabricated_credential_is_a_claim() -> None:
    """Regression: a degree nobody has is the most damaging thing a proposal can assert."""
    assert classify("I hold a PhD in machine learning from Stanford.")
    assert classify("I am certified in AWS machine learning.")
    assert classify("I graduated with a masters in statistics.")

    report = verify_proposal("I hold a PhD in machine learning from Stanford.", EVIDENCE)
    assert report["checkable_claims"] == 1
    assert report["unsupported"] == 1


def test_a_genuine_credential_backed_by_the_cv_is_supported() -> None:
    """The other side of the credential check, which is easy to over-tighten.

    An earlier version treated the verb "hold" as part of the credential, and since no CV
    contains the word "hold" it rejected a degree the CV states outright.
    """
    cv = {
        "id": "ai-cv",
        "type": "cv",
        "project": None,
        "skills": ["Python", "Machine Learning"],
        "client_industry": None,
        "text": "## Education\nBS Computer Science (synthetic)\n\n## Experience\n5 years.",
    }
    report = verify_proposal("I hold a BS in Computer Science.", [cv])
    assert report["unsupported"] == 0, report["claims"]

    # And the fabricated version against the same CV still fails.
    bad = verify_proposal("I hold a PhD from Stanford.", [cv])
    assert bad["unsupported"] == 1
    assert "stanford" in bad["claims"][0]["reason"].lower()


def test_questions_and_rates_are_not_claims() -> None:
    """Padding a draft with pleasantries must not improve the unsupported rate."""
    assert not classify("What does success look like for you in the first two weeks?")
    assert not classify("My rate for this work is $45/hour.")
    assert not classify("I can start on Monday.")

    report = verify_proposal(
        "My rate for this work is $45/hour. What does success look like for you?",
        EVIDENCE,
    )
    assert report["checkable_claims"] == 0
    assert report["unsupported_rate"] == 0.0


def test_the_rate_is_per_claim_not_per_sentence() -> None:
    """Two claims, one bad, plus three harmless sentences, is still 50%."""
    draft = (
        "You need a ticket classifier.\n"
        "I fine-tuned a small transformer for ticket triage and accuracy rose to 89%.\n"
        "I also built a trading system for a hedge fund handling 5 billion events.\n"
        "My rate is $45/hour.\n"
        "What does success look like?"
    )
    report = verify_proposal(draft, EVIDENCE)
    assert report["checkable_claims"] == 2
    assert report["unsupported"] == 1
    assert report["unsupported_rate"] == 0.5


def test_unsupported_claims_are_removed_before_the_critic_sees_the_draft() -> None:
    draft = (
        "I fine-tuned a small transformer for ticket triage and accuracy rose to 89%.\n"
        "I led a team of 40 engineers at Google."
    )
    report = verify_proposal(draft, EVIDENCE)
    cleaned, removed = strip_unsupported(draft, report)

    assert len(removed) == 1
    assert "Google" not in cleaned
    assert "transformer" in cleaned, "the supported claim must survive"


def test_splitting_keeps_decimals_and_percentages_intact() -> None:
    claims = split_claims("Accuracy rose to 89.5%. Latency fell by 35%.")
    assert len(claims) == 2
    assert "89.5%" in claims[0].text


def test_a_claim_citing_retrieved_evidence_is_held_to_that_evidence() -> None:
    """Naming a source raises the bar: that source has to actually say it."""
    claim = next(
        c
        for c in split_claims(
            "I built a retrieval system over policy documents with pgvector "
            "(portfolio://readme/ai-p02-readme)."
        )
        if c.is_claim
    )
    verified = verify_claim(claim, EVIDENCE)
    assert verified.support == ["ai-p02-readme"]
    assert verified.verdict in ("supported", "partly_supported")


def test_verification_works_with_no_evidence_at_all() -> None:
    """A Writer that retrieved nothing must not be able to claim anything."""
    report = verify_proposal("I built a ticket triage classifier.", [])
    assert report["unsupported"] == 1
    assert report["unsupported_rate"] == 1.0
