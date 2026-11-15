"""Compliance Guard: enforce the policy pack. No tools (A6).

The last check before a human sees the draft, and the only node that can block outright.
It holds no tools on purpose: it judges text against a fixed policy, and a tool call would
only give it a way to be talked into something by the text it is judging.

Rules first, model second, and in that order for a specific reason. Contact details,
off-platform suggestions and guarantees are *findable by pattern*, so patterns find them
and remove them deterministically. That half cannot fail, cannot be argued with, and runs
offline. The model then reads what remains for the violations a pattern cannot catch, such
as a claim that is technically true but misleading in context.

A redaction is reported, never silent. C5 requires the approval screen to show what
changed, and a guard that quietly edits a user's proposal is a guard they will stop
trusting.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from langchain_core.runnables import RunnableConfig

from agent.handoffs import ComplianceVerdict
from agent.nodes._common import call_model, node_trace, step
from agent.state import GigBidState

POLICY_PATH = Path(__file__).resolve().parents[2] / "data" / "policy_pack" / "marketplace_policy.md"

# Findable by pattern, so found by pattern. Each carries the policy rule it enforces, so
# the verdict can cite it rather than saying "policy violation".
_REDACTIONS: list[tuple[str, re.Pattern[str], str]] = [
    ("no off-platform contact", re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"), "[email removed]"),
    # A phone number, in the three shapes one actually appears in. The earlier single
    # pattern matched any run of nine or more digits, so "we processed 1794681917296
    # records" was redacted as off-platform contact and the proposal was blocked from
    # sending. A bare integer is a figure; what makes it a phone number is a country code,
    # separators, or someone telling you to ring it.
    (
        "no off-platform contact",
        # +44 7700 900123, +1 (555) 123-4567
        re.compile(r"(?<!\d)\+\d[\d\s().-]{7,}\d(?!\d)"),
        "[phone removed]",
    ),
    (
        "no off-platform contact",
        # 555-123-4567, 0300 123 4567, (021) 111 222 333
        re.compile(r"(?<![\d.])\(?\d{3,4}\)?[\s.-]+\d{3}[\s.-]+\d{3,4}(?![\d.])"),
        "[phone removed]",
    ),
    (
        "no off-platform contact",
        # "call me on 03001234567": a bare run of digits counts once someone has asked
        # for it to be dialled.
        re.compile(
            r"(?i)\b(?:call|text|ring|phone|whatsapp|reach|contact|"
            r"my\s+(?:number|cell|mobile))\b"
            # "call me on", "reach us at", "text via", "my number is"
            r"(?:\s+(?:me|us))?(?:\s+(?:on|at|via|is))?[:\s]*"
            r"\+?\d[\d\s().-]{6,}\d"
        ),
        "[phone removed]",
    ),
    (
        "no off-platform contact",
        re.compile(r"(?i)\b(?:whatsapp|telegram|skype|wechat|signal)\b[:\s]*[\w.@+-]*"),
        "[contact handle removed]",
    ),
    (
        "no off-platform payment",
        re.compile(r"(?i)\b(?:pay|payment|invoice)\s+(?:me\s+)?(?:directly|outside|off[- ]platform)"),
        "[off-platform payment suggestion removed]",
    ),
    (
        "no guarantees",
        re.compile(
            r"(?i)\bI\s+(?:guarantee|promise|assure you)\b[^.!?]*[.!?]"
        ),
        "",
    ),
    (
        "no guarantees",
        re.compile(r"(?i)\b(?:guaranteed|100% guaranteed)\s+(?:results?|ranking|revenue|success)\b"),
        "[guarantee removed]",
    ),
]


def load_policy() -> str:
    if POLICY_PATH.exists():
        return POLICY_PATH.read_text(encoding="utf-8")
    return (
        "No off-platform contact or payment. No fabricated experience. No guarantees of "
        "results. Fair use of automation."
    )


def apply_redactions(text: str) -> tuple[str, list[str], list[str]]:
    """Strip what policy forbids. Returns (cleaned, redactions, rules_cited)."""
    cleaned = text
    redactions: list[str] = []
    rules: list[str] = []
    for rule, pattern, replacement in _REDACTIONS:
        matches = list(pattern.finditer(cleaned))
        if not matches:
            continue
        for match in matches[:3]:
            redactions.append(f"{rule}: removed {match.group(0)[:48]!r}")
        if rule not in rules:
            rules.append(rule)
        cleaned = pattern.sub(replacement, cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned, redactions, rules


def is_blocked(
    violations: list[str], cleaned_text: str, parsed: dict[str, Any] | None
) -> bool:
    """Should this draft be stopped?

    A violation the guard found and did not remove has to stop the draft, whatever the
    model called its own verdict. The earlier rule required the model to *also* set
    `status: "blocked"`, which meant a policy rule could be detected, quoted in the trace
    by number, and then pass anyway. The rehearsal of the change request found it: a new
    disclosure rule was added to the policy pack, the guard cited it correctly, and the
    offending sentence went through.

    Remediation decides it, not the model's label. If the text the guard objected to is
    gone from the cleaned draft, the objection is satisfied. If it is still there, the
    draft is blocked and a person is told why, which is the safe direction for the one node
    whose job is the platform's rules.

    Matching is on the first eight words of the quoted violation, because the model quotes
    the offending sentence but rarely reproduces its punctuation exactly.
    """
    for quote in violations:
        probe = " ".join(str(quote).split()[:8]).strip(" .,'\"")
        if probe and probe.lower() in cleaned_text.lower():
            return True
    return bool(violations) and str((parsed or {}).get("status")) == "blocked"


async def compliance(state: GigBidState, config: RunnableConfig | None = None) -> dict[str, Any]:
    draft = state.get("proposal_draft")
    if draft is None:
        return {
            "steps_used": step(state),
            "execution_trace": node_trace(
                "compliance", "decision", result_summary={"skipped": "no draft"}
            ),
        }

    original = draft.cover_letter
    cleaned, redactions, rules = apply_redactions(original)

    violations: list[str] = []
    explanation = ""
    meta: dict[str, Any] = {}
    tokens = 0

    parsed, meta = call_model(
        "compliance",
        prompt_values={"policy": load_policy()[:3000], "text": cleaned[:4000]},
        trace_id=state.get("trace_id"),
        job_id=state.get("job_id"),
    )
    tokens = meta.get("tokens") or 0
    if isinstance(parsed, dict):
        violations = [str(v) for v in (parsed.get("violations") or [])][:6]
        explanation = str(parsed.get("explanation") or "")[:400]
        for rule in parsed.get("policy_rules_cited") or []:
            if str(rule) not in rules:
                rules.append(str(rule))
        # The model may propose further cleaning, but only if it did not rewrite the
        # proposal: a "cleaned" text half the length is a rewrite, and the user approved
        # neither version. Length is a cheap, reliable check for that.
        model_cleaned = str(parsed.get("cleaned_text") or "")
        if (
            parsed.get("status") == "fixed"
            and model_cleaned
            and 0.6 * len(cleaned) <= len(model_cleaned) <= len(cleaned) + 200
        ):
            cleaned = model_cleaned

    blocked = is_blocked(violations, cleaned, parsed if isinstance(parsed, dict) else None)
    status = "blocked" if blocked else ("fixed" if redactions else "pass")
    verdict = ComplianceVerdict(
        status=status,
        violations=violations,
        redactions=redactions,
        policy_rules_cited=rules,
        explanation=explanation
        or (
            "removed content the policy pack forbids"
            if redactions
            else "no policy problems found"
        ),
        cleaned_text=cleaned if cleaned != original else None,
    )

    updates: dict[str, Any] = {
        "compliance_verdict": verdict,
        "steps_used": step(state),
        "tokens_used": tokens,
    }

    if status == "blocked":
        # A block is visible and stops the thread before approval. R4 is about not sending
        # without a human; this is about not asking a human to approve something that
        # cannot be sent.
        updates["status"] = "needs_attention"
    elif verdict.cleaned_text:
        updates["proposal_draft"] = draft.model_copy(
            update={"cover_letter": verdict.cleaned_text, "status": "compliant"}
        )

    return {
        **updates,
        "execution_trace": node_trace(
            "compliance",
            "guardrail",
            meta,
            result_summary={
                "status": status,
                "redactions": len(redactions),
                "violations": violations[:3],
                "rules": rules[:4],
                "checked_by": "rule+model" if isinstance(parsed, dict) else "rule",
            },
        ),
    }
