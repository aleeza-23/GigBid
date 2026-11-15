"""Claim verification (B3): split a draft into claims and check each against evidence.

R5 says no sentence in a proposal may claim experience the portfolio does not support.
This module is where that is enforced, and where the unsupported-claim rate reported in
B4 comes from.

Three ideas carry most of the weight:

**Not every sentence is a claim.** "What does success look like in your first two weeks?"
asserts nothing. Neither does "My rate is $45/hour", which is a statement about the
present, not about past work. Treating every sentence as a claim inflates the denominator
and makes the metric look better after verification than it deserves, so claims are
classified before they are checked.

**Numbers are the sharpest test available.** "Routing accuracy rose from 71% to 89%" is
checkable without a model: either those figures appear in the cited evidence or the claim
is fabricated. A number that appears nowhere is the single most reliable fabrication
signal in this corpus, and catching it costs nothing.

**Verification runs without a model, and better with one.** The lexical and numeric checks
are deterministic and always run. An LLM pass refines `partly_supported` cases, where
wording differs but meaning may hold. The deterministic floor means CI and the offline
path still verify, rather than silently passing everything through.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

Verdict = Literal["supported", "partly_supported", "unsupported", "not_a_claim"]

# Sentence splitter that respects decimals, abbreviations and percentages, so
# "accuracy rose to 89.5%. Then..." does not split mid-number.
_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z(])")

_NUMBER = re.compile(r"\d+(?:\.\d+)?\s*(?:%|percent|x|k\b|m\b)?", re.IGNORECASE)

# Phrases that assert past work. A sentence with one of these is making a claim about
# experience and has to be backed.
_EXPERIENCE_MARKERS = (
    "i built", "i delivered", "i shipped", "i developed", "i implemented", "i designed",
    "i led", "i migrated", "i deployed", "i created", "i wrote", "i reduced", "i improved",
    "i increased", "i cut", "i raised", "i automated", "i integrated", "i trained",
    "i fine-tuned", "i scaled", "i optimised", "i optimized", "i have built",
    "i have worked", "i have shipped", "i have delivered", "i have used", "i have",
    "my work on", "for a client", "in my last", "previously i", "i was responsible",
    "we built", "we delivered", "we shipped",
)

# Credentials and qualifications. A fabricated degree is the most damaging claim a
# proposal can carry and the easiest to write, so these are checked separately: they are
# assertions about the past that use none of the verbs above.
_CREDENTIAL_MARKERS = (
    "i hold", "i have a degree", "i graduated", "i studied", "i am certified",
    "i am a certified", "my degree", "phd", "ph.d", "msc", "m.sc", "bsc", "b.sc",
    "master's", "masters in", "bachelor", "mba", "certified in", "certification in",
    "i am an aws", "i am a google", "years of experience", "years experience",
)

# The qualifications themselves, as they would be written in a CV. Separate from the
# markers above, which are the verbs that *assert* a qualification.
# Capitalised words that are ordinary English rather than the name of a tool. Without
# this list every sentence mentioning a weekday or a language would be read as naming a
# technology the portfolio has to prove.
_NOT_A_PRODUCT = frozenset(
    {
        "i", "a", "an", "the", "this", "that", "these", "those", "my", "your", "our",
        "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
        "january", "february", "march", "april", "may", "june", "july", "august",
        "september", "october", "november", "december",
        "english", "european", "american", "british", "indian", "german",
        "ai", "ml", "nlp", "api", "apis", "ui", "ux", "it", "sdk", "cv",
    }
)

# A token shaped like the name of a tool: all-caps (COBOL, SQL), capitalised mid-sentence
# (Snowflake, TensorFlow), or carrying the punctuation that version and language names use
# (C++, C#, Node.js, Python3).
_NAMED_TOOL = re.compile(
    r"\b(?:[A-Z]{2,}[0-9+#.]*|[A-Z][a-zA-Z]*[A-Z][a-zA-Z0-9+#.]*|[A-Z][a-z]{2,}[0-9+#.]*"
    r"|[A-Za-z]+(?:\+\+|#|\.js|\.net))\b"
)


_CREDENTIAL_NOUNS = frozenset(
    {
        "phd", "ph.d", "doctorate", "dphil",
        "msc", "m.sc", "ms", "mba", "masters", "master", "mphil",
        "bsc", "b.sc", "bs", "ba", "beng", "btech", "bachelor", "bachelors",
        "degree", "diploma", "certified", "certification", "certificate",
    }
)

# Sentences that assert nothing about past work.
_NOT_CLAIM_MARKERS = (
    "my rate", "my hourly rate", "i charge", "i am available", "i can start",
    "what does success", "could you share", "would you prefer", "let me know",
    "happy to", "i would", "i will", "i can ", "shall i", "do you",
)

_CITATION = re.compile(r"portfolio://[\w-]+/([\w.-]+)")

_STOP = frozenset(
    """
    a an the and or but if then than that this these those of in on at to for with by from
    as is are was were be been being do does did have has had having i you we they my your
    our their it its so not no can will would should could may might must about into over
    under again once here there when where why how all any both each few more most other
    some such only own same too very just also like need needs client clients work working
    project projects experience
    """.split()
)


def _strip_citations(text: str) -> str:
    """Remove citation handles before any textual analysis.

    Without this, `portfolio://case_study/ai-p01-case` contributes the "number" 01 and the
    tokens "portfolio" and "case", so a claim gets judged partly on the shape of its own
    footnote. That cost a true claim its support on the first run of this module.
    """
    return _CITATION.sub(" ", text)


def _tokens(text: str) -> set[str]:
    return {
        w
        for w in re.findall(r"[a-z0-9+#.]+", _strip_citations(text).lower())
        if w not in _STOP and len(w) > 1
    }


def _numbers(text: str) -> set[str]:
    """Numeric values in a string, normalised so '89%' and '89 percent' match."""
    out: set[str] = set()
    for raw in _NUMBER.findall(_strip_citations(text)):
        value = re.sub(r"\s+", "", raw.lower()).replace("percent", "%")
        # Trailing ".0" is noise: 89 and 89.0 are the same claim.
        value = re.sub(r"\.0+(?=[%a-z]|$)", "", value)
        out.add(value)
    return out


def _unsupported_tools(claim_text: str, evidence_tokens: set[str]) -> list[str]:
    """Named tools in a claim that appear in none of the evidence.

    Content overlap alone cannot catch this. "I built the churn prediction pipeline using
    COBOL" shares every word but one with the case study it cites, so it clears a 30%
    overlap threshold while asserting experience with a technology that appears nowhere in
    the portfolio. On the generated probe set the lexical check caught 36 of 36 inflated
    numbers and 36 of 36 false credentials, and 0 of 36 wrong-technology claims, which is
    the fabrication a freelancer is most likely to be caught out on in an interview.

    The first word is skipped: it is capitalised by sentence position rather than by being
    a name. A tool the evidence does mention passes, whatever its case.

    The deliberate trade-off: a sentence that names the *client's* stack is flagged too,
    for example "I can take over your Snowflake warehouse" when the portfolio has no
    Snowflake work. That reads as a false positive and is the behaviour R5 asks for, since
    the sentence asserts the freelancer can do the work. A claim meant as an offer rather
    than a record of experience should be written as one, and `classify` then treats it as
    not a claim.
    """
    body = _strip_citations(claim_text).strip()
    body = body.split(" ", 1)[-1] if " " in body else ""
    missing: list[str] = []
    for token in _NAMED_TOOL.findall(body):
        lowered = token.lower().strip(".")
        if lowered in _NOT_A_PRODUCT or lowered in _STOP:
            continue
        if lowered in evidence_tokens:
            continue
        if lowered not in missing:
            missing.append(lowered)
    return missing


@dataclass
class Claim:
    """One checkable assertion from a draft."""

    text: str
    index: int
    kind: str = "experience"
    """experience | credential | other. Credentials are checked differently."""
    verdict: Verdict = "unsupported"
    support: list[str] = field(default_factory=list)
    """Evidence ids that back this claim."""
    cited: list[str] = field(default_factory=list)
    """Citations the writer attached to this sentence."""
    reason: str = ""
    lexical_overlap: float = 0.0
    unmatched_numbers: list[str] = field(default_factory=list)
    unmatched_tools: list[str] = field(default_factory=list)
    """Named technologies the claim asserts that no evidence item mentions."""

    @property
    def is_claim(self) -> bool:
        return self.verdict != "not_a_claim"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def classify(sentence: str) -> bool:
    """Does this sentence assert something about past work?

    Deliberately conservative in both directions: an explicit experience marker makes it a
    claim, an explicit rate or question marker makes it not one, and anything carrying a
    number plus a past-tense verb is treated as a claim because that is the shape of a
    fabricated result.
    """
    lowered = sentence.lower().strip()

    # A sentence that cites evidence is always checkable, whatever else it looks like.
    # The citation itself is an assertion about provenance, and a handle pointing at
    # evidence that was never retrieved has to be caught even when the surrounding
    # sentence would otherwise read as harmless.
    if _CITATION.search(sentence):
        return True

    if len(lowered) < 15:
        return False
    if lowered.endswith("?"):
        return False

    # Credentials outrank the not-a-claim list: "I hold a PhD" contains no experience verb
    # and would otherwise slip through as conversational filler.
    if any(marker in lowered for marker in _CREDENTIAL_MARKERS):
        return True

    if any(marker in lowered for marker in _NOT_CLAIM_MARKERS) and not any(
        marker in lowered for marker in _EXPERIENCE_MARKERS
    ):
        return False
    if any(marker in lowered for marker in _EXPERIENCE_MARKERS):
        return True
    # A number next to a past-tense verb: "accuracy rose from 71% to 89%".
    if _NUMBER.search(lowered) and re.search(
        r"\b\w+(?:ed|ose|ell|ut|uilt|ade)\b", lowered
    ):
        return True
    return False


def split_claims(draft: str) -> list[Claim]:
    """Split a draft into sentences and mark which ones are checkable claims."""
    claims: list[Claim] = []
    index = 0
    for block in draft.split("\n"):
        block = block.strip()
        if not block:
            continue
        for sentence in _SENTENCE.split(block):
            sentence = sentence.strip()
            if not sentence:
                continue
            lowered = sentence.lower()
            claim = Claim(
                text=sentence,
                index=index,
                kind=(
                    "credential"
                    if any(m in lowered for m in _CREDENTIAL_MARKERS)
                    else "experience"
                ),
                cited=_CITATION.findall(sentence),
            )
            if not classify(sentence):
                claim.verdict = "not_a_claim"
                claim.reason = "asserts nothing about past work"
            claims.append(claim)
            index += 1
    return claims


def _evidence_text(item: dict[str, Any]) -> str:
    return " ".join(
        str(item.get(key) or "")
        for key in ("text", "project", "client_industry")
    ) + " " + " ".join(str(s) for s in (item.get("skills") or []))


def verify_claim(
    claim: Claim, evidence: list[dict[str, Any]], *, min_overlap: float = 0.30
) -> Claim:
    """Check one claim against retrieved evidence.

    A claim is `supported` when its content words substantially appear in one evidence
    item *and* every number it states appears there too. Spreading support across several
    items is not enough on its own: a claim assembled from two unrelated projects is how a
    plausible-sounding fabrication gets built.
    """
    if claim.verdict == "not_a_claim":
        return claim

    claim_tokens = _tokens(claim.text)
    claim_numbers = _numbers(claim.text)
    if not claim_tokens:
        claim.verdict = "unsupported"
        claim.reason = "no content words to check"
        return claim

    by_id = {str(item.get("id")): item for item in evidence}

    # A citation the writer attached is checked first and held to a higher standard: if
    # the writer named its source, that source has to actually say it.
    candidates = [by_id[c] for c in claim.cited if c in by_id] or list(evidence)
    bad_citations = [c for c in claim.cited if c not in by_id]

    best_overlap = 0.0
    best_id = ""
    best_numbers_ok = False

    for item in candidates:
        item_tokens = _tokens(_evidence_text(item))
        if not item_tokens:
            continue
        overlap = len(claim_tokens & item_tokens) / len(claim_tokens)
        item_numbers = _numbers(_evidence_text(item))
        numbers_ok = claim_numbers.issubset(item_numbers) if claim_numbers else True
        # Prefer an item that accounts for the numbers, even at slightly lower overlap:
        # a fabricated figure matters more than a few shared words.
        better = (numbers_ok, overlap) > (best_numbers_ok, best_overlap)
        if better:
            best_overlap, best_id, best_numbers_ok = overlap, str(item.get("id")), numbers_ok

    claim.lexical_overlap = round(best_overlap, 3)
    claim.support = [best_id] if best_id and best_overlap >= min_overlap / 2 else []

    if bad_citations:
        claim.verdict = "unsupported"
        claim.reason = f"cites evidence that was never retrieved: {bad_citations}"
        return claim

    # A credential claim needs its *specific* terms in the evidence, not merely the same
    # subject matter. "I hold a PhD in machine learning" shares "machine learning" with
    # almost every item in an ML portfolio, which is enough to clear a general overlap
    # threshold while the assertion itself — the degree — appears nowhere. Topic overlap
    # is the wrong test for a qualification, so the distinctive terms are required.
    if claim.kind == "credential":
        # The qualification itself and any institution named, never the verb used to
        # assert it. No CV contains the word "hold", so including the claiming verb here
        # rejected "I hold a BS in Computer Science" against a CV that says exactly that.
        claim_tokens_raw = _tokens(claim.text)
        distinctive = claim_tokens_raw & _CREDENTIAL_NOUNS
        # Proper nouns: an institution named in the claim has to appear in the evidence.
        # Skip the first word, which is capitalised by sentence position rather than by
        # being a name.
        body = _strip_citations(claim.text).split(" ", 1)[-1]
        distinctive |= {w.lower() for w in re.findall(r"\b[A-Z][a-z]{3,}\b", body)}

        corpus_tokens: set[str] = set()
        for item in candidates:
            corpus_tokens |= _tokens(_evidence_text(item))
        missing = sorted(t for t in distinctive if t not in corpus_tokens)
        if missing:
            claim.verdict = "unsupported"
            claim.reason = (
                f"credential claim: the evidence never mentions {missing}. "
                "Topic overlap is not proof of a qualification."
            )
            return claim

    # Every technology the claim names has to appear in the evidence, for the same reason
    # the numbers do: a single substituted tool turns a true sentence into one the
    # freelancer cannot defend, and it changes too few words to move the overlap score.
    evidence_tokens: set[str] = set()
    for item in candidates:
        evidence_tokens |= _tokens(_evidence_text(item))
    unmatched_tools = _unsupported_tools(claim.text, evidence_tokens)
    if unmatched_tools:
        claim.unmatched_tools = unmatched_tools
        claim.verdict = "unsupported"
        claim.reason = (
            f"names technology absent from the evidence: {unmatched_tools}"
        )
        return claim

    if claim_numbers and not best_numbers_ok:
        unmatched = sorted(
            claim_numbers
            - set().union(*[_numbers(_evidence_text(i)) for i in candidates] or [set()])
        )
        claim.unmatched_numbers = unmatched
        claim.verdict = "unsupported"
        claim.reason = f"states figures absent from the evidence: {unmatched}"
        return claim

    if best_overlap >= min_overlap:
        claim.verdict = "supported"
        claim.reason = f"{best_overlap:.0%} content overlap with {best_id}"
    elif best_overlap >= min_overlap / 2:
        claim.verdict = "partly_supported"
        claim.reason = f"only {best_overlap:.0%} overlap with {best_id}; wording may stretch it"
    else:
        claim.verdict = "unsupported"
        claim.reason = "no retrieved evidence covers this"

    return claim


# Prompt for the optional model pass. Kept here rather than in agent/prompts/ because it
# belongs to the RAG layer, not to a graph node.
_LLM_SYSTEM = """You check whether a claim in a freelance proposal is supported by evidence.

Reply with JSON only: {"verdict": "supported" | "partly_supported" | "unsupported",
"reason": "<one short sentence>"}

Rules:
- supported: the evidence states this, even in different words.
- partly_supported: the evidence shows something adjacent but weaker, or the claim
  generalises beyond what is shown.
- unsupported: the evidence does not show this, or states a different number.
Judge only against the evidence given. Absence of evidence is unsupported, not supported."""


def verify_proposal(
    draft: str,
    evidence: list[dict[str, Any]],
    *,
    llm: Any = None,
    min_overlap: float = 0.30,
    refine_partials: bool = True,
) -> dict[str, Any]:
    """Verify a whole draft. Returns claims, verdicts and the unsupported-claim rate.

    The deterministic pass always runs. When `llm` is given, claims the lexical pass
    marked `partly_supported` are re-judged by the model, because that is where wording
    differs but meaning may hold and where a lexical check is weakest. Clear verdicts are
    left alone: paying a model to re-confirm an obvious match buys nothing.
    """
    claims = [verify_claim(c, evidence, min_overlap=min_overlap) for c in split_claims(draft)]
    checkable = [c for c in claims if c.is_claim]
    llm_calls = 0

    if llm is not None and refine_partials:
        by_id = {str(item.get("id")): item for item in evidence}
        for claim in checkable:
            if claim.verdict != "partly_supported":
                continue
            relevant = [by_id[i] for i in claim.support if i in by_id] or evidence[:3]
            block = "\n\n".join(
                f"[{item.get('id')}] {str(item.get('text') or '')[:600]}" for item in relevant
            )
            try:
                reply = llm.chat(
                    "verifier",
                    [
                        {"role": "system", "content": _LLM_SYSTEM},
                        {
                            "role": "user",
                            "content": f"EVIDENCE:\n{block}\n\nCLAIM:\n{claim.text}",
                        },
                    ],
                    json_mode=True,
                )
                llm_calls += 1
                parsed = reply.json()
                verdict = str(parsed.get("verdict") or "").strip()
                if verdict in ("supported", "partly_supported", "unsupported"):
                    claim.verdict = verdict  # type: ignore[assignment]
                    claim.reason = f"model: {parsed.get('reason') or verdict}"
            except Exception as exc:
                # A verifier that cannot run must not silently approve. The lexical
                # verdict stands, which is the conservative direction.
                claim.reason += f" (model check failed: {type(exc).__name__})"

    unsupported = [c for c in checkable if c.verdict == "unsupported"]
    partly = [c for c in checkable if c.verdict == "partly_supported"]
    supported = [c for c in checkable if c.verdict == "supported"]

    return {
        "claims": [c.to_dict() for c in claims],
        "total_sentences": len(claims),
        "checkable_claims": len(checkable),
        "supported": len(supported),
        "partly_supported": len(partly),
        "unsupported": len(unsupported),
        # The B4 metric. Per claim, not per sentence, so padding a draft with questions
        # cannot improve the score.
        "unsupported_rate": round(len(unsupported) / len(checkable), 4) if checkable else 0.0,
        "unsupported_texts": [c.text for c in unsupported],
        "llm_refinements": llm_calls,
        "evidence_ids_available": [str(i.get("id")) for i in evidence],
    }


def strip_unsupported(draft: str, report: dict[str, Any]) -> tuple[str, list[str]]:
    """Remove unsupported sentences from a draft before the Critic sees it (B3).

    Removal rather than rewriting: rewriting an unsupported claim with the same model that
    invented it tends to produce a vaguer version of the same falsehood. The Writer gets
    told what was cut and why, and may try again with evidence.
    """
    removed = list(report.get("unsupported_texts") or [])
    if not removed:
        return draft, []

    cleaned = draft
    for sentence in removed:
        cleaned = cleaned.replace(sentence, "")
    # Tidy the gaps left behind.
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = "\n".join(line.strip() for line in cleaned.split("\n"))
    return cleaned.strip(), removed


def verify_claims(draft: str, evidence: list[dict]) -> list[dict]:
    """The signature the scaffold defined, kept so existing callers keep working."""
    return verify_proposal(draft, evidence)["claims"]
