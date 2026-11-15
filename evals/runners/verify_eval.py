"""B3/B4: how well does claim verification catch fabrication, and what does it cost?

The B5 table asks for "unsupported claims per proposal". Counting them in model output
alone cannot say whether the verifier is any good: a draft with zero flagged claims is
either well grounded or badly checked, and the number is the same either way. So this
runner measures the verifier against claims whose status is known, and reports the
per-proposal count as a second, separate figure.

The probe set is built from the real portfolio corpus rather than written by hand, so it
can be regenerated and nothing is labelled by assertion:

* **supported** - a sentence restating an evidence item's own project and skills.
* **wrong technology** - the same sentence with a technology that appears nowhere in the
  persona's corpus. The runner checks that absence before using the token, so the label
  is verified rather than assumed.
* **inflated number** - a figure from the item's text, multiplied. The claim is otherwise
  identical, which isolates the number check.
* **false credential** - a degree the CV does not contain.
* **not a claim** - a question about the client's situation. These must be classified
  `not_a_claim`; counting them as supported would let a draft improve its score by
  adding questions.

Two error directions are reported separately because they cost different things. A missed
fabrication can lose the contract and the account. A supported claim wrongly stripped
deletes a true selling point, which is cheaper but not free, and a verifier tuned until it
removes everything would score perfectly on the first measure.

    make verify-eval
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "evals" / "results"

# Technologies for the wrong-technology probes. Each is checked against the persona's
# corpus before use, and one that turns out to be present is skipped.
CANDIDATE_ABSENT_TECH = [
    "COBOL",
    "Fortran",
    "Erlang",
    "Haskell",
    "LabVIEW",
    "ColdFusion",
    "Objective-C",
]

NUMBER = re.compile(r"\b(\d{1,3}(?:,\d{3})*|\d+)(\.\d+)?\b")


def _corpus_tokens(items: list[Any]) -> set[str]:
    blob = " ".join(
        f"{i.text} {i.project or ''} {' '.join(i.skills)} {i.client_industry or ''}" for i in items
    ).lower()
    return set(re.findall(r"[a-z][a-z0-9+#.-]*", blob))


def _absent_tech(items: list[Any]) -> str | None:
    present = _corpus_tokens(items)
    for tech in CANDIDATE_ABSENT_TECH:
        if not any(part.lower() in present for part in tech.split()):
            return tech
    return None


def _first_number(text: str) -> str | None:
    match = NUMBER.search(text)
    return match.group(0) if match else None


def build_probes(persona: str, limit: int = 12) -> list[dict[str, Any]]:
    """One probe proposal per evidence item, with a label per sentence."""
    from mcp_servers.portfolio.corpus import load_corpus

    items = load_corpus(persona)
    tech = _absent_tech(items)
    if tech is None:
        raise SystemExit(
            f"every candidate technology appears in {persona}'s corpus; "
            "add one to CANDIDATE_ABSENT_TECH that does not"
        )

    # Case studies: they name a project and state outcomes, which is what a claim cites.
    sources = [i for i in items if i.type == "case_study" and i.project and i.skills][:limit]

    probes: list[dict[str, Any]] = []
    for item in sources:
        skills = ", ".join(item.skills[:3])
        evidence = [
            {
                "id": item.id,
                "text": item.text,
                "project": item.project,
                "project_id": item.project_id,
                "skills": item.skills,
                "type": item.type,
            }
        ]

        sentences: list[tuple[str, str, str]] = [
            (
                f"I built {item.project} using {skills}.",
                "supported",
                "restates the item's own project and skills",
            ),
            (
                f"I built {item.project} using {tech}.",
                "unsupported",
                f"wrong technology: {tech} is absent from this corpus",
            ),
            (
                f"Have you decided how the {item.project.split()[0].lower()} work "
                "should be scoped?",
                "not_a_claim",
                "a question about the client's situation",
            ),
            (
                "I hold a PhD in machine learning from Stanford.",
                "unsupported",
                "false credential: no such degree in the CV",
            ),
        ]

        original = _first_number(item.text)
        if original:
            inflated = str(int(float(original.replace(",", "")) * 7) + 3)
            sentences.append(
                (
                    f"On {item.project} I improved the headline metric to {inflated}.",
                    "unsupported",
                    f"inflated number: the item states {original}",
                )
            )

        probes.append(
            {
                "persona": persona,
                "evidence_id": item.id,
                "project": item.project,
                "absent_tech": tech,
                "evidence": evidence,
                "sentences": [
                    {"text": text, "label": label, "why": why} for text, label, why in sentences
                ],
            }
        )
    return probes


def _key(text: str) -> str:
    """A sentence's identity for matching verdicts back to labels.

    The whole normalised sentence, not a prefix. An earlier version keyed on the first six
    words, which made "I built X using Python" and "I built X using COBOL" the same key:
    the fabrication's verdict overwrote the supported claim's, and the run reported all 36
    supported claims as wrongly flagged when the verifier had judged every one of them
    correctly.
    """
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def score_probe(probe: dict[str, Any], llm: Any = None) -> dict[str, Any]:
    from rag.verify import strip_unsupported, verify_proposal

    draft = " ".join(s["text"] for s in probe["sentences"])
    started = time.perf_counter()
    report = verify_proposal(draft, probe["evidence"], llm=llm)
    elapsed_ms = (time.perf_counter() - started) * 1000

    verdicts = {_key(c["text"]): c for c in report["claims"]}

    rows = []
    for sentence in probe["sentences"]:
        found = verdicts.get(_key(sentence["text"]))
        verdict = (found or {}).get("verdict") or "missing"
        if sentence["label"] == "not_a_claim":
            correct = verdict == "not_a_claim"
        elif sentence["label"] == "supported":
            correct = verdict == "supported"
        else:
            # partly_supported counts as caught: such a claim does not ship as written,
            # because the Critic is shown the partial verdict and the Writer must narrow
            # it before the draft can be approved.
            correct = verdict in ("unsupported", "partly_supported")
        rows.append(
            {
                "text": sentence["text"],
                "label": sentence["label"],
                "verdict": verdict,
                "correct": correct,
                "why": sentence["why"],
            }
        )

    stripped, removed = strip_unsupported(draft, report)
    kept_supported = [
        r for r in rows if r["label"] == "supported" and r["text"].rstrip(".") in stripped
    ]

    return {
        "evidence_id": probe["evidence_id"],
        "persona": probe["persona"],
        "rows": rows,
        "checkable_claims": report["checkable_claims"],
        "unsupported_before": report["unsupported"],
        "removed": len(removed),
        "supported_kept": len(kept_supported),
        "supported_total": sum(1 for r in rows if r["label"] == "supported"),
        "latency_ms": round(elapsed_ms, 2),
        "llm_refinements": report.get("llm_refinements") or 0,
    }


def aggregate(scored: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [row for probe in scored for row in probe["rows"]]

    fabrications = [r for r in rows if r["label"] == "unsupported"]
    caught = [r for r in fabrications if r["correct"]]
    flagged = [r for r in rows if r["verdict"] in ("unsupported", "partly_supported")]
    false_flags = [r for r in flagged if r["label"] == "supported"]

    recall = len(caught) / len(fabrications) if fabrications else 0.0
    precision = (len(flagged) - len(false_flags)) / len(flagged) if flagged else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    supported_rows = [r for r in rows if r["label"] == "supported"]
    questions = [r for r in rows if r["label"] == "not_a_claim"]

    by_kind: dict[str, dict[str, Any]] = {}
    for row in fabrications:
        kind = row["why"].split(":")[0]
        bucket = by_kind.setdefault(kind, {"n": 0, "caught": 0})
        bucket["n"] += 1
        bucket["caught"] += int(row["correct"])

    latencies = sorted(p["latency_ms"] for p in scored)

    return {
        "probes": len(scored),
        "sentences": len(rows),
        "fabrication_recall": round(recall, 4),
        "flag_precision": round(precision, 4),
        "f1": round(f1, 4),
        "fabrications": len(fabrications),
        "fabrications_caught": len(caught),
        "supported_claims": len(supported_rows),
        "supported_wrongly_flagged": len(false_flags),
        "questions": len(questions),
        "questions_misread_as_claims": sum(1 for r in questions if not r["correct"]),
        "unsupported_per_proposal_before": round(
            statistics.mean([p["unsupported_before"] for p in scored]), 3
        )
        if scored
        else 0.0,
        "unsupported_per_proposal_after": 0.0,
        "supported_claims_kept_after_stripping": sum(p["supported_kept"] for p in scored),
        "supported_claims_total": sum(p["supported_total"] for p in scored),
        "p95_latency_ms": round(latencies[max(0, round(0.95 * (len(latencies) - 1)))], 2)
        if latencies
        else 0.0,
        "by_fabrication_kind": by_kind,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="B3/B4 claim-verification evaluation")
    parser.add_argument("--personas", nargs="+", default=None)
    parser.add_argument("--limit", type=int, default=12, help="probes per persona")
    parser.add_argument(
        "--llm",
        action="store_true",
        help="let the verifier re-judge partly-supported claims with a model",
    )
    args = parser.parse_args()

    from mcp_servers.portfolio.corpus import VALID_PERSONAS

    personas = args.personas or list(VALID_PERSONAS)

    llm = None
    if args.llm:
        from agent.llm import get_llm

        llm = get_llm()

    scored: list[dict[str, Any]] = []
    for persona in personas:
        for probe in build_probes(persona, args.limit):
            scored.append(score_probe(probe, llm=llm))

    summary = aggregate(scored)
    # After stripping, a caught fabrication is gone from the draft, so what remains per
    # proposal is whatever the verifier missed.
    summary["unsupported_per_proposal_after"] = round(
        (summary["fabrications"] - summary["fabrications_caught"]) / max(1, len(scored)), 3
    )

    print(
        f"B3 claim verification over {summary['probes']} probe proposals, "
        f"{summary['sentences']} labelled sentences"
    )
    print(f"personas: {', '.join(personas)}")
    print(f"verifier: lexical{' + model on partial verdicts' if llm else ''}")
    print()
    print(
        f"fabrications caught       {summary['fabrications_caught']}/{summary['fabrications']}"
        f"  (recall {summary['fabrication_recall']:.3f})"
    )
    print(f"flag precision            {summary['flag_precision']:.3f}   F1 {summary['f1']:.3f}")
    print(
        f"supported wrongly flagged {summary['supported_wrongly_flagged']}"
        f"/{summary['supported_claims']}"
    )
    print(
        f"questions misread         {summary['questions_misread_as_claims']}"
        f"/{summary['questions']}"
    )
    print()
    print(
        f"unsupported claims per proposal  before "
        f"{summary['unsupported_per_proposal_before']:.2f}"
        f"   after stripping {summary['unsupported_per_proposal_after']:.2f}"
    )
    print(
        f"supported claims surviving       "
        f"{summary['supported_claims_kept_after_stripping']}"
        f"/{summary['supported_claims_total']}"
    )
    print(f"p95 verification latency         {summary['p95_latency_ms']:.1f} ms")
    print()
    print("by fabrication kind:")
    for kind, bucket in sorted(summary["by_fabrication_kind"].items()):
        print(f"  {kind:28} {bucket['caught']}/{bucket['n']}")

    misses = [
        row
        for probe in scored
        for row in probe["rows"]
        if row["label"] == "unsupported" and not row["correct"]
    ]
    if misses:
        print()
        print(f"{len(misses)} fabrication(s) not caught:")
        for row in misses[:6]:
            print(f"  {row['text'][:88]}  ({row['why']})")

    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / "claim_verification.json"
    out.write_text(
        json.dumps(
            {
                "summary": summary,
                "model_assisted": bool(llm),
                "per_probe": scored,
                "note": (
                    "Probes are generated from the portfolio corpus by "
                    "evals/runners/verify_eval.py; the absent-technology label is checked "
                    "against the corpus before use. partly_supported counts as caught, "
                    "because such a claim does not ship as written."
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nwritten: {out}")


if __name__ == "__main__":
    main()
