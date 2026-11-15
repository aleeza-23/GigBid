"""The CI quality gate (E2): a 30-job eval checked against committed thresholds.

E2 asks for a smoke eval on every pull request that fails if unsupported claims rise or
bid-or-skip F1 drops past thresholds we set. The thresholds live in
`evals/thresholds.json` with a note on why each number is what it is, because a gate
nobody can justify gets raised until it stops failing and then means nothing.

Runs offline and on the lightweight retrieval backends, which is what makes it usable in
CI: no provider key, no 130MB model download, about a minute. That costs some fidelity —
the absolute numbers differ from a full run with BGE and a cross-encoder — so the gate
measures *regression against the same configuration*, not absolute quality. The reported
numbers in `evals/results/` come from the full run.

    make smoke-eval
    python -m evals.runners.smoke_eval --limit 30
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
THRESHOLDS = ROOT / "evals" / "thresholds.json"
RESULTS = ROOT / "evals" / "results"


def load_thresholds() -> dict[str, Any]:
    if not THRESHOLDS.exists():
        raise SystemExit(f"missing {THRESHOLDS}")
    return json.loads(THRESHOLDS.read_text(encoding="utf-8"))


def extract(report: dict[str, Any]) -> dict[str, float]:
    """Flatten the decision report into the values the gate checks."""
    bos = report.get("bid_or_skip") or {}
    scam = report.get("scam_detection") or {}
    pricing = report.get("pricing") or {}
    injection = report.get("injection_defence") or {}

    return {
        "bid_or_skip_accuracy": float(bos.get("accuracy") or 0.0),
        "bid_or_skip_macro_f1": float(bos.get("macro_f1") or 0.0),
        "scam_recall": float(scam.get("recall") or 0.0),
        "price_inside_fair_band": float(pricing.get("inside_fair_band") or 0.0),
        "injection_detection_rate": float(injection.get("detection_rate") or 0.0),
        "injection_attack_success_rate": float(injection.get("attack_success_rate") or 0.0),
        "injection_false_positives": float(injection.get("false_positives_on_clean_jobs") or 0),
    }


def check(measured: dict[str, float], thresholds: dict[str, Any]) -> list[dict[str, Any]]:
    """Compare measured values against the gate. Returns one row per checked metric."""
    rows: list[dict[str, Any]] = []

    for metric, floor in (thresholds.get("minimums") or {}).items():
        if metric not in measured:
            rows.append({"metric": metric, "status": "MISSING", "note": "not in the report"})
            continue
        value = measured[metric]
        rows.append(
            {
                "metric": metric,
                "measured": round(value, 4),
                "bound": f">= {floor}",
                "status": "PASS" if value >= float(floor) else "FAIL",
            }
        )

    for metric, ceiling in (thresholds.get("maximums") or {}).items():
        if metric not in measured:
            # Not every maximum comes from the decision report; unsupported claims come
            # from a drafting run, which the smoke eval does not do. Reported as skipped
            # rather than silently passing.
            rows.append(
                {"metric": metric, "status": "SKIPPED", "note": "needs a drafting run"}
            )
            continue
        value = measured[metric]
        rows.append(
            {
                "metric": metric,
                "measured": round(value, 4),
                "bound": f"<= {ceiling}",
                "status": "PASS" if value <= float(ceiling) else "FAIL",
            }
        )

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="CI quality gate (E2)")
    parser.add_argument("--limit", type=int, default=30, help="jobs to evaluate")
    # The shipped variant, so the gate guards what users get.
    parser.add_argument("--variant", default="V1")
    parser.add_argument(
        "--write-baseline",
        action="store_true",
        help="print thresholds derived from this run, to paste into thresholds.json",
    )
    args = parser.parse_args()

    from evals.runners.decision_eval import run

    thresholds = load_thresholds()

    print(f"smoke eval: {args.limit} jobs, retrieval {args.variant}")
    print(f"gate: {THRESHOLDS.relative_to(ROOT)}\n")

    report = run(args.variant, None, args.limit)
    measured = extract(report)

    # Claim verification, from the labelled probe set. It needs no provider key and takes
    # a couple of milliseconds a probe, so the unsupported-claims threshold can be a real
    # gate in CI instead of being skipped for want of a drafting run. The probes are
    # generated from the corpora by evals/runners/verify_eval.py.
    try:
        from evals.runners.verify_eval import aggregate, build_probes, score_probe
        from mcp_servers.portfolio.corpus import VALID_PERSONAS

        scored = [
            score_probe(probe)
            for persona in list(VALID_PERSONAS)
            for probe in build_probes(persona, 6)
        ]
        verification = aggregate(scored)
        missed = verification["fabrications"] - verification["fabrications_caught"]
        measured["unsupported_claims_per_proposal"] = round(missed / max(1, len(scored)), 3)
        measured["fabrication_recall"] = verification["fabrication_recall"]
    except Exception as exc:  # pragma: no cover - the gate must still report the rest
        print(f"  (claim verification could not run: {type(exc).__name__}: {exc})")

    rows = check(measured, thresholds)

    width = max(len(str(row["metric"])) for row in rows)
    for row in rows:
        if row["status"] in ("SKIPPED", "MISSING"):
            print(f"  {row['status']:7} {row['metric']!s:{width}}  {row.get('note', '')}")
        else:
            print(
                f"  {row['status']:7} {row['metric']!s:{width}}  "
                f"{row['measured']:>8}  needs {row['bound']}"
            )

    failures = [row for row in rows if row["status"] == "FAIL"]

    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / "smoke_eval.json"
    out.write_text(
        json.dumps(
            {
                "jobs": args.limit,
                "variant": args.variant,
                "configuration": (
                    "offline model, lightweight retrieval backends; this gate detects "
                    "regression against the same configuration, not absolute quality"
                ),
                "measured": measured,
                "checks": rows,
                "passed": not failures,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    if args.write_baseline:
        print("\nthresholds derived from this run (floors set ~10% under measured):")
        print(
            json.dumps(
                {
                    "minimums": {
                        key: round(max(0.0, value * 0.9), 2)
                        for key, value in measured.items()
                        if not key.startswith("injection_attack")
                        and not key.endswith("false_positives")
                    }
                },
                indent=2,
            )
        )

    print(f"\nwritten: {out}")
    if failures:
        print(f"\nFAILED {len(failures)} gate(s):")
        for row in failures:
            print(f"  {row['metric']}: {row['measured']} does not satisfy {row['bound']}")
        print("\nIf this is an intended change, update evals/thresholds.json in the same")
        print("commit and say why in the PR.")
        sys.exit(1)

    print("\nall gates passed")


if __name__ == "__main__":
    main()
