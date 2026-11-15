"""E5: declared budgets against measured values.

R8 and E5 require declaring p95 latency and cost per job analysed and per proposal, then
reporting the measured figures. The declarations live in `agent/config.yaml` under
`budgets`, so the numbers being checked are the same ones the graph enforces rather than
a second set written for the report.

Measurements come from the local span logs in `data/audit/traces/`, which record tokens,
latency and cost per node for every run (E3). That means this script measures *runs that
actually happened* rather than a synthetic benchmark — the honest version, and the one
where a result can be traced back to a specific thread.

A run made with the model offline is excluded. Offline latency is about a millisecond and
offline cost is zero, and averaging those in would produce a budget report that looks
excellent and means nothing.

    make budget-report
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "evals" / "results"


def declared() -> dict[str, Any]:
    from agent.marketplace_port import load_config

    budgets = load_config().get("budgets") or {}
    return {
        "p95_latency_job_analysis_sec": budgets.get("p95_latency_job_analysis_sec"),
        "p95_latency_proposal_sec": budgets.get("p95_latency_proposal_sec"),
        "usd_per_job_analysed": budgets.get("usd_per_job_analysed"),
        "usd_per_proposal": budgets.get("usd_per_proposal"),
        "max_tokens_per_job": budgets.get("max_tokens_per_job"),
        "max_tokens_per_proposal": budgets.get("max_tokens_per_proposal"),
    }


# Nodes that run for any analysed job, versus those that only run when a proposal is
# drafted. The split is what makes "per job analysed" and "per proposal" different
# numbers rather than the same number reported twice.
ANALYSIS_NODES = {"scout", "analyst", "risk_assessor", "fit_matcher", "pricing", "decision"}
PROPOSAL_NODES = {"writer", "verifier", "critic", "compliance", "approval", "submit"}


def collect(trace_dir: Path, include_offline: bool = False) -> dict[str, Any]:
    """Read every span log and split each run into analysis and drafting."""
    runs: list[dict[str, Any]] = []

    for path in sorted(trace_dir.glob("*.jsonl")):
        spans = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not spans:
            continue
        offline = any(span.get("offline") for span in spans)
        if offline and not include_offline:
            continue

        analysis = [s for s in spans if s["node"] in ANALYSIS_NODES]
        proposal = [s for s in spans if s["node"] in PROPOSAL_NODES]

        def total(rows: list[dict[str, Any]], key: str) -> float:
            return sum(float(r.get(key) or 0) for r in rows)

        # A cost total is only reported when every priced span had a known price.
        def usd(rows: list[dict[str, Any]]) -> float | None:
            if any(r.get("usd") is None and r.get("total_tokens") for r in rows):
                return None
            return round(total(rows, "usd"), 6)

        runs.append(
            {
                "trace_id": spans[0].get("trace_id"),
                "job_id": next((s.get("job_id") for s in spans if s.get("job_id")), None),
                "offline": offline,
                "drafted": bool(proposal),
                "analysis_sec": round(total(analysis, "duration_ms") / 1000, 3),
                "analysis_tokens": int(total(analysis, "total_tokens")),
                "analysis_usd": usd(analysis),
                "proposal_sec": round(total(proposal, "duration_ms") / 1000, 3),
                "proposal_tokens": int(total(proposal, "total_tokens")),
                "proposal_usd": usd(proposal),
            }
        )

    return {"runs": runs}


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return round(ordered[0], 3)
    index = min(len(ordered) - 1, round(fraction * (len(ordered) - 1)))
    return round(ordered[index], 3)


def measure(runs: list[dict[str, Any]]) -> dict[str, Any]:
    analysed = runs
    drafted = [r for r in runs if r["drafted"]]

    def stat(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
        values = [float(r[key]) for r in rows if r.get(key) is not None]
        return {
            "n": len(values),
            "mean": round(statistics.mean(values), 4) if values else None,
            "p50": percentile(values, 0.50),
            "p95": percentile(values, 0.95),
            "max": round(max(values), 4) if values else None,
        }

    return {
        "jobs_analysed": len(analysed),
        "proposals_drafted": len(drafted),
        "latency_job_analysis_sec": stat(analysed, "analysis_sec"),
        "latency_proposal_sec": stat(drafted, "proposal_sec"),
        "tokens_per_job_analysed": stat(analysed, "analysis_tokens"),
        "tokens_per_proposal": stat(drafted, "proposal_tokens"),
        "usd_per_job_analysed": stat(analysed, "analysis_usd"),
        "usd_per_proposal": stat(drafted, "proposal_usd"),
    }


def compare(declared_budgets: dict[str, Any], measured: dict[str, Any]) -> list[dict[str, Any]]:
    """One row per declared budget: what was promised, what happened, and whether it held."""
    checks = [
        ("p95 latency, job analysis (s)", "p95_latency_job_analysis_sec",
         measured["latency_job_analysis_sec"]["p95"]),
        ("p95 latency, proposal (s)", "p95_latency_proposal_sec",
         measured["latency_proposal_sec"]["p95"]),
        ("USD per job analysed", "usd_per_job_analysed",
         measured["usd_per_job_analysed"]["mean"]),
        ("USD per proposal", "usd_per_proposal", measured["usd_per_proposal"]["mean"]),
        ("tokens per job", "max_tokens_per_job", measured["tokens_per_job_analysed"]["max"]),
        ("tokens per proposal", "max_tokens_per_proposal",
         measured["tokens_per_proposal"]["max"]),
    ]

    rows: list[dict[str, Any]] = []
    for label, key, value in checks:
        budget = declared_budgets.get(key)
        if budget is None:
            rows.append({"metric": label, "declared": None, "measured": value,
                         "status": "NOT DECLARED"})
        elif value is None:
            # No data is not the same as within budget, and must not be reported as a pass.
            rows.append({"metric": label, "declared": budget, "measured": None,
                         "status": "NO DATA"})
        else:
            rows.append(
                {
                    "metric": label,
                    "declared": budget,
                    "measured": value,
                    "headroom": round(float(budget) - float(value), 6),
                    "status": "WITHIN" if float(value) <= float(budget) else "OVER",
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="E5: declared budgets vs measured")
    parser.add_argument(
        "--include-offline",
        action="store_true",
        help="include runs made with no provider key (their latency and cost are not real)",
    )
    args = parser.parse_args()

    from agent.tracing import TRACE_DIR

    budgets = declared()
    collected = collect(TRACE_DIR, include_offline=args.include_offline)
    runs = collected["runs"]

    print("E5: declared budgets against measured values")
    print(f"spans from {TRACE_DIR}")
    print(f"{len(runs)} run(s) counted" + ("" if args.include_offline else ", offline runs excluded"))
    print()

    if not runs:
        print("No usable runs recorded yet.")
        print("Offline runs are excluded because their latency is ~1ms and their cost is")
        print("zero; averaging those in would produce a flattering, meaningless report.")
        print("Run `make demo` or analyse a job in the UI with a provider key set, then")
        print("rerun this. --include-offline shows the offline runs anyway.")
        return

    measured = measure(runs)
    rows = compare(budgets, measured)

    width = max(len(row["metric"]) for row in rows)
    print(f"{'metric':{width}}  {'declared':>10}  {'measured':>10}  status")
    print("-" * (width + 36))
    for row in rows:
        declared_value = "—" if row["declared"] is None else f"{row['declared']}"
        measured_value = "—" if row["measured"] is None else f"{row['measured']}"
        print(f"{row['metric']:{width}}  {declared_value:>10}  {measured_value:>10}  {row['status']}")

    over = [row for row in rows if row["status"] == "OVER"]
    print()
    if over:
        print(f"{len(over)} budget(s) exceeded:")
        for row in over:
            print(f"  {row['metric']}: {row['measured']} against a declared {row['declared']}")
        print("\nE5 asks for the measured values reported without excuses. Either the")
        print("budget was wrong or the system is too slow; say which in the report.")
    else:
        print("every declared budget held")

    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / "budget_report.json"
    out.write_text(
        json.dumps(
            {
                "declared": budgets,
                "measured": measured,
                "checks": rows,
                "runs_counted": len(runs),
                "offline_included": args.include_offline,
                "source": "data/audit/traces/*.jsonl, written per model call by agent/tracing.py",
                "note": (
                    "Latency is the sum of model-call durations per run, not wall clock: it "
                    "excludes the client-side token-per-minute throttle, which is our own "
                    "rate limit rather than a property of the system."
                ),
                "per_run": runs[:50],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nwritten: {out}")


if __name__ == "__main__":
    main()
