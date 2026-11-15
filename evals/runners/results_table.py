"""B5: regenerate the results table from the committed result files.

The assignment requires "a script that regenerates the results table" and that every
number in the README and the report trace to a file in `evals/results/`. So this script
reads only those files and writes `evals/results/results_table.md`. It computes nothing
itself, which is the point: if a number appears here, there is a JSON file behind it.

Missing cells are printed as `-` with a footnote saying which runner produces them.
Nothing is estimated, interpolated or carried over from a previous run. A blank column in
a submitted report is a known gap; a plausible-looking number with no file behind it is a
fabrication, and the assignment names that as misconduct.

    make results-table
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

RESULTS = Path(__file__).resolve().parents[2] / "evals" / "results"

# Which runner fills which column, so a gap tells you what to run next.
SOURCES = {
    "rag_eval.json": ("Recall@5, nDCG@10", "make rag-eval"),
    "decision_eval.json": ("bid/skip F1, scam P/R, price in band", "make decision-eval"),
    "ablation.json": ("component contributions", "make ablation"),
    "judge_calibration.json": ("judge score, Cohen's kappa", "make judge-calibration"),
    # GigSim does not model client replies, so there is no runner to produce this and the
    # column stays a gap. A number here would have to be invented.
    "gigsim_outcomes.json": (
        "reply rate, hire rate",
        "not measurable: GigSim does not model client replies",
    ),
    "claim_verification.json": ("unsupported claims per proposal", "make verify-eval"),
    "rnd_a_tool_granularity.json": ("tool-selection accuracy", "make tool-study"),
    "rnd_c_agent_comparison.json": ("single vs supervisor vs graph", "make agent-comparison"),
    "c6_crash_resume.json": ("exactly-once across a restart", "make crash-resume"),
    "budget_report.json": ("declared budgets vs measured", "make budget-report"),
    "x1_agency_routing.json": ("X1 routing accuracy and coverage gained", "make agency-eval"),
}


def load(name: str) -> dict[str, Any] | None:
    path = RESULTS / name
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def fmt(value: Any, spec: str = ".3f", scale: float = 1.0) -> str:
    if value is None:
        return "-"
    try:
        return format(float(value) * scale, spec)
    except (TypeError, ValueError):
        return str(value)


def build() -> str:
    rag = load("rag_eval.json")
    decision = load("decision_eval.json")
    ablation = load("ablation.json")
    judge = load("judge_calibration.json")
    outcomes = load("gigsim_outcomes.json")
    claims = load("claim_verification.json")
    tools = load("rnd_a_tool_granularity.json")
    comparison = load("rnd_c_agent_comparison.json")
    durability = load("c6_crash_resume.json")
    budgets = load("budget_report.json")
    agency = load("x1_agency_routing.json")

    lines: list[str] = ["# Results", ""]
    missing: list[str] = []

    # ------------------------------------------------------------- provenance
    lines.append("Every number below is read from a file in `evals/results/`. Cells marked")
    lines.append("`-` have no result file yet; the footnotes say which command produces them.")
    lines.append("")

    if rag and not rag.get("embeddings_spec_compliant", True):
        lines.append("> **These retrieval numbers are not spec-compliant.** The dense half ran on")
        lines.append("> `lsa` (TF-IDF + SVD), not an open-weight embedding model. Install")
        lines.append("> sentence-transformers, set `retrieval.embeddings.backend: local` and rerun")
        lines.append("> `make rag-eval` before quoting them.")
        lines.append("")

    # ------------------------------------------------------------- B5 table
    lines.append("## B5 variant comparison")
    lines.append("")
    header = (
        "| Variant | Evidence Recall@5 | nDCG@10 | Unsupported claims / proposal "
        "| Judge score | GigSim reply rate | p95 latency (s) | Cost / proposal (USD) |"
    )
    lines.append(header)
    lines.append("|---|---|---|---|---|---|---|---|")

    variants = (rag or {}).get("variants") or []
    if not variants:
        missing.append("rag_eval.json")
        lines.append("| *no retrieval results yet* | - | - | - | - | - | - | - |")
    else:
        claims_by_variant = {
            row.get("variant"): row for row in ((claims or {}).get("variants") or [])
        }
        outcomes_by_variant = {
            row.get("variant"): row for row in ((outcomes or {}).get("variants") or [])
        }
        for row in variants:
            vid = row["variant"]
            claim_row = claims_by_variant.get(vid) or {}
            outcome_row = outcomes_by_variant.get(vid) or {}
            lines.append(
                f"| {vid} "
                f"| {fmt(row.get('recall_at_5'))} "
                f"| {fmt(row.get('ndcg_at_10'))} "
                f"| {fmt(claim_row.get('unsupported_per_proposal'), '.2f')} "
                f"| {fmt((judge or {}).get('mean_score') if vid == (judge or {}).get('variant') else None, '.2f')} "
                f"| {fmt(outcome_row.get('reply_rate'), '.1f', 100) if outcome_row.get('reply_rate') is not None else '-'} "
                f"| {fmt((row.get('latency_ms_p95') or 0) / 1000, '.2f')} "
                f"| {fmt(outcome_row.get('usd_per_proposal'), '.4f')} |"
            )
        if claims and not (claims.get("variants") or []):
            lines.append("")
            lines.append("*The unsupported-claims column is per retrieval variant and is "
                         "empty on purpose: `make verify-eval` measures the verifier "
                         "against labelled claims with fixed evidence, which does not vary "
                         "by variant. Its numbers are in **Claim verification** below. "
                         "Filling this column needs model-written drafts per variant.*")
        for name, key in (
            ("claim_verification.json", claims),
            ("judge_calibration.json", judge),
            ("gigsim_outcomes.json", outcomes),
        ):
            if not key:
                missing.append(name)

    # ------------------------------------------------- retrieval detail
    if variants:
        baseline = next((v for v in variants if v["variant"] == "V0"), None)
        lines.append("")
        lines.append("### Retrieval detail")
        lines.append("")
        lines.append("| Variant | Recall@5 | P@5 | MRR | nDCG@10 | vs V0 (R@5 pts) | 4+ reqs R@5 |")
        lines.append("|---|---|---|---|---|---|---|")
        for row in variants:
            delta = (
                f"{(row['recall_at_5'] - baseline['recall_at_5']) * 100:+.1f}"
                if baseline and row["variant"] != "V0"
                else "-"
            )
            many = (row.get("by_requirement_count") or {}).get("four_or_more") or {}
            lines.append(
                f"| {row['variant']} | {fmt(row.get('recall_at_5'))} "
                f"| {fmt(row.get('precision_at_5'))} | {fmt(row.get('mrr'))} "
                f"| {fmt(row.get('ndcg_at_10'))} | {delta} "
                f"| {fmt(many.get('recall_at_5'))} |"
            )
        lines.append("")
        lines.append(f"*{rag.get('dev_rows')} dev rows with gold evidence; "
                     f"embeddings backend `{(rag.get('setup') or {}).get('embeddings_backend')}`.*")

    # ------------------------------------------------------------- decisions
    lines.append("")
    lines.append("## Decision metrics")
    lines.append("")
    if decision:
        bos = decision["bid_or_skip"]
        scam = decision["scam_detection"]
        price = decision["pricing"]
        inj = decision["injection_defence"]
        lines.append("| Metric | Value |")
        lines.append("|---|---|")
        lines.append(f"| bid-or-skip accuracy | {fmt(bos['accuracy'])} |")
        lines.append(f"| bid-or-skip macro F1 | {fmt(bos['macro_f1'])} |")
        lines.append(f"| bid class F1 | {fmt(bos['bid_class']['f1'])} |")
        lines.append(f"| skip class F1 | {fmt(bos['skip_class']['f1'])} |")
        lines.append(f"| scam precision | {fmt(scam['precision'])} |")
        lines.append(f"| scam recall | {fmt(scam['recall'])} |")
        lines.append(f"| price inside fair band | {fmt(price['inside_fair_band'])} |")
        lines.append(f"| injection detection rate | {fmt(inj['detection_rate'])} |")
        lines.append(f"| **injection attack success rate** | {fmt(inj['attack_success_rate'])} |")
        lines.append(f"| injection false positives | {inj['false_positives_on_clean_jobs']} |")
        lines.append("")
        lines.append(f"*{decision['rows']} dev jobs, retrieval variant "
                     f"{decision['variant']}.*")
    else:
        missing.append("decision_eval.json")
        lines.append("*no decision results yet - run `make decision-eval`.*")

    # ------------------------------------------------------------- ablation
    lines.append("")
    lines.append("## Ablation (R&D question B)")
    lines.append("")
    if ablation:
        lines.append(f"Pipeline: `{ablation['pipeline']}`")
        lines.append("")
        lines.append("| Arm | Removed | Recall@5 | nDCG@10 | MRR |")
        lines.append("|---|---|---|---|---|")
        for arm in ablation["arms"]:
            lines.append(
                f"| `{arm['arm']}` | {arm['removed'] or 'nothing (reference)'} "
                f"| {fmt(arm.get('recall_at_5'))} | {fmt(arm.get('ndcg_at_10'))} "
                f"| {fmt(arm.get('mrr'))} |"
            )
        lines.append("")
        lines.append("Components ranked by Recall@5 points lost when removed:")
        lines.append("")
        for contribution in ablation.get("contributions_ranked") or []:
            lines.append(
                f"- **{contribution['recall_cost']:+.1f}** {contribution['component']}"
            )
    else:
        missing.append("ablation.json")
        lines.append("*no ablation yet - run `make ablation`.*")

    # ------------------------------------------------------ judge calibration
    lines.append("")
    lines.append("## Judge calibration")
    lines.append("")
    if judge:
        bid = judge.get("bid_skip") or {}
        lines.append(f"- judge model: `{judge.get('judge_model')}`")
        lines.append(f"- hand-labelled rows: {judge.get('hand_labelled_rows')}")
        lines.append(f"- Cohen's kappa (bid/skip): **{bid.get('kappa')}** "
                     f"({bid.get('interpretation')})")
        lines.append(f"- raw agreement: {fmt(bid.get('observed_agreement'))}")
        lines.append(f"- usable for reporting: "
                     f"**{'yes' if judge.get('judge_usable_for_reporting') else 'no'}**")
    else:
        missing.append("judge_calibration.json")
        lines.append("*Not calibrated yet.* Fill in the `human_*` columns of")
        lines.append("`evals/datasets/hand_check_sample.csv`, then run `make judge-calibration`.")
        lines.append("Until then no judge score may be reported: B4 requires a calibrated judge.")

    # --------------------------------------------------- R&D question A
    lines.append("")
    lines.append("## R&D question A: tool granularity")
    lines.append("")
    if tools:
        lines.append(f"Model `{tools.get('model')}`, {tools.get('task_count')} tasks.")
        lines.append("")
        lines.append("| Arm | Tools | Tool-selection accuracy | Argument accuracy | Tokens/task | Failed |")
        lines.append("|---|---|---|---|---|---|")
        for arm in tools.get("arms") or []:
            lines.append(
                f"| {arm['arm']} | {arm['tool_count']} "
                f"| {fmt(arm.get('tool_selection_accuracy'))} "
                f"| {fmt(arm.get('argument_accuracy'))} "
                f"| {fmt(arm.get('mean_tokens_per_task'), '.0f')} "
                f"| {arm.get('tasks_errored', 0)} |"
            )
        worst = max((a.get("tasks_errored", 0) for a in tools.get("arms") or []), default=0)
        if worst:
            lines.append("")
            lines.append(f"> {worst} request(s) failed and were excluded from the accuracy "
                         "figures. Treat as provisional until a clean run.")
    else:
        missing.append("rnd_a_tool_granularity.json")
        lines.append("*not run yet - `make tool-study` (needs a provider key).*")

    # --------------------------------------------------- R&D question C
    lines.append("")
    lines.append("## R&D question C: is the multi-agent design worth it?")
    lines.append("")
    if comparison:
        lines.append(f"Model `{comparison.get('model')}`, {comparison.get('rows')} jobs.")
        lines.append("")
        lines.append("| Arm | bid/skip acc | bid F1 | scam recall | unsupported/proposal "
                     "| tokens/job | ms/job |")
        lines.append("|---|---|---|---|---|---|---|")
        for arm in comparison.get("arms") or []:
            lines.append(
                f"| {arm['arm']} | {fmt(arm.get('bid_or_skip_accuracy'))} "
                f"| {fmt(arm.get('bid_f1'))} | {fmt(arm.get('scam_recall'))} "
                f"| {fmt(arm.get('unsupported_claims_per_proposal'), '.2f')} "
                f"| {fmt(arm.get('mean_tokens'), '.0f')} "
                f"| {fmt(arm.get('mean_latency_ms'), '.0f')} |"
            )
        if int(comparison.get("rows") or 0) < 50:
            lines.append("")
            lines.append(f"> Only {comparison.get('rows')} jobs, not the 50 the brief asks "
                         "for. The provider meters tokens per day per model and three arms "
                         "over 50 jobs needs roughly 775k; this is a verification run, not "
                         "the result. Rerun `make agent-comparison` with quota.")
    else:
        missing.append("rnd_c_agent_comparison.json")
        lines.append("*not run yet - `make agent-comparison` (needs a provider key).*")

    # --------------------------------------------------- C6 durability
    lines.append("")
    lines.append("## C6: exactly-once across a restart")
    lines.append("")
    if durability:
        verdict = "PASSED" if durability.get("passed") else "FAILED"
        lines.append(f"**{verdict}** - job `{durability.get('job_id')}`, "
                     f"{durability.get('proposals_created')} proposal created, "
                     f"{durability.get('connects_spent')} Connects spent "
                     f"(job costs {durability.get('connects_cost')}).")
        lines.append("")
        for label, passed in (durability.get("checks") or {}).items():
            lines.append(f"- {'PASS' if passed else 'FAIL'} {label}")
        lines.append("")
        lines.append("Three separate OS processes: run to the interrupt, exit, approve from "
                     "a new process, then retry from a third.")
    else:
        missing.append("c6_crash_resume.json")
        lines.append("*not run yet - `make gigsim` then `make crash-resume`.*")

    # --------------------------------------------------- B3 claim verification
    lines.append("")
    lines.append("## Claim verification (B3)")
    lines.append("")
    if claims:
        summary = claims.get("summary") or {}
        lines.append("| Metric | Value |")
        lines.append("|---|---|")
        lines.append(f"| fabrications caught | {summary.get('fabrications_caught')}"
                     f"/{summary.get('fabrications')} |")
        lines.append(f"| fabrication recall | {fmt(summary.get('fabrication_recall'))} |")
        lines.append(f"| flag precision | {fmt(summary.get('flag_precision'))} |")
        lines.append(f"| F1 | {fmt(summary.get('f1'))} |")
        lines.append(f"| supported claims wrongly flagged | "
                     f"{summary.get('supported_wrongly_flagged')}"
                     f"/{summary.get('supported_claims')} |")
        lines.append(f"| unsupported per proposal, before stripping | "
                     f"{fmt(summary.get('unsupported_per_proposal_before'), '.2f')} |")
        lines.append(f"| unsupported per proposal, after stripping | "
                     f"{fmt(summary.get('unsupported_per_proposal_after'), '.2f')} |")
        lines.append(f"| p95 verification latency (ms) | "
                     f"{fmt(summary.get('p95_latency_ms'), '.1f')} |")
        by_kind = summary.get("by_fabrication_kind") or {}
        if by_kind:
            lines.append("")
            lines.append("Caught, by kind of fabrication:")
            lines.append("")
            for kind, bucket in sorted(by_kind.items()):
                lines.append(f"- {kind}: {bucket.get('caught')}/{bucket.get('n')}")
        lines.append("")
        lines.append(f"*{summary.get('probes')} probe proposals, "
                     f"{summary.get('sentences')} labelled sentences, generated from the "
                     f"real corpora.*")
    else:
        missing.append("claim_verification.json")
        lines.append("*not run yet - `make verify-eval`.*")

    # --------------------------------------------------- E5 budgets
    lines.append("")
    lines.append("## Declared budgets against measurements (E5)")
    lines.append("")
    if budgets:
        lines.append("| Metric | Declared | Measured | Status |")
        lines.append("|---|---|---|---|")
        for row in budgets.get("checks") or []:
            declared = "-" if row.get("declared") is None else row["declared"]
            measured = "-" if row.get("measured") is None else row["measured"]
            lines.append(f"| {row['metric']} | {declared} | {measured} | {row['status']} |")
        lines.append("")
        lines.append(f"*{budgets.get('runs_counted')} recorded run(s); runs made with the "
                     f"model offline are excluded, since their latency and cost are not "
                     f"real.*")
    else:
        missing.append("budget_report.json")
        lines.append("*not run yet - `make budget-report`.*")

    # --------------------------------------------------- X1 extension
    lines.append("")
    lines.append("## X1 agency mode (signature extension)")
    lines.append("")
    if agency:
        summary = agency.get("summary") or {}
        lines.append("| Metric | Value |")
        lines.append("|---|---|")
        lines.append(f"| jobs routed | {summary.get('routed')}/{summary.get('jobs')} |")
        lines.append(f"| declined to route | {summary.get('declined_to_route')} |")
        lines.append(f"| accuracy when committed | "
                     f"{fmt(summary.get('routing_accuracy_when_committed'))} |")
        lines.append(f"| accuracy over all jobs | "
                     f"{fmt(summary.get('routing_accuracy_over_all_jobs'))} |")
        lines.append(f"| mean coverage, routed | "
                     f"{fmt(summary.get('mean_coverage_routed'))} |")
        lines.append(f"| mean coverage, best fixed member | "
                     f"{fmt(summary.get('mean_coverage_best_fixed'))} "
                     f"({summary.get('best_fixed_member')}) |")
        lines.append(f"| coverage gained (points) | "
                     f"{fmt(summary.get('coverage_points_gained'), '+.2f')} |")
        lines.append(f"| cost | {summary.get('retrieval_passes_per_job')} retrieval passes "
                     f"per job, {fmt(summary.get('seconds_per_job'), '.2f')} s/job |")
        lines.append("")
        lines.append("See `docs/EXTENSION.md` for the design and what the numbers mean.")
    else:
        missing.append("x1_agency_routing.json")
        lines.append("*not run yet - `make agency-eval`.*")

    # ------------------------------------------------------------- footnotes
    if missing:
        lines.append("")
        lines.append("## Missing results")
        lines.append("")
        for name in dict.fromkeys(missing):
            what, command = SOURCES.get(name, ("?", "?"))
            lines.append(f"- `{name}` ({what}) - run `{command}`")

    lines.append("")
    return "\n".join(lines)


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    table = build()
    out = RESULTS / "results_table.md"
    out.write_text(table, encoding="utf-8")
    print(table)
    print(f"\nwritten: {out}")


if __name__ == "__main__":
    main()
