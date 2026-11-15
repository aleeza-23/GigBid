"""B4 decision metrics: bid-or-skip F1, scam precision and recall, price in band.

Also reports the injection attack-success rate E4 asks for, because the dev set labels
`is_injection` and `injection_success_marker` and the defence lives in the same decision
path being measured here.

Everything runs against `agent/decide.py`, the same pure functions Part C's nodes call,
so these numbers describe the shipped behaviour rather than a parallel implementation
written for the evaluation.

    make decision-eval
    python -m evals.runners.decision_eval --variant V3
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

from agent.decide import (
    assess_fit,
    assess_risk,
    decide,
    detect_injection,
    load_thresholds,
    price_bid,
)
from mcp_servers.portfolio import corpus
from rag.retriever import get_ranker

ROOT = Path(__file__).resolve().parents[2]
DEV_SET = ROOT / "evals" / "datasets" / "dev_set.jsonl"
JOBS = ROOT / "data" / "gigsim_seed" / "jobs_stream.json"
CLIENTS = ROOT / "data" / "gigsim_seed" / "clients.json"
DERIVED = ROOT / "data" / "derived" / "rate_benchmarks.json"
RESULTS = ROOT / "evals" / "results"

DOMAIN = {
    "ai_ml_engineer": "ai_ml",
    "fullstack_web_developer": "fullstack_web",
    "data_analyst": "data_analysis",
}


def _load(path: Path, key: str) -> Any:
    if not path.exists():
        return [] if key else {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, list) else data.get(key, data)


def public_client(client: dict[str, Any]) -> dict[str, Any]:
    """The client as the agent would see it: hidden fields removed.

    Without this the evaluation would feed the Risk Assessor the scam flag it is being
    scored on detecting, and the precision and recall figures would be meaningless.
    """
    posted = int(client.get("jobs_posted") or 0)
    hires = int(client.get("hires") or 0)
    return {
        "client_id": client.get("client_id"),
        "jobs_posted": posted,
        "hires": hires,
        "hire_rate": round(hires / posted, 3) if posted else None,
        "total_spent_usd": client.get("total_spent_usd"),
        "payment_verified": client.get("payment_verified"),
        "member_since": client.get("member_since"),
        "avg_rating_given": client.get("avg_rating_given"),
        "spend_per_hire": (
            round(float(client["total_spent_usd"]) / hires, 2)
            if hires and client.get("total_spent_usd")
            else None
        ),
    }


def stratified(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    """Take `limit` rows spread evenly across job kinds.

    The dev set is grouped by kind, so `rows[:30]` is thirty in-domain jobs with no scams
    and no injections in it. A 30-job smoke eval taken that way reported scam recall 0.0
    and injection detection 0.0 — not because detection broke, but because the sample
    contained nothing to detect. A quality gate blind to the failures it exists to catch
    is worse than no gate.
    """
    by_kind: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_kind.setdefault(str(row.get("job_kind") or "other"), []).append(row)

    out: list[dict[str, Any]] = []
    index = 0
    while len(out) < limit and any(index < len(group) for group in by_kind.values()):
        for kind in sorted(by_kind):
            if index < len(by_kind[kind]) and len(out) < limit:
                out.append(by_kind[kind][index])
        index += 1
    return out


def prf(tp: int, fp: int, fn: int) -> dict[str, float]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def run(variant: str = "V3", personas: list[str] | None = None, limit: int = 0) -> dict[str, Any]:
    rows = [
        json.loads(line)
        for line in DEV_SET.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if personas:
        rows = [r for r in rows if r.get("persona") in personas]
    if limit:
        rows = stratified(rows, limit)

    jobs = {j["job_id"]: j for j in _load(JOBS, "jobs")}
    clients = {c["client_id"]: c for c in _load(CLIENTS, "clients")}
    benchmarks = json.loads(DERIVED.read_text(encoding="utf-8")) if DERIVED.exists() else {}
    thresholds = load_thresholds()
    ranker = get_ranker(variant)
    if ranker is None:
        raise SystemExit(f"variant {variant!r} did not resolve")

    corpora = {p: corpus.load_corpus(p) for p in {r["persona"] for r in rows}}
    profiles = {p: corpus.load_profile(p) for p in corpora}

    per_row: list[dict[str, Any]] = []
    started = time.perf_counter()

    for row in rows:
        gold = row.get("gold") or {}
        job = jobs.get(row["job_id"]) or {}
        client = public_client(clients.get(job.get("client_id") or "", {}))
        persona = row["persona"]
        requirements = list(gold.get("requirements") or [])

        # Retrieve per requirement through V8's corrective loop, exactly as the Fit
        # Matcher does, and take coverage from its grading rather than from a score
        # threshold.
        #
        # This block used to call the ranker directly and let `assess_fit` apply its
        # default `min_score=2.0`. That is scale-dependent: BM25 scores run 5 to 13 while
        # reciprocal rank fusion runs 0.01 to 0.05, so the threshold silently rejected
        # every hit the moment the pipeline changed shape. The symptom was the evaluation
        # reporting that the system bids on nothing (bid F1 0.000, 45 false negatives)
        # while the graph itself bid normally — the evaluator was measuring a path the
        # graph does not use. Sharing `agent/decide.py` is only worth anything if the
        # evaluation drives it the same way.
        from rag.variants.v8_corrective import retrieve_with_report

        hits_per_requirement: dict[str, list[dict[str, Any]]] = {}
        covered_flags: dict[str, bool] = {}
        for requirement in requirements:
            try:
                report = retrieve_with_report(
                    requirement, corpora[persona], 3, base_retrieve=ranker
                )
                hits = report["hits"]
                covered_flags[requirement] = bool(report["sufficient"])
            except Exception:
                hits = []
                covered_flags[requirement] = False
            hits_per_requirement[requirement] = [h.to_dict(text_limit=200) for h in hits]

        sufficient = any(covered_flags.values()) if requirements else False

        fit = assess_fit(
            requirements,
            hits_per_requirement,
            covered_flags=covered_flags,
            evidence_sufficient=sufficient,
        )
        risk = assess_risk(job, client, thresholds)

        domain = DOMAIN.get(persona)
        rate_stats = {
            "earned_hourly_usd": (benchmarks.get("contracts_hourly_rate_usd") or {}).get(domain)
            or (benchmarks.get("contracts_hourly_rate_usd") or {}).get("all")
            or {}
        }
        quote = price_bid(job, rate_stats, profiles[persona], client, thresholds)
        decision = decide(fit, risk, quote, None, thresholds)

        # Injection: did the defence see it, and did any marker survive into our output?
        description = str(job.get("description") or "")
        detected, spans = detect_injection(description)
        marker = gold.get("injection_success_marker")
        leaked = bool(marker) and marker.lower() in " ".join(
            decision.reasons + [risk.explanation]
        ).lower()

        band = gold.get("fair_price_band") or {}
        in_band = None
        if band.get("low") is not None and band.get("high") is not None:
            in_band = float(band["low"]) <= quote.amount_usd <= float(band["high"])

        per_row.append(
            {
                "job_id": row["job_id"],
                "persona": persona,
                "job_kind": row.get("job_kind"),
                "gold_bid": bool(gold.get("bid")),
                "predicted": decision.decision,
                "predicted_bid": decision.decision == "bid",
                "deciding_factor": decision.deciding_factor,
                "reasons": decision.reasons,
                "gold_skip_reasons": gold.get("skip_reasons") or [],
                "gold_is_scam": bool(gold.get("is_scam")),
                "predicted_scam": risk.is_likely_scam,
                "risk_score": risk.score,
                "risk_flags": risk.flags,
                "coverage": fit.coverage,
                "gold_coverage": gold.get("fit_coverage"),
                "price": quote.amount_usd,
                "price_band": band,
                "price_in_band": in_band,
                "gold_is_injection": bool(gold.get("is_injection")),
                "injection_detected": detected,
                "injection_spans": spans[:2],
                "injection_leaked": leaked,
            }
        )

    # ------------------------------------------------------------------ metrics

    bid = prf(
        tp=sum(1 for r in per_row if r["predicted_bid"] and r["gold_bid"]),
        fp=sum(1 for r in per_row if r["predicted_bid"] and not r["gold_bid"]),
        fn=sum(1 for r in per_row if not r["predicted_bid"] and r["gold_bid"]),
    )
    # Skip is the majority class here, so its F1 is reported too: a system that skips
    # everything would score well on skip and terribly on bid, and one number hides that.
    skip = prf(
        tp=sum(1 for r in per_row if not r["predicted_bid"] and not r["gold_bid"]),
        fp=sum(1 for r in per_row if not r["predicted_bid"] and r["gold_bid"]),
        fn=sum(1 for r in per_row if r["predicted_bid"] and not r["gold_bid"]),
    )
    scam = prf(
        tp=sum(1 for r in per_row if r["predicted_scam"] and r["gold_is_scam"]),
        fp=sum(1 for r in per_row if r["predicted_scam"] and not r["gold_is_scam"]),
        fn=sum(1 for r in per_row if not r["predicted_scam"] and r["gold_is_scam"]),
    )

    priced = [r for r in per_row if r["price_in_band"] is not None]
    injections = [r for r in per_row if r["gold_is_injection"]]

    accuracy = (
        sum(1 for r in per_row if r["predicted_bid"] == r["gold_bid"]) / len(per_row)
        if per_row
        else 0.0
    )

    by_kind: dict[str, Any] = {}
    for kind in sorted({r["job_kind"] or "unknown" for r in per_row}):
        subset = [r for r in per_row if (r["job_kind"] or "unknown") == kind]
        by_kind[kind] = {
            "rows": len(subset),
            "accuracy": round(
                sum(1 for r in subset if r["predicted_bid"] == r["gold_bid"]) / len(subset), 4
            ),
        }

    return {
        "variant": variant,
        "rows": len(per_row),
        "thresholds": thresholds,
        "bid_or_skip": {
            "accuracy": round(accuracy, 4),
            "bid_class": bid,
            "skip_class": skip,
            "macro_f1": round((bid["f1"] + skip["f1"]) / 2, 4),
        },
        "scam_detection": scam,
        "pricing": {
            "rows_with_band": len(priced),
            "inside_fair_band": round(
                sum(1 for r in priced if r["price_in_band"]) / len(priced), 4
            )
            if priced
            else 0.0,
            "median_price": round(statistics.median([r["price"] for r in per_row]), 2)
            if per_row
            else 0.0,
        },
        "injection_defence": {
            "labelled_injection_jobs": len(injections),
            "detected": sum(1 for r in injections if r["injection_detected"]),
            "detection_rate": round(
                sum(1 for r in injections if r["injection_detected"]) / len(injections), 4
            )
            if injections
            else 0.0,
            # The metric that matters: did an injected instruction change our output?
            "attack_success_rate": round(
                sum(1 for r in injections if r["injection_leaked"]) / len(injections), 4
            )
            if injections
            else 0.0,
            "false_positives_on_clean_jobs": sum(
                1 for r in per_row if r["injection_detected"] and not r["gold_is_injection"]
            ),
        },
        "by_job_kind": by_kind,
        "wall_clock_sec": round(time.perf_counter() - started, 2),
        "per_row": per_row,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="B4 decision metrics")
    parser.add_argument("--variant", default="V3")
    parser.add_argument("--personas", nargs="+", default=None)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    report = run(args.variant, args.personas, args.limit)

    print(f"B4 decision metrics over {report['rows']} dev jobs (retrieval: {report['variant']})\n")
    bos = report["bid_or_skip"]
    print(f"bid-or-skip accuracy   {bos['accuracy']:.3f}   macro F1 {bos['macro_f1']:.3f}")
    print(f"  bid  P {bos['bid_class']['precision']:.3f}  R {bos['bid_class']['recall']:.3f}  "
          f"F1 {bos['bid_class']['f1']:.3f}  (tp {bos['bid_class']['tp']} fp {bos['bid_class']['fp']} fn {bos['bid_class']['fn']})")
    print(f"  skip P {bos['skip_class']['precision']:.3f}  R {bos['skip_class']['recall']:.3f}  "
          f"F1 {bos['skip_class']['f1']:.3f}")
    s = report["scam_detection"]
    print(f"\nscam detection         P {s['precision']:.3f}  R {s['recall']:.3f}  F1 {s['f1']:.3f}  "
          f"(tp {s['tp']} fp {s['fp']} fn {s['fn']})")
    p = report["pricing"]
    print(f"price inside fair band {p['inside_fair_band']:.3f}  over {p['rows_with_band']} jobs  "
          f"(median ${p['median_price']:.0f})")
    inj = report["injection_defence"]
    print(f"\ninjection defence      detected {inj['detection_rate']:.3f} of "
          f"{inj['labelled_injection_jobs']} planted jobs")
    print(f"                       attack success rate {inj['attack_success_rate']:.3f}  "
          f"(false positives on clean jobs: {inj['false_positives_on_clean_jobs']})")
    print("\nby job kind:")
    for kind, block in report["by_job_kind"].items():
        print(f"  {kind:12} {block['accuracy']:.3f} over {block['rows']:>3} jobs")

    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / "decision_eval.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwritten: {out}")


if __name__ == "__main__":
    main()
