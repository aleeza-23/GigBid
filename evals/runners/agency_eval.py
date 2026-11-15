"""X1: does routing a job to the right team member beat a single freelancer?

The dev set labels every job with the persona it was written for, so routing can be scored
directly: route each job by portfolio coverage, ignore the label while deciding, then
compare the choice with the label.

Three numbers, because routing accuracy alone would not say whether agency mode is worth
running:

* **Routing accuracy** against the labelled persona, over the jobs where the router
  committed to a member. Jobs it declined to route are reported separately rather than
  counted as wrong, since declining and being wrong are different behaviours and only one
  of them puts a bad proposal in front of a client.
* **Coverage gained.** The coverage the routed member reaches against the coverage a fixed
  single freelancer reaches on the same job. This is the number that decides whether the
  extension earns its place: if one member covers everything anyway, routing is overhead.
* **Cost.** Retrieval passes per job, which is the whole price of routing. There is no
  extra model call.

The comparison is deliberately unkind to agency mode: the single-freelancer baseline is
the member who scores best over the *whole* dev set, not a random one, so routing has to
beat the best fixed choice rather than an average one.

    make agency-eval
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "evals" / "results"
DEV_SET = ROOT / "evals" / "datasets" / "dev_set.jsonl"
JOBS = ROOT / "data" / "gigsim_seed" / "jobs_stream.json"


def load_rows(limit: int = 0) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in DEV_SET.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return rows[:limit] if limit else rows


def load_jobs() -> dict[str, dict[str, Any]]:
    payload = json.loads(JOBS.read_text(encoding="utf-8"))
    rows = payload["jobs"] if isinstance(payload, dict) else payload
    return {row["job_id"]: row for row in rows}


def main() -> None:
    parser = argparse.ArgumentParser(description="X1 agency-mode routing evaluation")
    parser.add_argument("--limit", type=int, default=60, help="dev jobs to route")
    parser.add_argument("--members", nargs="+", default=None)
    args = parser.parse_args()

    from agent.nodes.agency import agency_config, choose, score_members
    from rag.retriever import describe as retrieval_describe

    settings = agency_config()
    members = args.members or [str(m) for m in settings["members"]]
    min_coverage = float(settings["min_coverage"])
    min_margin = float(settings["min_margin"])

    rows = load_rows(args.limit)
    jobs = load_jobs()

    per_job: list[dict[str, Any]] = []
    started = time.perf_counter()

    for row in rows:
        job = dict(jobs.get(row["job_id"]) or {})
        if not job:
            continue
        label = row["persona"]

        scored = score_members(job, members)
        if not scored:
            continue
        by_member = {entry["member"]: entry for entry in scored}
        verdict = choose(
            scored,
            default_member=members[0],
            min_coverage=min_coverage,
            min_margin=min_margin,
        )

        per_job.append(
            {
                "job_id": row["job_id"],
                "job_kind": row.get("job_kind"),
                "labelled_persona": label,
                "routed": verdict["routed"],
                "chosen": verdict["member"],
                "why": verdict["reason"],
                "correct": bool(verdict["routed"]) and verdict["member"] == label,
                "coverage_chosen": by_member.get(verdict["member"], {}).get("coverage", 0.0),
                "coverage_by_member": {
                    entry["member"]: entry["coverage"] for entry in scored
                },
            }
        )

    elapsed = time.perf_counter() - started
    if not per_job:
        raise SystemExit("no jobs could be routed; run `make seed` and `make ingest` first")

    committed = [r for r in per_job if r["routed"]]
    declined = [r for r in per_job if not r["routed"]]
    correct = [r for r in committed if r["correct"]]

    # The baseline: the single member with the best mean coverage over every job, which is
    # the best a one-freelancer setup could do by hiring the right person once.
    mean_by_member = {
        member: round(
            statistics.mean([r["coverage_by_member"].get(member, 0.0) for r in per_job]), 4
        )
        for member in members
    }
    best_fixed = max(mean_by_member, key=lambda m: mean_by_member[m])

    routed_coverage = statistics.mean([r["coverage_chosen"] for r in per_job])
    fixed_coverage = mean_by_member[best_fixed]

    # Where the mistakes go: a confusion count over the committed decisions.
    confusion = Counter(
        (r["labelled_persona"], r["chosen"]) for r in committed if not r["correct"]
    )

    summary = {
        "jobs": len(per_job),
        "members": members,
        "retrieval": retrieval_describe().get("configured"),
        "min_coverage": min_coverage,
        "min_margin": min_margin,
        "routed": len(committed),
        "declined_to_route": len(declined),
        "routing_accuracy_when_committed": round(len(correct) / len(committed), 4)
        if committed
        else 0.0,
        "routing_accuracy_over_all_jobs": round(len(correct) / len(per_job), 4),
        "mean_coverage_routed": round(routed_coverage, 4),
        "best_fixed_member": best_fixed,
        "mean_coverage_best_fixed": fixed_coverage,
        "coverage_points_gained": round((routed_coverage - fixed_coverage) * 100, 2),
        "mean_coverage_per_member": mean_by_member,
        "retrieval_passes_per_job": len(members),
        "seconds_per_job": round(elapsed / len(per_job), 3),
        "confusions": [
            {"labelled": label, "chosen": chosen, "n": n}
            for (label, chosen), n in confusion.most_common(8)
        ],
    }

    print(f"X1 agency routing over {summary['jobs']} dev jobs")
    print(f"members: {', '.join(members)}")
    print(f"retrieval: {summary['retrieval']}")
    print()
    print(f"routed                     {summary['routed']}/{summary['jobs']}"
          f"   declined {summary['declined_to_route']}")
    print(f"accuracy when committed    {summary['routing_accuracy_when_committed']:.3f}")
    print(f"accuracy over all jobs     {summary['routing_accuracy_over_all_jobs']:.3f}")
    print()
    print(f"mean coverage, routed      {summary['mean_coverage_routed']:.3f}")
    print(f"mean coverage, best fixed  {summary['mean_coverage_best_fixed']:.3f}"
          f"  ({best_fixed})")
    print(f"coverage gained            {summary['coverage_points_gained']:+.2f} points")
    print()
    print("mean coverage per member:")
    for member, value in sorted(mean_by_member.items(), key=lambda kv: -kv[1]):
        print(f"  {member:26} {value:.3f}")
    if summary["confusions"]:
        print()
        print("wrong choices (labelled -> chosen):")
        for entry in summary["confusions"]:
            print(f"  {entry['labelled']:26} -> {entry['chosen']:26} {entry['n']}")
    print()
    print(f"cost: {summary['retrieval_passes_per_job']} retrieval passes per job, "
          f"{summary['seconds_per_job']:.2f} s/job, no extra model call")

    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / "x1_agency_routing.json"
    out.write_text(
        json.dumps({"summary": summary, "per_job": per_job}, indent=2), encoding="utf-8"
    )
    print(f"\nwritten: {out}")


if __name__ == "__main__":
    main()
