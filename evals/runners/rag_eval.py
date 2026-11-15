"""B4 retrieval metrics: Recall@5, MRR and nDCG@10 per variant, per persona.

This is the file that makes R6 true: the deployed retrieval design is chosen by these
numbers rather than by preference. It writes `evals/results/rag_eval.json`, which every
retrieval figure in the README and the report must cite.

**How gold labels map onto retrieved items.** The dev set grades *projects*
(`ai-p10`, relevance 3), while retrieval returns *evidence items* (`ai-p10-case`,
`ai-p10-readme`, `ai-p10-code`). An item is relevant when its `project_id` is a gold
project, and it inherits that project's grade. That is the right mapping for this corpus:
the same project is described four ways on purpose, so surfacing the README instead of the
case study is a different presentation of the same correct answer, not a miss.

Credit is counted **once per project**. Without that, a variant returning four views of
one gold project would score a perfect Recall@5 while having found a single piece of
evidence, which is precisely the failure V5 exists to address and would be invisible here.

    make rag-eval
    python -m evals.runners.rag_eval --variants V0 V2 V3 --personas ai_ml_engineer
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any

from mcp_servers.portfolio.corpus import VALID_PERSONAS, EvidenceItem, load_corpus
from rag.retriever import describe, get_ranker, list_variants

ROOT = Path(__file__).resolve().parents[2]
DEV_SET = ROOT / "evals" / "datasets" / "dev_set.jsonl"
JOBS = ROOT / "data" / "gigsim_seed" / "jobs_stream.json"
RESULTS = ROOT / "evals" / "results"


def load_jobs() -> dict[str, dict[str, Any]]:
    if not JOBS.exists():
        return {}
    rows = json.loads(JOBS.read_text(encoding="utf-8"))
    rows = rows if isinstance(rows, list) else rows.get("jobs", [])
    return {r["job_id"]: r for r in rows}


def load_dev_set(personas: list[str] | None = None) -> list[dict[str, Any]]:
    """Dev rows that carry gold evidence, which is what retrieval can be scored against."""
    if not DEV_SET.exists():
        raise SystemExit(f"missing {DEV_SET}. Run `make seed` first.")
    rows = [json.loads(line) for line in DEV_SET.read_text(encoding="utf-8").splitlines() if line.strip()]
    kept = [r for r in rows if (r.get("gold") or {}).get("gold_evidence_ids")]
    if personas:
        kept = [r for r in kept if r.get("persona") in personas]
    return kept


# ----------------------------------------------------------------------- metrics


def _gold_grades(row: dict[str, Any]) -> dict[str, int]:
    """Gold project id -> graded relevance (1 to 3).

    `relevant_evidence` carries the grades; `gold_evidence_ids` is the unweighted list.
    Prefer the graded form, since nDCG needs it, and fall back to a flat grade of 1.
    """
    gold = row.get("gold") or {}
    graded = {
        str(entry["project_id"]): int(entry.get("relevance") or 1)
        for entry in (gold.get("relevant_evidence") or [])
        if entry.get("project_id")
    }
    if graded:
        return graded
    return {str(pid): 1 for pid in (gold.get("gold_evidence_ids") or [])}


def _attributed_projects(hit: EvidenceItem) -> frozenset[str]:
    """Which portfolio projects this hit is evidence *about*.

    Most of the corpus is past proposals: 150 of the ~197 items per persona. A proposal
    has no `project_id` of its own; it cites the projects it drew on, in
    `cited_project_ids`. Attributing a hit only by `project_id` therefore scored 76% of
    the corpus as a miss no matter how good the retrieval was, and every variant's
    Recall@5 was capped near 0.2 by the metric rather than by the retriever.

    A past proposal citing ai-p07 is evidence that the freelancer did ai-p07's work, which
    is exactly what the Writer needs in order to claim it. So a hit counts for the project
    it belongs to, or, lacking one, for the projects it cites (one to three in this
    corpus, so this is credit for a specific claim and not a wildcard).

    An item with neither -- the CV -- attributes to nothing and occupies a rank without
    scoring, which is the correct treatment: it is not evidence of a particular project.
    """
    if hit.project_id:
        return frozenset({hit.project_id})
    return frozenset(str(pid) for pid in (hit.cited_project_ids or ()) if pid)


def _dedupe_by_project(hits: list[EvidenceItem]) -> list[frozenset[str]]:
    """Ranked attribution sets, with projects already credited removed.

    Four views of one project is one answer, so only its best-ranked view counts. A hit
    whose every project has already been seen is dropped; one that brings at least one new
    project keeps its rank.
    """
    seen: set[str] = set()
    out: list[frozenset[str]] = []
    for hit in hits:
        attributed = _attributed_projects(hit)
        if attributed and not (attributed - seen):
            continue
        seen |= attributed
        out.append(attributed)
    return out


def _best_grade(attributed: frozenset[str], grades: dict[str, int]) -> int:
    """The grade this hit earns: the highest among the projects it is evidence for."""
    return max((grades.get(pid, 0) for pid in attributed), default=0)


def recall_at_k(
    ranked_projects: list[frozenset[str]], grades: dict[str, int], k: int = 5
) -> float:
    if not grades:
        return 0.0
    found: set[str] = set()
    for attributed in ranked_projects[:k]:
        found |= attributed & set(grades)
    return len(found) / len(grades)


def reciprocal_rank(ranked_projects: list[frozenset[str]], grades: dict[str, int]) -> float:
    for rank, attributed in enumerate(ranked_projects, start=1):
        if _best_grade(attributed, grades):
            return 1.0 / rank
    return 0.0


def ndcg_at_k(
    ranked_projects: list[frozenset[str]], grades: dict[str, int], k: int = 10
) -> float:
    """Graded nDCG. Uses 2**rel - 1 gain, so relevance 3 counts well above relevance 1."""
    if not grades:
        return 0.0
    gains = [2 ** _best_grade(a, grades) - 1 for a in ranked_projects[:k]]
    dcg = sum(g / math.log2(i + 2) for i, g in enumerate(gains))
    ideal = sorted((2**g - 1 for g in grades.values()), reverse=True)[:k]
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg else 0.0


def precision_at_k(
    ranked_projects: list[frozenset[str]], grades: dict[str, int], k: int = 5
) -> float:
    window = ranked_projects[:k]
    if not window:
        return 0.0
    return sum(1 for a in window if _best_grade(a, grades)) / len(window)


# -------------------------------------------------------------------- evaluation


def build_query(row: dict[str, Any], jobs: dict[str, dict[str, Any]]) -> tuple[str, dict[str, Any]]:
    """The query a retriever sees: the whole job post, as V0 would use it.

    Returns the text and the job dict, because V1 needs the structured post to decompose
    and the others need only the text.
    """
    job = jobs.get(row["job_id"]) or {}
    text = " ".join(
        filter(
            None,
            [
                str(job.get("title") or row.get("title") or ""),
                str(job.get("description") or "")[:3000],
                " ".join(str(s) for s in (job.get("skills") or [])),
            ],
        )
    )
    if not text.strip():
        # No job text available: fall back to the gold requirement list so the row still
        # scores something rather than silently contributing a zero to every variant.
        text = " ".join((row.get("gold") or {}).get("requirements") or [])
    return text, job


def evaluate_variant(
    variant_id: str,
    rows: list[dict[str, Any]],
    jobs: dict[str, dict[str, Any]],
    *,
    k: int = 5,
    depth: int = 10,
    llm: Any = None,
) -> dict[str, Any]:
    """Score one variant over the dev rows."""
    ranker = get_ranker(variant_id)
    if ranker is None:
        raise SystemExit(f"variant {variant_id!r} did not resolve; check rag/retriever.py")

    corpora = {p: load_corpus(p) for p in {r["persona"] for r in rows}}
    per_row: list[dict[str, Any]] = []
    started = time.perf_counter()

    for row in rows:
        grades = _gold_grades(row)
        query, job = build_query(row, jobs)
        items = corpora[row["persona"]]

        options: dict[str, Any] = {}
        if llm is not None:
            options["llm"] = llm
        # V1 decomposes, so give it the structured post rather than the flattened text.
        if variant_id == "V1":
            options["requirements"] = None
            query_arg: Any = job or query
        else:
            query_arg = query

        call_started = time.perf_counter()
        try:
            hits = list(ranker(query_arg, items, depth, **options))
            failed = None
        except Exception as exc:
            hits, failed = [], f"{type(exc).__name__}: {exc}"
        latency_ms = (time.perf_counter() - call_started) * 1000

        ranked = _dedupe_by_project(hits)
        per_row.append(
            {
                "job_id": row["job_id"],
                "persona": row["persona"],
                "job_kind": row.get("job_kind"),
                "requirement_count": len((row.get("gold") or {}).get("requirements") or []),
                "gold_projects": sorted(grades),
                "retrieved_projects": [sorted(a) for a in ranked[:depth] if a],
                "recall_at_5": recall_at_k(ranked, grades, 5),
                "precision_at_5": precision_at_k(ranked, grades, 5),
                "mrr": reciprocal_rank(ranked, grades),
                "ndcg_at_10": ndcg_at_k(ranked, grades, 10),
                "latency_ms": round(latency_ms, 2),
                "error": failed,
                "sufficient": (hits[0].extra.get("sufficient") if hits else None),
            }
        )

    def mean(key: str, subset: list[dict[str, Any]] | None = None) -> float:
        pool = subset if subset is not None else per_row
        values = [r[key] for r in pool if r["error"] is None]
        return round(statistics.mean(values), 4) if values else 0.0

    # R&D question B's hypothesis is specifically about jobs with four or more
    # requirements, so that split is computed here rather than left to a spreadsheet.
    many = [r for r in per_row if r["requirement_count"] >= 4]
    few = [r for r in per_row if r["requirement_count"] < 4]

    by_persona = {}
    for persona in sorted({r["persona"] for r in per_row}):
        subset = [r for r in per_row if r["persona"] == persona]
        by_persona[persona] = {
            "rows": len(subset),
            "recall_at_5": mean("recall_at_5", subset),
            "mrr": mean("mrr", subset),
            "ndcg_at_10": mean("ndcg_at_10", subset),
        }

    latencies = [r["latency_ms"] for r in per_row if r["error"] is None]
    return {
        "variant": variant_id,
        "rows": len(per_row),
        "errors": sum(1 for r in per_row if r["error"]),
        "recall_at_5": mean("recall_at_5"),
        "precision_at_5": mean("precision_at_5"),
        "mrr": mean("mrr"),
        "ndcg_at_10": mean("ndcg_at_10"),
        "by_persona": by_persona,
        "by_requirement_count": {
            "four_or_more": {"rows": len(many), "recall_at_5": mean("recall_at_5", many),
                             "ndcg_at_10": mean("ndcg_at_10", many)},
            "fewer_than_four": {"rows": len(few), "recall_at_5": mean("recall_at_5", few),
                                "ndcg_at_10": mean("ndcg_at_10", few)},
        },
        "latency_ms_mean": round(statistics.mean(latencies), 2) if latencies else 0.0,
        "latency_ms_p95": round(
            sorted(latencies)[int(len(latencies) * 0.95)] if len(latencies) > 1 else (latencies or [0])[0],
            2,
        ),
        "wall_clock_sec": round(time.perf_counter() - started, 2),
        "per_row": per_row,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="B4 retrieval metrics per variant")
    parser.add_argument("--variants", nargs="+", default=None, help="default: all registered")
    parser.add_argument("--personas", nargs="+", default=list(VALID_PERSONAS))
    parser.add_argument("--limit", type=int, default=0, help="cap rows, for a quick check")
    parser.add_argument("--llm", action="store_true", help="let V1/V8 use a model")
    args = parser.parse_args()

    setup = describe()
    rows = load_dev_set(args.personas)
    if args.limit:
        rows = rows[: args.limit]
    jobs = load_jobs()

    variant_ids = args.variants or [spec["id"] for spec in list_variants()]
    llm = None
    if args.llm:
        from agent.llm import get_llm

        llm = get_llm()
        if llm.offline:
            print("--llm given but no provider key found; running rule-based paths")
            llm = None

    print(f"B4 retrieval evaluation: {len(rows)} dev rows with gold evidence")
    print(f"personas: {', '.join(args.personas)}")
    print(f"embeddings backend: {setup['embeddings_backend']}")
    if setup["embeddings_backend"] == "lsa":
        # The spec requires open-weight embeddings. LSA is a development stand-in, and a
        # number produced with it is not a spec-compliant result.
        print("  WARNING: 'lsa' is TF-IDF+SVD, not an open-weight embedding model.")
        print("  Install sentence-transformers and set retrieval.embeddings.backend: local")
        print("  before quoting these numbers as the measured comparison.")
    print()

    results = [
        evaluate_variant(vid, rows, jobs, llm=llm) for vid in variant_ids
    ]

    print(f"{'variant':8} {'Recall@5':>9} {'P@5':>7} {'MRR':>7} {'nDCG@10':>8} "
          f"{'ms/query':>9} {'err':>4}")
    print("-" * 62)
    for r in results:
        print(f"{r['variant']:8} {r['recall_at_5']:>9.3f} {r['precision_at_5']:>7.3f} "
              f"{r['mrr']:>7.3f} {r['ndcg_at_10']:>8.3f} {r['latency_ms_mean']:>9.1f} "
              f"{r['errors']:>4}")
    print()

    baseline = next((r for r in results if r["variant"] == "V0"), None)
    if baseline:
        print("Against V0 (Recall@5 / nDCG@10 percentage points):")
        for r in results:
            if r["variant"] == "V0":
                continue
            dr = (r["recall_at_5"] - baseline["recall_at_5"]) * 100
            dn = (r["ndcg_at_10"] - baseline["ndcg_at_10"]) * 100
            print(f"  {r['variant']:4} {dr:+6.1f}  {dn:+6.1f}")
        print()
        print("Jobs with 4+ requirements (R&D question B's hypothesis):")
        for r in results:
            block = r["by_requirement_count"]["four_or_more"]
            delta = (block["recall_at_5"] - baseline["by_requirement_count"]["four_or_more"]["recall_at_5"]) * 100
            print(f"  {r['variant']:4} Recall@5 {block['recall_at_5']:.3f} ({delta:+.1f} vs V0) "
                  f"over {block['rows']} jobs")

    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / "rag_eval.json"
    out.write_text(
        json.dumps(
            {
                "metric_definitions": {
                    "relevance": "an evidence item is relevant when its project_id is a gold project",
                    "credit": "counted once per project, so several views of one project score once",
                    "ndcg_gain": "2**relevance - 1, relevance graded 1 to 3 in the dev set",
                },
                "setup": setup,
                "embeddings_spec_compliant": setup["embeddings_backend"] != "lsa",
                "dev_rows": len(rows),
                "personas": args.personas,
                "used_llm": llm is not None,
                "variants": results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nwritten: {out}")


if __name__ == "__main__":
    main()
