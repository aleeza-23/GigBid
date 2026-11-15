"""R&D question B's ablation: remove one component of the best pipeline at a time.

The best pipeline is a composition, not a single variant:

    decompose (V1) -> per requirement: BM25 + dense fused by RRF (V2) -> rerank (V3)

Comparing V0 to V3 tells you the stack helps. It does not tell you *which part* helps,
and on this corpus that question has a real answer: V1 alone gains more than V2, which
suggests the dominant problem is query blending rather than retrieval quality. An ablation
is the only way to separate those, because V3 is layered on V2's candidates and their
contributions are otherwise impossible to attribute.

Each arm removes exactly one thing from the full pipeline and keeps everything else, so
the delta is attributable. The arms that *add* rather than remove (outcome-aware ranking,
metadata-free indexing) are marked as such.

    make ablation
    python -m evals.runners.ablation --limit 40
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from evals.runners.rag_eval import (
    _dedupe_by_project,
    _gold_grades,
    build_query,
    load_dev_set,
    load_jobs,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from mcp_servers.portfolio.corpus import EvidenceItem, load_corpus
from rag.index.bm25 import get_bm25_index
from rag.index.dense import get_dense_index
from rag.variants import v1_decompose, v3_rerank
from rag.variants.v2_hybrid import reciprocal_rank_fusion

RESULTS = Path(__file__).resolve().parents[2] / "evals" / "results"


# --------------------------------------------------------------- pipeline pieces


def _hybrid_once(
    query: str,
    items: Sequence[EvidenceItem],
    depth: int,
    *,
    use_bm25: bool = True,
    use_dense: bool = True,
    config: dict[str, Any] | None = None,
) -> list[tuple[str, float]]:
    """One fused retrieval, with either retriever switchable off."""
    rankings: list[list[str]] = []
    if use_bm25:
        lexical = get_bm25_index(items)
        rankings.append([item.id for item, _ in lexical.rank(query, depth)])
    if use_dense:
        from rag.embeddings import cosine_scores

        dense = get_dense_index(items, (config or {}).get("embeddings"))
        vector = dense.encode_query(query)
        scores = cosine_scores(vector, dense.matrix)
        rankings.append(
            [
                item.id
                for item, _ in sorted(
                    zip(dense.items, scores, strict=True), key=lambda p: -float(p[1])
                )[:depth]
            ]
        )
    if not rankings:
        return []
    fused = reciprocal_rank_fusion(rankings)
    return sorted(fused.items(), key=lambda pair: (-pair[1], pair[0]))[:depth]


def full_pipeline(
    job: dict[str, Any] | str,
    items: Sequence[EvidenceItem],
    k: int = 10,
    *,
    decompose: bool = True,
    use_bm25: bool = True,
    use_dense: bool = True,
    rerank: bool = True,
    config: dict[str, Any] | None = None,
    depth: int = 30,
) -> list[EvidenceItem]:
    """The composed best pipeline, with each stage switchable for the ablation."""
    by_id = {item.id: item for item in items}

    if decompose:
        parts = v1_decompose.decompose(job)
    else:
        # The un-decomposed query: the whole post as one string, as V0 uses it.
        parts = [
            job
            if isinstance(job, str)
            else " ".join(
                filter(
                    None,
                    [
                        str(job.get("title") or ""),
                        str(job.get("description") or "")[:3000],
                        " ".join(str(s) for s in (job.get("skills") or [])),
                    ],
                )
            )
        ]

    # Merge per-requirement retrievals by max fused score, as V1 does.
    best: dict[str, float] = {}
    for part in parts:
        for item_id, score in _hybrid_once(
            part, items, depth, use_bm25=use_bm25, use_dense=use_dense, config=config
        ):
            best[item_id] = max(best.get(item_id, 0.0), score)

    ordered = sorted(best.items(), key=lambda pair: (-pair[1], pair[0]))
    candidates = [
        by_id[item_id].clone(score=round(score, 5))
        for item_id, score in ordered
        if item_id in by_id
    ]

    if not rerank or not candidates:
        return candidates[:k]

    # Rerank against the original post, not a single requirement: the cross-encoder should
    # judge relevance to the job as a whole.
    whole = parts[0] if len(parts) == 1 else " ; ".join(parts)
    model = v3_rerank.get_cross_encoder()
    if model is not None:
        pairs = [
            (whole, f"{c.project or ''} {' '.join(c.skills)} {c.text}"[:2000]) for c in candidates
        ]
        scores = model.predict(pairs)
    else:
        scores = [v3_rerank._lexical_rerank_score(whole, c) for c in candidates]

    ranked = sorted(zip(candidates, scores, strict=True), key=lambda p: (-float(p[1]), p[0].id))
    return [item.clone(score=round(float(score), 4)) for item, score in ranked[:k]]


# ------------------------------------------------------------------------- arms


def arms() -> list[dict[str, Any]]:
    """One arm per component removed. `full` is the reference."""
    return [
        {
            "name": "full",
            "removed": None,
            "description": "decompose + BM25 + dense (RRF) + rerank",
            "kwargs": {},
        },
        {
            "name": "-decompose",
            "removed": "requirement decomposition (V1)",
            "description": "whole post as one query, everything else unchanged",
            "kwargs": {"decompose": False},
        },
        {
            "name": "-rerank",
            "removed": "cross-encoder rerank (V3)",
            "description": "fused candidates returned in RRF order",
            "kwargs": {"rerank": False},
        },
        {
            "name": "-bm25",
            "removed": "the lexical half of the hybrid (V2)",
            "description": "dense only, then rerank",
            "kwargs": {"use_bm25": False},
        },
        {
            "name": "-dense",
            "removed": "the dense half of the hybrid (V2)",
            "description": "BM25 only, then rerank",
            "kwargs": {"use_dense": False},
        },
        {
            "name": "-decompose-rerank",
            "removed": "both V1 and V3",
            "description": "plain hybrid retrieval, closest thing to V2 alone",
            "kwargs": {"decompose": False, "rerank": False},
        },
        # The two arms above that remove half the hybrid are uninformative while the
        # reranker is active: it re-scores every candidate from scratch, so swapping the
        # candidate *order* changes nothing as long as the candidate *set* overlaps, and
        # at depth 30 over ~197 items the sets overlap almost entirely. Measuring the
        # retrieval halves therefore requires turning the reranker off.
        {
            "name": "-bm25 (no rerank)",
            "removed": "the lexical half, measured without the reranker masking it",
            "description": "dense only, RRF order kept",
            "kwargs": {"use_bm25": False, "rerank": False},
        },
        {
            "name": "-dense (no rerank)",
            "removed": "the dense half, measured without the reranker masking it",
            "description": "BM25 only, RRF order kept",
            "kwargs": {"use_dense": False, "rerank": False},
        },
    ]


def run_arm(
    arm: dict[str, Any],
    rows: list[dict[str, Any]],
    jobs: dict[str, dict[str, Any]],
    corpora: dict[str, tuple[EvidenceItem, ...]],
    depth: int = 10,
) -> dict[str, Any]:
    per_row: list[dict[str, Any]] = []
    started = time.perf_counter()

    for row in rows:
        grades = _gold_grades(row)
        query, job = build_query(row, jobs)
        items = corpora[row["persona"]]
        try:
            hits = full_pipeline(job or query, items, depth, **arm["kwargs"])
            error = None
        except Exception as exc:
            hits, error = [], f"{type(exc).__name__}: {exc}"

        ranked = _dedupe_by_project(hits)
        per_row.append(
            {
                "job_id": row["job_id"],
                "persona": row["persona"],
                "requirement_count": len((row.get("gold") or {}).get("requirements") or []),
                "recall_at_5": recall_at_k(ranked, grades, 5),
                "precision_at_5": precision_at_k(ranked, grades, 5),
                "mrr": reciprocal_rank(ranked, grades),
                "ndcg_at_10": ndcg_at_k(ranked, grades, 10),
                "error": error,
            }
        )

    ok = [r for r in per_row if r["error"] is None]

    def mean(key: str, pool: list[dict[str, Any]] | None = None) -> float:
        values = [r[key] for r in (pool if pool is not None else ok)]
        return round(statistics.mean(values), 4) if values else 0.0

    many = [r for r in ok if r["requirement_count"] >= 4]
    return {
        "arm": arm["name"],
        "removed": arm["removed"],
        "description": arm["description"],
        "rows": len(per_row),
        "errors": len(per_row) - len(ok),
        "recall_at_5": mean("recall_at_5"),
        "precision_at_5": mean("precision_at_5"),
        "mrr": mean("mrr"),
        "ndcg_at_10": mean("ndcg_at_10"),
        "recall_at_5_four_plus_requirements": mean("recall_at_5", many),
        "wall_clock_sec": round(time.perf_counter() - started, 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="R&D question B ablation")
    parser.add_argument("--personas", nargs="+", default=None)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    rows = load_dev_set(args.personas)
    if args.limit:
        rows = rows[: args.limit]
    jobs = load_jobs()
    corpora = {p: load_corpus(p) for p in {r["persona"] for r in rows}}

    from rag.retriever import describe

    setup = describe()
    print(f"ablation over {len(rows)} dev rows; embeddings backend: "
          f"{setup['embeddings_backend']}")
    if setup["embeddings_backend"] == "lsa":
        print("  WARNING: dense half is TF-IDF+SVD, not an open-weight embedding model.")
        print("  The -dense and -bm25 arms especially will read differently on a real model.")
    print()

    results = [run_arm(arm, rows, jobs, corpora) for arm in arms()]
    reference = results[0]

    print(f"{'arm':20} {'Recall@5':>9} {'nDCG@10':>8} {'MRR':>7} {'dR@5':>7} {'dnDCG':>7}")
    print("-" * 64)
    for r in results:
        dr = (r["recall_at_5"] - reference["recall_at_5"]) * 100
        dn = (r["ndcg_at_10"] - reference["ndcg_at_10"]) * 100
        marker = "" if r["arm"] == "full" else f"{dr:+7.1f} {dn:+7.1f}"
        print(f"{r['arm']:20} {r['recall_at_5']:>9.3f} {r['ndcg_at_10']:>8.3f} "
              f"{r['mrr']:>7.3f} {marker}")
    print()
    print("A large negative delta means the removed component was carrying the pipeline.")
    print("A delta near zero means it is not earning its latency and should not ship.")

    # An arm that exactly reproduces the reference is almost always a masked measurement
    # rather than a component that genuinely does nothing. Say so rather than letting the
    # report conclude "BM25 contributes nothing".
    identical = [
        r["arm"]
        for r in results[1:]
        if abs(r["recall_at_5"] - reference["recall_at_5"]) < 1e-9
        and abs(r["ndcg_at_10"] - reference["ndcg_at_10"]) < 1e-9
    ]
    if identical:
        print(f"\nNOTE: {identical} scored identically to 'full'.")
        print("  That is a masked measurement, not a zero contribution: the reranker")
        print("  re-scores candidates from scratch, so removing one retriever changes the")
        print("  candidate order but not the set the reranker chooses from. Read the")
        print("  '(no rerank)' arms for those components instead.")

    # Rank components by how much their removal costs, which is the answer the report wants.
    contributions = sorted(
        (
            {
                "component": r["removed"],
                "arm": r["arm"],
                "recall_cost": round((reference["recall_at_5"] - r["recall_at_5"]) * 100, 2),
                "ndcg_cost": round((reference["ndcg_at_10"] - r["ndcg_at_10"]) * 100, 2),
            }
            for r in results
            if r["arm"] != "full"
        ),
        key=lambda c: -c["recall_cost"],
    )
    print("\ncomponents by contribution (Recall@5 points lost when removed):")
    for c in contributions:
        print(f"  {c['recall_cost']:+6.1f}  {c['component']}")

    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / "ablation.json"
    out.write_text(
        json.dumps(
            {
                "pipeline": "decompose (V1) -> BM25 + dense RRF (V2) -> cross-encoder rerank (V3)",
                "setup": setup,
                "embeddings_spec_compliant": setup["embeddings_backend"] != "lsa",
                "dev_rows": len(rows),
                "arms": results,
                "contributions_ranked": contributions,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nwritten: {out}")


if __name__ == "__main__":
    main()
