"""Fit Matcher: map every requirement to portfolio evidence. Portfolio read only (A6).

One retrieval per requirement, which is the one component Part B found worth its cost:
+6.9 Recall@5 over the blended-query baseline, for about 50 ms a query
(`evals/results/rag_eval.json`). Reranking, measured the same way, cost 9.8 points, which
is why the shipped variant has none.

Retrieval runs through the configured variant, so whichever design the evaluation picked
is the one that ships (R6). Coverage is computed by rule, because it is a counting
operation with a measured threshold; the model is used only to catch the case a lexical
score cannot see, which is topical adjacency masquerading as coverage. A project using
PostgreSQL is not evidence of Snowflake modelling even though every word overlaps.
"""

from __future__ import annotations

from typing import Any

from langchain_core.runnables import RunnableConfig

from agent.decide import assess_fit
from agent.nodes._common import call_model, compact, node_trace, retrieval_config, step
from agent.state import GigBidState

# Below this retrieval score a requirement is not considered covered. Taken from the
# threshold the Part B evaluation used, so the graph and the metrics agree.
MIN_EVIDENCE_SCORE = 2.0


async def fit_matcher(state: GigBidState, config: RunnableConfig | None = None) -> dict[str, Any]:
    from rag.variants.v8_corrective import retrieve_with_report

    persona = state.get("persona") or "ai_ml_engineer"

    # The Fit Matcher derives its own requirements from the post rather than waiting for
    # the Job Analyst's JobSpec. That is what makes R3's three-way parallelism real: if
    # this node read `job_spec`, it would be serialised behind the Analyst and the fan-out
    # would be decorative. An earlier version did exactly that and silently returned 0%
    # coverage on every job, because the Analyst had not finished when this node ran.
    #
    # It uses the same rule-based `decompose` the V1 variant uses, so the coverage numbers
    # here are the ones Part B measured. The Analyst's richer JobSpec is still what the
    # approval screen shows; the two are used for different things on purpose.
    from rag.variants.v1_decompose import decompose

    job = dict(state.get("job") or {})
    job["description"] = state.get("job_text_sanitised") or job.get("description") or ""
    requirements = decompose(job)

    if not requirements:
        return {
            "analyses_complete": ["fit_matcher"],
            "steps_used": step(state),
            "execution_trace": node_trace(
                "fit_matcher",
                "decision",
                result_summary={"requirements": 0, "skipped": "no requirements in the post"},
            ),
        }

    from mcp_servers.portfolio.corpus import load_corpus

    items = load_corpus(persona)
    hits_per_requirement: dict[str, list[dict[str, Any]]] = {}
    all_evidence: list[dict[str, Any]] = []
    covered_flags: dict[str, bool] = {}
    rounds = 0

    for requirement in requirements:
        # V8's corrective loop: it re-queries when the evidence is weak and reports when
        # there is none worth writing from, which is what lets the decision node skip
        # rather than let the Writer invent.
        report = retrieve_with_report(requirement, items, 3, config=retrieval_config(config))
        hits = [hit.to_dict(text_limit=700) for hit in report["hits"]]
        hits_per_requirement[requirement] = hits
        all_evidence.extend(hits)
        # Coverage comes from V8's grading, not from a score threshold. Retrieval scores
        # are scale-dependent (BM25 runs 5-13, RRF runs 0.01-0.05), so a fixed cutoff
        # silently rejects everything from the wrong variant.
        covered_flags[requirement] = bool(report["sufficient"])
        rounds += int(report.get("rounds") or 1)

    fit = assess_fit(
        requirements,
        hits_per_requirement,
        covered_flags=covered_flags,
        evidence_sufficient=any(covered_flags.values()),
    )

    # Ask the model only about requirements the rule marked covered: those are where a
    # false positive is expensive, because the Writer will claim them.
    meta: dict[str, Any] = {}
    tokens = 0
    covered_rows = [row for row in fit.requirement_matrix if row["covered"]]
    if covered_rows:
        parsed, meta = call_model(
            "fit_matcher",
            prompt_values={
                "requirement_evidence": compact(
                    [
                        {
                            "requirement": row["requirement"],
                            "evidence": [
                                {"id": e["id"], "project": e["project"], "type": e["type"]}
                                for e in row["evidence"][:2]
                            ],
                        }
                        for row in covered_rows[:8]
                    ],
                    2000,
                )
            },
            trace_id=state.get("trace_id"),
            job_id=state.get("job_id"),
        )
        tokens = meta.get("tokens") or 0
        if isinstance(parsed, dict):
            overruled = {
                str(v.get("requirement")): v
                for v in (parsed.get("requirement_verdicts") or [])
                if v.get("covered") is False
            }
            if overruled:
                # The model can take coverage away but never grant it: a rule-covered
                # requirement it rejects becomes uncovered, and one it likes that the rule
                # rejected stays rejected. Models are better at spotting a bad match than
                # at vouching for a good one, and this direction cannot inflate coverage.
                for row in fit.requirement_matrix:
                    if row["requirement"] in overruled and row["covered"]:
                        row["covered"] = False
                        row["overruled_by_model"] = overruled[row["requirement"]].get("why")
                covered = [r for r in fit.requirement_matrix if r["covered"]]
                fit.coverage = round(len(covered) / len(fit.requirement_matrix), 3)
                fit.score = fit.coverage
                fit.matched_skills = [r["requirement"] for r in covered]
                fit.missing_skills = [
                    r["requirement"] for r in fit.requirement_matrix if not r["covered"]
                ]

    # Evidence concatenates across branches, so the state reducer appends and reads
    # deduplicate. Capped here because a 40-item evidence list would blow the Writer's
    # context for no gain.
    return {
        "fit_report": fit,
        "evidence": all_evidence[:24],
        "analyses_complete": ["fit_matcher"],
        "steps_used": step(state),
        "tokens_used": tokens,
        "execution_trace": node_trace(
            "fit_matcher",
            "model" if covered_rows else "decision",
            meta,
            tool="search_evidence",
            result_summary={
                "requirements": len(requirements),
                "coverage": fit.coverage,
                "retrieval_rounds": rounds,
                "evidence_sufficient": fit.evidence_sufficient,
                "missing": fit.missing_skills[:4],
            },
        ),
    }
