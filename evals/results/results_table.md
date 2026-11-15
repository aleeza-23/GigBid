# Results

Every number below is read from a file in `evals/results/`. Cells marked
`-` have no result file yet; the footnotes say which command produces them.

## B5 variant comparison

| Variant | Evidence Recall@5 | nDCG@10 | Unsupported claims / proposal | Judge score | GigSim reply rate | p95 latency (s) | Cost / proposal (USD) |
|---|---|---|---|---|---|---|---|
| V0 | 0.582 | 0.750 | - | - | - | 0.41 | - |
| V1 | 0.651 | 0.774 | - | - | - | 0.18 | - |
| V12 | 0.550 | 0.666 | - | - | - | 0.35 | - |
| V2 | 0.581 | 0.768 | - | - | - | 0.01 | - |
| V3 | 0.592 | 0.744 | - | - | - | 15.15 | - |
| V8 | 0.571 | 0.747 | - | - | - | 0.09 | - |

*The unsupported-claims column is per retrieval variant and is empty on purpose: `make verify-eval` measures the verifier against labelled claims with fixed evidence, which does not vary by variant. Its numbers are in **Claim verification** below. Filling this column needs model-written drafts per variant.*

### Retrieval detail

| Variant | Recall@5 | P@5 | MRR | nDCG@10 | vs V0 (R@5 pts) | 4+ reqs R@5 |
|---|---|---|---|---|---|---|
| V0 | 0.582 | 0.664 | 0.851 | 0.750 | - | 0.583 |
| V1 | 0.651 | 0.698 | 0.898 | 0.774 | +6.9 | 0.618 |
| V12 | 0.550 | 0.635 | 0.795 | 0.666 | -3.2 | 0.550 |
| V2 | 0.581 | 0.654 | 0.805 | 0.768 | -0.1 | 0.547 |
| V3 | 0.592 | 0.672 | 0.800 | 0.744 | +1.0 | 0.566 |
| V8 | 0.571 | 0.674 | 0.817 | 0.747 | -1.2 | 0.553 |

*142 dev rows with gold evidence; embeddings backend `local`.*

## Decision metrics

| Metric | Value |
|---|---|
| bid-or-skip accuracy | 0.773 |
| bid-or-skip macro F1 | 0.705 |
| bid class F1 | 0.563 |
| skip class F1 | 0.846 |
| scam precision | 0.750 |
| scam recall | 1.000 |
| price inside fair band | 0.752 |
| injection detection rate | 1.000 |
| **injection attack success rate** | 0.000 |
| injection false positives | 0 |

*198 dev jobs, retrieval variant V1.*

## Ablation (R&D question B)

Pipeline: `decompose (V1) -> BM25 + dense RRF (V2) -> cross-encoder rerank (V3)`

| Arm | Removed | Recall@5 | nDCG@10 | MRR |
|---|---|---|---|---|
| `full` | nothing (reference) | 0.563 | 0.670 | 0.536 |
| `-decompose` | requirement decomposition (V1) | 0.556 | 0.649 | 0.496 |
| `-rerank` | cross-encoder rerank (V3) | 0.661 | 0.825 | 0.935 |
| `-bm25` | the lexical half of the hybrid (V2) | 0.575 | 0.695 | 0.542 |
| `-dense` | the dense half of the hybrid (V2) | 0.561 | 0.673 | 0.537 |
| `-decompose-rerank` | both V1 and V3 | 0.581 | 0.768 | 0.805 |
| `-bm25 (no rerank)` | the lexical half, measured without the reranker masking it | 0.636 | 0.749 | 0.785 |
| `-dense (no rerank)` | the dense half, measured without the reranker masking it | 0.614 | 0.813 | 0.939 |

Components ranked by Recall@5 points lost when removed:

- **+0.7** requirement decomposition (V1)
- **+0.2** the dense half of the hybrid (V2)
- **-1.2** the lexical half of the hybrid (V2)
- **-1.8** both V1 and V3
- **-5.1** the dense half, measured without the reranker masking it
- **-7.3** the lexical half, measured without the reranker masking it
- **-9.8** cross-encoder rerank (V3)

## Judge calibration

*Not calibrated yet.* Fill in the `human_*` columns of
`evals/datasets/hand_check_sample.csv`, then run `make judge-calibration`.
Until then no judge score may be reported: B4 requires a calibrated judge.

## R&D question A: tool granularity

Model `openai/gpt-oss-20b`, 40 tasks.

| Arm | Tools | Tool-selection accuracy | Argument accuracy | Tokens/task | Failed |
|---|---|---|---|---|---|
| fine | 17 | 0.892 | 0.838 | 1204 | 0 |
| coarse | 4 | 0.850 | 0.825 | 781 | 0 |
| best | 11 | 0.925 | 0.850 | 1465 | 0 |

## R&D question C: is the multi-agent design worth it?

Model `openai/gpt-oss-120b`, 12 jobs.

| Arm | bid/skip acc | bid F1 | scam recall | unsupported/proposal | tokens/job | ms/job |
|---|---|---|---|---|---|---|
| single_react | 0.750 | 0.571 | 1.000 | 0.00 | 1638 | 2306 |
| supervisor | 0.500 | 0.000 | 1.000 | 2.00 | 8829 | 14049 |
| hand_built_graph | 0.750 | 0.571 | 1.000 | 0.00 | 4149 | 88258 |

> Only 12 jobs, not the 50 the brief asks for. The provider meters tokens per day per model and three arms over 50 jobs needs roughly 775k; this is a verification run, not the result. Rerun `make agent-comparison` with quota.

## C6: exactly-once across a restart

**PASSED** - job `upw-01934328d0460f74ad`, 1 proposal created, 12 Connects spent (job costs 12).

- PASS paused before spending anything
- PASS pending approval survived the restart
- PASS submitted after approval
- PASS retry did not create a second proposal
- PASS retry reported itself as a replay
- PASS connects spent exactly once
- PASS the same key was reused, not regenerated

Three separate OS processes: run to the interrupt, exit, approve from a new process, then retry from a third.

## Claim verification (B3)

| Metric | Value |
|---|---|
| fabrications caught | 108/108 |
| fabrication recall | 1.000 |
| flag precision | 0.947 |
| F1 | 0.973 |
| supported claims wrongly flagged | 6/36 |
| unsupported per proposal, before stripping | 3.17 |
| unsupported per proposal, after stripping | 0.00 |
| p95 verification latency (ms) | 1.6 |

Caught, by kind of fabrication:

- false credential: 36/36
- inflated number: 36/36
- wrong technology: 36/36

*36 probe proposals, 180 labelled sentences, generated from the real corpora.*

## Declared budgets against measurements (E5)

| Metric | Declared | Measured | Status |
|---|---|---|---|
| p95 latency, job analysis (s) | 25 | 7.642 | WITHIN |
| p95 latency, proposal (s) | 90 | 79.42 | WITHIN |
| USD per job analysed | 0.01 | 0.0011 | WITHIN |
| USD per proposal | 0.05 | 0.0069 | WITHIN |
| tokens per job | 12000 | 4068.0 | WITHIN |
| tokens per proposal | 20000 | 24298.0 | OVER |

*6 recorded run(s); runs made with the model offline are excluded, since their latency and cost are not real.*

## X1 agency mode (signature extension)

| Metric | Value |
|---|---|
| jobs routed | 41/60 |
| declined to route | 19 |
| accuracy when committed | 0.878 |
| accuracy over all jobs | 0.600 |
| mean coverage, routed | 0.840 |
| mean coverage, best fixed member | 0.816 (ai_ml_engineer) |
| coverage gained (points) | +2.42 |
| cost | 3 retrieval passes per job, 1.75 s/job |

See `docs/EXTENSION.md` for the design and what the numbers mean.

## Missing results

- `judge_calibration.json` (judge score, Cohen's kappa) - run `make judge-calibration`
- `gigsim_outcomes.json` (reply rate, hire rate) - run `not measurable: GigSim does not model client replies`
