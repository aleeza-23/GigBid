# Résumé bullet

Two lines, with numbers from `evals/results/`. Pick whichever of the two framings fits the
role.

## Measurement framing

> Built GigBid, a LangGraph multi-agent system that drafts freelance proposals grounded in
> retrieved portfolio evidence, with human approval before any send and exactly-once
> spending proven across process restarts.
> Chose the retrieval design by measurement over 142 labelled jobs: requirement
> decomposition gained 6.9 Recall@5 points while cross-encoder reranking cost 9.8 and 11.7 s
> a query, so it shipped without a reranker; claim verification caught 108 of 108 planted
> fabrications at 0.947 precision.

## Safety framing

> Built GigBid, a multi-agent freelance-proposal system across three MCP servers, 22 tools
> and two LangGraph graphs, where nothing reaches a client without human approval.
> Blocked 24 of 24 planted prompt injections with zero attack success and no false
> positives on clean jobs, caught every scam post (recall 1.000), and removed 108 of 108
> unsupported claims before any draft reached a reviewer, at $0.007 per proposal.

## Where each number comes from

| Number | File |
|---|---|
| 6.9 Recall@5 points from decomposition, 9.8 lost to reranking, 11.7 s/query | `rag_eval.json`, `ablation.json` |
| 108/108 fabrications caught, 0.947 flag precision | `claim_verification.json` |
| 24/24 injections detected, 0.000 attack success, 0 false positives | `decision_eval.json` |
| scam recall 1.000 | `decision_eval.json` |
| $0.0069 per proposal | `budget_report.json` |
| exactly-once across a restart, 7 of 7 invariants | `c6_crash_resume.json` |
