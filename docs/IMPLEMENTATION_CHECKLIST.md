# Requirement to code map

Where each requirement in the brief is implemented, and how to see it working. The
numbering follows the assignment.

## Part A — MCP tool layer

| | Where | Check it |
|---|---|---|
| A1 Marketplace port and GigSim adapter | `mcp_servers/marketplace/adapters/` | `make gigsim`, then `make demo` |
| A2 Portfolio server: 4 tools, 5 resources, 5 prompts | `mcp_servers/portfolio/` | `make inspect-portfolio` |
| A3 Marketplace server: 11 tools, idempotent writes | `mcp_servers/marketplace/` | `docs/TOOLS.md` |
| A4 Workspace server | `mcp_servers/workspace/` | `make mcp-workspace` |
| A5 Live Upwork adapter behind the same port | `mcp_servers/marketplace/adapters/upwork_live.py` | writes are drafts only |
| A6 Per-agent tool allowlists | `agent/mcp_client.py` | `docs/TOOLS.md` section 5 |
| A7 Schemas, annotations, auth, idempotency, tests | `mcp_servers/_common/` | `make test`, `tests/contract/` |
| R&D A Tool-design study | `evals/runners/tool_granularity.py` | `make tool-study` |

## Part B — Retrieval and grounding

| | Where | Check it |
|---|---|---|
| B1 Ingestion: CVs, case studies, READMEs, code, reviews, proposals | `rag/ingest/` | `make ingest`, `make ingest-stats` |
| B2 V0 plus five variants | `rag/variants/` | `make rag-eval` |
| B3 Claim splitting and verification | `rag/verify.py` | `make verify-eval` |
| B4 Retrieval and decision metrics | `evals/runners/` | `make rag-eval`, `make decision-eval` |
| B5 Results table | `evals/runners/results_table.py` | `make results-table` |
| R&D B Ablation | `evals/runners/ablation.py` | `make ablation` |

## Part C — Multi-agent graphs

| | Where | Check it |
|---|---|---|
| C1 Bid graph and conversation graph | `agent/graphs/` | `make graphs` |
| C2 Typed hand-offs and state reducers | `agent/handoffs.py`, `agent/state.py` | `make test-graphs` |
| C3 Parallel analysis branches | `agent/graphs/bid_graph.py` (`fan_out`, `dispatch`) | the trace lists three branches |
| C4 Bounded Critic loop | `agent/nodes/critic.py` (`route_after_critique`) | `make test-graphs` |
| C5 Interrupts before every external action | `agent/nodes/approval.py` | `make e2e` |
| C6 Durable checkpointing, crash and resume | `agent/checkpointer.py` | `make crash-resume` |
| C7 Long-term memory with promotion rules | `agent/memory/` | `make test` |
| R&D C Single agent vs supervisor vs this graph | `evals/runners/agent_comparison.py` | `make agent-comparison` |

## Part D — Application

| | Where | Check it |
|---|---|---|
| D1 API: auth, jobs, runs, approvals, messages | `apps/api/` | `make api`, `make test-api` |
| D2 Streaming agent progress | `apps/api/streaming.py` | the run viewer |
| D3 Screens: inbox, job, studio, run, pipeline, analytics | `apps/web/src/app/` | `make web` |
| D4 Approve, edit, reject | `apps/web/src/app/studio/[id]/` | `make e2e` |
| D5 Playwright journeys | `apps/web/tests/e2e/` | `make e2e` |

## Part E — Operations

| | Where | Check it |
|---|---|---|
| E1 Scheduled worker with a dedupe ledger | `worker/` | `make worker-once`, `make worker-stats` |
| E2 Evaluation gate in CI | `evals/runners/smoke_eval.py`, `.github/workflows/` | `make smoke-eval` |
| E3 Tracing per node, with cost | `agent/tracing.py` | `make trace-summary` |
| E4 Untrusted input handling | `agent/decide.py` (`detect_injection`) | `make decision-eval` |
| E5 Declared budgets against measurements | `evals/runners/budget_report.py` | `make budget-report` |
| E6 Containers for the whole stack | `infra/` | `make up` |

## Deliverables

| | Where |
|---|---|
| Architecture write-up | `docs/ARCHITECTURE.md` |
| Tool inventory | `docs/TOOLS.md` |
| Data provenance | `docs/DATA_CARD.md` |
| Operations runbook | `docs/RUNBOOK.md` |
| Signature extension (X1 agency mode) | `docs/EXTENSION.md` |
| AI usage disclosure | `docs/AI_USAGE.md` |
| Decisions and why | `docs/adr/` |
| Committed results | `evals/results/` |

## Not done, and why

- **Judge calibration** needs 30 human labels in `evals/datasets/hand_check_sample.csv`.
  The runner refuses to report agreement against labels it generated itself.
- **GigSim reply and hire rates** are not measurable: the simulator does not model client
  replies, so those columns in the results table stay gaps.
- **R&D C at 50 jobs** needs about 775k tokens across three arms against a 200k per day
  per model free-tier limit. The committed run covers a stratified sample and says so.
