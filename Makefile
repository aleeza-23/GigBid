.PHONY: install dev api test lint eval seed data gigsim demo
.PHONY: rag-eval decision-eval ablation judge-calibration ingest ingest-stats
.PHONY: results-table eval-b tool-study
.PHONY: graphs persistence test-graphs crash-resume agent-comparison
.PHONY: mcp-portfolio mcp-marketplace mcp-workspace inspect-portfolio inspect-marketplace
.PHONY: snapshot-update
.PHONY: web web-install web-build e2e e2e-report test-api openapi
.PHONY: worker worker-once worker-stats smoke-eval budget-report trace-summary
.PHONY: up down logs verify-eval agency-eval
install:
	python -m pip install -e ".[dev,data]"

# Everything a reviewer needs running, in one terminal each. The assignment's `make dev`.
dev:
	@echo "Start these in separate terminals:"
	@echo "  make gigsim           GigSim marketplace      :8081"
	@echo "  make api              FastAPI                 :8000"
	@echo "  make mcp-marketplace  Marketplace MCP (HTTP)  :8702"
	@echo "  make mcp-portfolio    Portfolio MCP (HTTP)    :8701"
	@echo "  make mcp-workspace    Workspace MCP (HTTP)    :8703"
	@echo ""
	@echo "Or: docker compose -f infra/docker-compose.yml up"

api:
	uvicorn apps.api.main:app --reload --port 8000

test:
	pytest -q

lint:
	ruff check .

eval:
	python -m evals.runners.run_demo_eval

# Real data (Kaggle, via kagglehub cache) -> data/processed + data/derived
data:
	python -m data.prepare

# Synthetic personas, clients, scams, injections, conversations + labelled dev set
seed: data
	python -m data.synth.generate --seed 42

# GigSim marketplace simulator (ours; no simulator shipped with the assignment)
gigsim:
	python -m gigsim

# ----------------------------------------------------------------- MCP servers
# stdio by default (what Claude Desktop and Cursor expect); --http for the deployed stack.

mcp-portfolio:
	python -m mcp_servers.portfolio.server --http

mcp-marketplace:
	python -m mcp_servers.marketplace.server --http

mcp-workspace:
	python -m mcp_servers.workspace.server --http

# MCP Inspector, for the screenshots A7 asks for. Needs Node.
inspect-portfolio:
	npx @modelcontextprotocol/inspector python -m mcp_servers.portfolio.server

inspect-marketplace:
	npx @modelcontextprotocol/inspector python -m mcp_servers.marketplace.server

# Re-record the tool-schema snapshots after an intended signature change (A7).
# Review the diff before committing: that review is the point of the snapshot.
snapshot-update:
	UPDATE_SNAPSHOTS=1 pytest tests/contract -q

# ------------------------------------------------------------------ Part A demo
# One agent drafting one grounded proposal for a GigSim job, start to finish.
demo:
	python -m evals.runners.single_agent_demo

# R&D question A: tool granularity (fine-grained vs coarse vs rewritten descriptions).
tool-study:
	python -m evals.runners.tool_granularity

# ------------------------------------------------------------------- Part B evals
# B4 retrieval metrics per variant: Recall@5, MRR, nDCG@10, by persona.
rag-eval:
	python -m evals.runners.rag_eval

# B4 decision metrics: bid/skip F1, scam P/R, price in band, injection defence.
decision-eval:
	python -m evals.runners.decision_eval

# R&D question B: ablate the best pipeline one component at a time.
ablation:
	python -m evals.runners.ablation

# B4 judge calibration. Needs the human_* columns of
# evals/datasets/hand_check_sample.csv filled in first; it refuses otherwise.
judge-calibration:
	python -m evals.runners.judge_calibration

# B1 ingestion: load, chunk and index a portfolio. Incremental by default.
ingest:
	python -m rag.ingest.pipeline --all

ingest-stats:
	python -m rag.ingest.pipeline --persona ai_ml_engineer --stats

# B3: does claim verification catch fabrication? Measured against labelled probes
# generated from the real corpora, so it runs offline and in CI.
verify-eval:
	python -m evals.runners.verify_eval

# X1 (signature extension): routing accuracy and the coverage it gains over a single
# freelancer. See docs/EXTENSION.md.
agency-eval:
	python -m evals.runners.agency_eval

# B5: regenerate the results table from the committed result files.
results-table:
	python -m evals.runners.results_table

# ------------------------------------------------------------------ Part C graphs
# Render both graphs from the compiled code, so docs cannot drift from the graph.
graphs:
	@python -c "from agent.graphs.bid_graph import render_mermaid as f; print(f())"
	@echo ""
	@python -c "from agent.graphs.conversation_graph import render_mermaid as f; print(f())"

# What persistence is actually configured, and whether it survives a restart.
persistence:
	@python -c "import json; from agent.checkpointer import describe; print(json.dumps(describe(), indent=2))"

# C6: kill the process while a proposal waits for approval, restart, approve once.
# Needs GigSim running as a real server (make gigsim) in another terminal.
crash-resume:
	python -m evals.runners.crash_resume_demo

# R&D question C: single ReAct vs prebuilt supervisor vs the hand-built graph.
# Needs a provider key. Groq meters tokens per day per model, so a full 50-job run
# across three arms needs roughly 775k tokens; check your quota before the full run.
agent-comparison:
	python -m evals.runners.agent_comparison --limit 50

# Graph tests run with the model forced offline: they check routing, reducers, loop
# bounds and interrupts, which should not depend on what a model returned.
test-graphs:
	GIGBID_FORCE_OFFLINE=1 pytest tests/integration/test_bid_graph.py tests/integration/test_conversation_graph.py -q

# Everything in Part B that runs without a provider key.
eval-b: ingest rag-eval decision-eval verify-eval ablation results-table

# ------------------------------------------------------------------ Part D app
# The API is `make api` above. These are the frontend and the end-to-end suite.

web-install:
	cd apps/web && npm ci

# Dev server with hot reload, on :3000. Expects `make api` in another terminal.
web:
	cd apps/web && npm run dev

web-build:
	cd apps/web && npm run build

# D5: the seven flows, against real GigSim + API + Next. Playwright starts all three
# itself on private ports (8481/8400/3400) so it cannot adopt another project's server.
# The model is forced offline, so a failure here is a wiring failure, not a bad draft.
e2e: web-build
	cd apps/web && npx playwright test

e2e-report:
	cd apps/web && npx playwright show-report

# API integration tests only: auth, approvals, idempotency, SSE.
test-api:
	pytest tests/integration/test_api.py -q

# Regenerate openapi.json, which the frontend's types are generated from.
openapi:
	@python -c "import json; from apps.api.main import app; print(json.dumps(app.openapi(), indent=2))" > openapi.json
	@echo "wrote openapi.json"
	cd apps/web && npm run generate:api

# ------------------------------------------------------------- Part E operations
# E1: the scheduled worker. Scans on a cron-ish loop, dedupes against the ledger,
# and never queues the same job twice. Ctrl-C is a clean shutdown.
worker:
	python -m worker.scheduler

# One pass and exit, ignoring the schedule. What CI and a first run should use.
worker-once:
	python -m worker.scheduler --once --force

# What the ledger has seen: jobs queued, duplicates suppressed, last run.
worker-stats:
	python -m worker.scheduler --stats

# E2: the CI quality gate. Thresholds live in evals/thresholds.json; a regression
# below any of them exits non-zero and fails the build.
smoke-eval:
	python -m evals.runners.smoke_eval

# E5: declared budgets against what the traces actually measured.
budget-report:
	python -m evals.runners.budget_report

# E3: the span log for one run, by node, with tokens, latency and cost.
# Pass a trace: make trace-summary TRACE=tr-abc123. Without one, the newest.
trace-summary:
	@python -c "import json,sys; from pathlib import Path; from agent.tracing import TRACE_DIR, summarise_trace; t='$(TRACE)' or (sorted(TRACE_DIR.glob('*.jsonl'), key=lambda p: p.stat().st_mtime)[-1].stem if any(TRACE_DIR.glob('*.jsonl')) else ''); print('no traces recorded yet; run `make demo`') if not t else print(json.dumps(summarise_trace(t), indent=2))"

# E6: the whole stack in containers. GigSim, API, web, three MCP servers, the worker,
# Postgres and Redis.
up:
	docker compose -f infra/docker-compose.yml up --build

down:
	docker compose -f infra/docker-compose.yml down -v

logs:
	docker compose -f infra/docker-compose.yml logs -f --tail=100
