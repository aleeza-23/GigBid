.PHONY: install dev api test lint eval seed data gigsim demo \
        mcp-portfolio mcp-marketplace mcp-workspace inspect-portfolio inspect-marketplace \
        snapshot-update tool-study

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
