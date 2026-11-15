# Getting started

For someone opening this repository for the first time and wanting to run it or change it.
The README has the quick start; this is the orientation.

## Run it

```bash
python -m venv .venv && . .venv/Scripts/activate   # or .venv/bin/activate
make install
cp .env.example .env
make seed      # synthetic personas, clients, scams, injections, labelled dev set
make ingest    # chunk and index the portfolio corpora
```

Then three terminals: `make gigsim`, `make api`, `make web`. Sign in at
`http://localhost:3000` with `reviewer` and the password in `REVIEWER_PASSWORD`.

A provider key in `.env` (`OPENAI_API_KEY` works for Groq) turns on drafting. Without one
everything still runs: rule-based decisions are unaffected, drafts become placeholders, and
the UI shows a banner saying so. `make persistence` and `GET /health` both report what is
degraded and why.

## Read it in this order

1. `docs/ARCHITECTURE.md` — the two graphs, the state, what is parallel and what is not.
2. `docs/TOOLS.md` — every MCP tool, its annotations, and which agent may call it.
3. `agent/graphs/bid_graph.py` — the graph itself. The module docstring explains the three
   things in it that are easy to get wrong.
4. `rag/variants/v12_decompose_hybrid.py` — the retrieval configuration that ships, and the
   measurements that chose it over reranking.
5. `rag/verify.py` — claim verification, which is what stops the Writer inventing
   experience.
6. `docs/DATA_CARD.md` — where the data comes from and what is synthetic.

## Folder responsibilities

| Path | What belongs there |
|---|---|
| `mcp_servers/` | Tool surface. Three servers plus the shared auth, audit and idempotency code |
| `rag/` | Ingestion, retrieval variants, claim verification. No agent logic |
| `agent/` | State, hand-offs, nodes, graphs, prompts, memory, tracing |
| `apps/api/` | HTTP surface. Thin: routers resume graphs, they do not contain decisions |
| `apps/web/` | Next.js app. Server-side rendering plus one proxy route to the API |
| `worker/` | Scheduled sweeps and the dedupe ledger |
| `evals/` | Runners and committed results. Nothing here is imported by the product |
| `gigsim/` | The provided simulator. Unchanged |

## Changing things

**Rules, thresholds, prompts and model routing are configuration.** `agent/config.yaml`
holds the thresholds, budgets, retrieval choice, model profiles and the agency-mode block;
`agent/prompts/*.md` holds the prompts, versioned so a trace records which version ran.
Changing a rule should not mean changing code, which is what made the GigSim policy update
a configuration edit.

**Adding a retrieval variant:** write it in `rag/variants/`, export a `VARIANT` dict, add
it to the list in `rag/retriever.py`, then `make rag-eval`. Set
`retrieval.variant` once the numbers say to.

**Adding a tool:** add it to the server, give every parameter an `Annotated` alias with its
constraints (FastMCP reads the signature, not a metadata dict), add it to the right agent's
allowlist in `agent/mcp_client.py`, then `make snapshot-update` and read the diff. That
diff is the point of the snapshot.

**Adding a node:** nodes take `(state, config)` and return a state update. Every field
written by more than one node needs a reducer in `agent/state.py`; a plain field written by
two parallel branches is a lost update. Use `call_model` from `agent/nodes/_common.py` so
the call is traced and the prompt version is recorded.

## Before committing

```bash
make lint
make test
make test-graphs
```

`make e2e` needs a built frontend and starts GigSim, the API and Next.js itself on private
ports. CI runs lint, the tests, the tool-schema snapshots and `make smoke-eval`, which
fails the build if a measured value drops below a threshold in `evals/thresholds.json`.

Never commit `.env`, real client data from live mode, or tokens. `data/audit/` is ignored.
