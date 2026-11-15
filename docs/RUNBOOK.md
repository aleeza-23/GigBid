# Runbook

How to start GigBid, what breaks, and what to do about it.

## Local stack, from nothing

```bash
python -m venv .venv && .venv/Scripts/activate      # or source .venv/bin/activate
make install                                         # pinned deps
cp .env.example .env                                 # then edit, see below
make data                                            # real job corpus  -> data/processed, data/derived
make seed                                            # personas, clients, job stream, dev set
make ingest                                          # chunk the portfolios -> data/processed/rag_index
```

Then, one per terminal:

```bash
make gigsim            # marketplace simulator        :8081
make api               # FastAPI                      :8000
make mcp-marketplace   # Marketplace MCP over HTTP    :8702
make mcp-portfolio     # Portfolio MCP over HTTP      :8701
make mcp-workspace     # Workspace MCP over HTTP      :8703
```

`make dev` prints that list. `docker compose -f infra/docker-compose.yml up` is the
one-command alternative.

### The .env trap that costs the most time

**`GIGSIM_API_TOKEN` must match on both sides, and the GigSim server does not read
`.env`.** The server falls back to its built-in default (`gigsim-dev-token`) while any
client process that imports `agent.*` loads `.env` and sends whatever is in there. Leave
the placeholder in and every client call returns 401 while `/health` looks perfectly fine
— which reads like a broken client rather than a config mismatch.

`.env.example` ships `GIGSIM_API_TOKEN=gigsim-dev-token` for exactly this reason. If you
change it, export the same value before starting the server.

### Minimum viable .env

| Variable | Why |
|---|---|
| `GIGSIM_API_TOKEN` | must match the server (above) |
| `GROQ_API_KEY` | any model work. `OPENAI_API_KEY` also accepted |
| `MCP_TOKENS` | only for MCP over HTTP; stdio needs none |
| `TOKEN_ENCRYPTION_KEY` | only for live Upwork mode |

Everything runs without a model key: the system drops to offline mode, returns canned
structures, and stamps `offline: true` on every cost report. Useful for exercising the
graph; never report a number produced that way.

## Checking what you have

```bash
make persistence     # which checkpointer, and whether it survives a restart
make graphs          # both graphs, rendered from the compiled code
make ingest-stats    # what is indexed per persona
curl localhost:8081/health
```

`make persistence` is the one worth running before a demo. It prints
`survives_restart: false` for the in-memory checkpointer, which is the difference between
a working crash-resume demo and an embarrassing one.

## Scheduling (E1)

Scouting and long agent runs belong in the worker, not in a web request. Saved searches
poll on an interval, and the interval is a deliberate choice rather than a default:

| Mode | Interval | Reasoning |
|---|---|---|
| GigSim | 5 minutes | Local, free, rate-limited by us at 20/min against its 30/min |
| **Upwork live** | **no more often than every 3 hours per saved search**, and no more than one read every 3 seconds within a run | See below |

The live figure is the one that needs defending. Upwork's own guidance treats scraping and
unapproved bot traffic as a violation, and a freelancer checking saved searches by hand
would look at them a few times a day. Three hours is slower than that, so our traffic is
quieter than the human it replaces. `UPWORK_MIN_INTERVAL_SEC` (default 3) paces individual
reads inside a run, enforced in the adapter rather than left to a caller.

Jobs are processed idempotently: one thread per job id, so a worker restart re-runs a
thread rather than creating a second one, and the idempotency key stops a re-run
re-submitting.

## Failure modes and what the user sees

| Symptom | Cause | What to do |
|---|---|---|
| Every client call 401, `/health` fine | `GIGSIM_API_TOKEN` mismatch | Set it to `gigsim-dev-token` in `.env` |
| `MARKETPLACE_UNAVAILABLE` in a tool error | GigSim down or 5xx | UI shows the degraded banner, work queues. Restart `make gigsim` |
| Thread stuck in `needs_attention` with "in flight" | process died mid-submit | **Do not retry.** Check the pipeline board for whether the proposal exists. See below |
| `DAILY_QUOTA_EXHAUSTED` | provider's per-day token cap | Wait for reset, or switch `models.profile` to a profile on a different model |
| 429 "request too large ... OTPM" | model's output-per-minute cap below `max_output_tokens` | Set `max_output_tokens` on that profile |
| Retrieval returns nothing sensible | wrong embedding backend | `make rag-eval` prints the backend; `lsa` is a dev fallback, not spec-compliant |
| `NotImplementedError: SqliteSaver does not support async` | sync saver with an async graph | Use `acheckpointer()`, not `checkpointer()` |
| Every job skips with 0% coverage | a fan-out node reading another branch's output | See ARCHITECTURE.md; the Fit Matcher must derive its own requirements |

### A thread stuck "in flight"

This is the deliberate behaviour, not a bug. The idempotency key is claimed *before* the
marketplace write, so a crash in between leaves the claim unresolved. At that moment
nobody knows whether the write landed, so the retry is refused.

To resolve it:

1. Look up the job on the pipeline board, or ask GigSim directly:
   `curl -H "Authorization: Bearer $GIGSIM_API_TOKEN" localhost:8081/api/v1/proposals`
2. If the proposal exists, the write landed. Mark the thread submitted.
3. If it does not, the write never happened: release the key and re-approve.

```python
from mcp_servers._common.idempotency import IdempotencyStore
IdempotencyStore().release("<the key from the thread state>")
```

A stale claim ages out after `MCP_IDEMPOTENCY_STALE_SEC` (120s) and then allows one more
attempt, so this is only needed when the window has not passed.

## Demonstrations

```bash
make demo           # one grounded proposal, stops at approval, never submits
make crash-resume   # C6: kill mid-approval, restart, approve, submitted exactly once
                    #     needs `make gigsim` running first
```

`make crash-resume` runs three separate processes and checks seven invariants. It writes
`evals/results/c6_crash_resume.json`.

### The end-to-end suite

```bash
make e2e            # builds the frontend, then runs the Playwright journeys
make e2e-report     # the HTML report from the last run
```

`make e2e` starts GigSim, the API and Next.js itself, on ports 8481, 8400 and 3400 rather
than the defaults. That is deliberate: with `reuseExistingServer` the suite would adopt
whatever was already listening, and on a machine running a second Next.js project on :3000
every spec failed at sign-in against that other app's landing page. Each run also gets its
own SQLite checkpoint file, because sharing one between runs meant an approved thread
replayed as finished and the approval specs quietly skipped.

Six of the seven specs pass. The seventh, the evidence-hover check, skips when the model is
offline: a placeholder draft makes no claims, so there is nothing to hover. Run it with a
provider key to see it execute.

## Evaluation

```bash
make eval-b             # ingest, retrieval, decisions, verification, ablation, table
make rag-eval           # retrieval metrics per variant
make decision-eval      # bid/skip F1, scam P/R, pricing, injection defence
make verify-eval        # claim verification against labelled probes (offline)
make ablation           # R&D question B component contributions
make agency-eval        # X1 routing accuracy and coverage gained
make budget-report      # E5 declared budgets against the recorded traces
make judge-calibration  # refuses until hand_check_sample.csv has human labels
make agent-comparison   # R&D question C, needs a provider key and quota
make tool-study         # R&D question A
make results-table      # assemble the table from committed result files
```

`make rag-eval` takes about forty minutes, almost all of it V3: the cross-encoder is a
forward pass per candidate and runs on CPU. Everything else here finishes in under a
minute. `RERANK_BACKEND=lexical make rag-eval` swaps the cross-encoder for the
deterministic stand-in and takes two minutes, with V3's row labelled accordingly.

`make verify-eval`, `make agency-eval` and `make budget-report` need no provider key.
`make budget-report` reads `data/audit/traces/` and excludes runs made with the model
offline, so it reports nothing until a real run has happened.

### Provider quotas, which will bite

Groq's free tier meters **tokens per day, per model**, at 200k. That is enough for one
tool-granularity study (138k) or roughly thirteen jobs of the three-arm agent comparison,
not both in a day. The client distinguishes a per-minute 429 (waits and retries) from a
per-day one (fails immediately with `DAILY_QUOTA_EXHAUSTED`, because retrying a daily cap
just turns one clear failure into ten confusing ones).

Three open profiles exist so a spent quota is not a blocked afternoon:

| Profile | Models |
|---|---|
| `open` | gpt-oss-20b / gpt-oss-120b |
| `open_120b` | gpt-oss-120b for every node |
| `open_alt` | qwen3.8-27b (output capped at 900) |

Switch with `MODEL_PROFILE=open_120b make agent-comparison`.

## Live Upwork mode

1. `pip install -e '.[live]'` for the encryption dependency.
2. Generate a key:
   `python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"`
   into `TOKEN_ENCRYPTION_KEY`. A missing key raises rather than storing tokens in
   plaintext.
3. Connect the account in Settings to complete OAuth. Tokens are stored encrypted per
   user and never logged.
4. Set `marketplace.mode: upwork_live` in `agent/config.yaml`.

What changes in the code: nothing above `mcp_servers/marketplace/adapters/`. What live
mode does differently, enforced in the adapter rather than in a prompt:

- every write creates a **draft**; the user confirms it on Upwork. There is no code path
  in the adapter that completes a submission;
- reads are paced at `UPWORK_MIN_INTERVAL_SEC`;
- no browser automation, no cookies, no scraping, ever.

`get_rate_stats` reports `available: false` in live mode because Upwork publishes no rate
statistics. The Pricing Strategist falls back to the user's floor and the client's spend
history and says so in the rationale, rather than pricing from a number we invented.

## Logs and audit

| What | Where |
|---|---|
| MCP tool calls, append-only JSONL | `data/audit/mcp.jsonl` |
| Idempotency claims | `data/audit/idempotency.sqlite3` |
| Checkpoints (sqlite backend) | `data/audit/checkpoints.sqlite3` |
| Encrypted OAuth tokens | `data/audit/oauth_tokens.sqlite3` |
| Workspace holds and email drafts | `data/audit/workspace.sqlite3` |

All of `data/audit/` is git-ignored: it is runtime state, and one file holds secrets.

Emails, phone numbers, card numbers and chat handles are masked before anything is
written, and keys named like secrets are dropped entirely rather than masked. A trace id
flows from the UI request through the API, the graph and every MCP server into each audit
row, so one id reconstructs a whole run.
