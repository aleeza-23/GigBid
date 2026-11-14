# MCP Tool Inventory

Every capability in GigBid is an MCP tool with a schema, an owner and a test. No agent
calls GigSim, Upwork, a database or a file directly.

Three servers are ours. Two are third-party, listed with a trust rationale at the end.
The authoritative machine-readable version of this table is
[`tests/contract/snapshots/`](../tests/contract/snapshots/), which CI compares on every
run; if this document and those snapshots disagree, the snapshots are right and this
document is stale.

## Layering

```
nodes  ->  MCP tool (per-agent allowlist)  ->  MarketplacePort  ->  GigSim | Upwork live
```

A node receives only the tools its allowlist names
([`agent/mcp_client.py`](../agent/mcp_client.py)). The port is the only place that reads
`marketplace.mode`, so switching marketplaces is a config change.

## Reading the annotation columns

MCP asks two questions that are easy to conflate, and the interesting cases depend on
keeping them apart:

- **destructive** — can this be undone? Spending Connects cannot.
- **idempotent** — does calling it twice change anything the second time? With an
  idempotency key, no.

`submit_proposal` is therefore destructive *and* idempotent. A host should confirm it with
the user (destructive) but may safely retry it after a timeout (idempotent).

`openWorld` is false for the portfolio and workspace, whose answers come from local state,
and true for the marketplace, whose state changes without us.

---

## 1. Portfolio server — read-only

`mcp_servers/portfolio/` · stdio or `:8701` · [server.py](../mcp_servers/portfolio/server.py)

Nothing on this server can change anything, which is why every row reads the same. Built
on the official MCP Python SDK and independent of the rest of GigBid: it reads only
`data/personas/`, so another freelancer can install it on its own (see
[its README](../mcp_servers/portfolio/README.md)).

| Tool | R/W | readOnly | destructive | idempotent | openWorld | Auth | Purpose |
|---|---|---|---|---|---|---|---|
| `search_evidence` | read | ✓ | ✗ | ✓ | ✗ | bearer (HTTP) | Evidence for **one** requirement, with citations, type, date and outcome |
| `get_case_study` | read | ✓ | ✗ | ✓ | ✗ | bearer (HTTP) | One case study in full, plus the repo, code and review for the same project |
| `list_skills` | read | ✓ | ✗ | ✓ | ✗ | bearer (HTTP) | Skills with evidence counts, won-proposal counts and `unbacked_skills` |
| `portfolio_health` | read | ✓ | ✗ | ✓ | ✗ | bearer (HTTP) | Corpus loadable? Drives the degraded banner |

**Resource templates**

| URI | Returns |
|---|---|
| `portfolio://case/{case_id}` | One case study with provenance (required by A1) |
| `portfolio://proposal/{proposal_id}` | One past proposal, its cited evidence, and whether it won (required by A1) |
| `portfolio://project/{project_id}` | Every description of one project at once: case study, README, code, review |
| `portfolio://profile/{persona}` | Rates, floor, availability, declared skills |
| `portfolio://wins/{persona}` | Past proposals that won, newest first |

**Prompts** — `answer_screening_question` (named in A1), `prove_requirement`,
`draft_proposal_paragraph`, `should_i_bid`, `portfolio_gaps`. Each one ends by constraining
the answer to retrieved evidence, because these are installable in hosts that have no
Compliance Guard downstream.

---

## 2. Marketplace server — read and write

`mcp_servers/marketplace/` · stdio or `:8702` · [server.py](../mcp_servers/marketplace/server.py)

| Tool | R/W | readOnly | destructive | idempotent | openWorld | Auth | Purpose |
|---|---|---|---|---|---|---|---|
| `search_jobs` | read | ✓ | ✗ | ✓ | ✓ | bearer + rate limit | Page of open jobs, filtered; `limit` capped at 100 |
| `get_job` | read | ✓ | ✗ | ✓ | ✓ | bearer + rate limit | One job in full, with screening questions and Connects cost |
| `get_client_history` | read | ✓ | ✗ | ✓ | ✓ | bearer + rate limit | Spend, hires, verification, reviews, plus derived `hire_rate` |
| `get_connects_balance` | read | ✓ | ✗ | ✓ | ✓ | bearer + rate limit | Balance and what is left of the weekly budget |
| `get_rate_stats` | read | ✓ | ✗ | ✓ | ✓ | bearer + rate limit | Earned and posted rate percentiles, with sample sizes |
| `list_threads` | read | ✓ | ✗ | ✓ | ✓ | bearer + rate limit | Client conversation threads |
| `list_messages` | read | ✓ | ✗ | ✓ | ✓ | bearer + rate limit | Messages in one thread |
| `marketplace_health` | read | ✓ | ✗ | ✓ | ✓ | bearer + rate limit | Reachable? Distinguishes down from misconfigured |
| `create_proposal_draft` | **dry run** | ✓ | ✗ | ✓ | ✓ | bearer + rate limit | Exactly what would be sent and what it costs. **Spends nothing** |
| `submit_proposal` | **WRITE** | ✗ | **✓** | ✓ | ✓ | bearer + rate limit + **idempotency key** | Submits. **Spends Connects, cannot be undone** |
| `send_message` | **WRITE** | ✗ | ✗ | ✓ | ✓ | bearer + rate limit + **idempotency key** | Sends a message to a client |

### Why `create_proposal_draft` is annotated read-only

It is a write-shaped tool that performs no write. The approval screen renders its output,
so a host that gated it behind a confirmation prompt would make the user approve twice —
once to see the preview, once to accept it. Annotating it honestly as a read avoids that.

### Idempotency

Both writes require a key of at least 8 characters. The flow is claim-then-complete
against SQLite ([`_common/idempotency.py`](../mcp_servers/_common/idempotency.py)):

1. `claim(key)` inserts an `in_flight` row, or reports the key as a replay or in flight.
2. The write runs.
3. `complete(key, result)` stores the result, making later calls replays.

A claim is taken **before** the write, not after, because the dangerous moment is a crash
*during* the write, when there is no result to cache. A key still in flight is refused
rather than retried, so a thread goes to "needs attention" instead of silently spending
twice. A *refusal* releases the key — nothing happened, so a corrected payload may reuse
it; a *timeout* does not, because the write may well have landed.

Proven by `test_one_key_spends_once_however_often_it_is_called` and
`test_a_claim_left_in_flight_blocks_a_retry`.

### Auth, rate limiting, audit

Applied as HTTP middleware in `build_http_app()`, so a tool added next week inherits all
three without its author remembering to.

- **Bearer token** — `MCP_TOKENS` maps tokens to user ids. Unset in production means no
  caller can authenticate, which is the safe direction to fail in.
- **Rate limit** — token bucket, 20/min per user by default against GigSim's 30/min, so
  GigBid is never the caller that trips the marketplace's own limiter. Returns 429 with
  `Retry-After`.
- **Audit log** — append-only JSON Lines at `data/audit/mcp.jsonl`: timestamp, trace id,
  user, server, tool, arguments, result summary, status, duration. Arguments pass through
  PII masking and secret redaction on the way in.

### Trace propagation

An `X-Trace-Id` header is adopted if sent and minted otherwise, carried in a `ContextVar`,
written to every audit row, and echoed back on the response. One id spans UI → API → graph
→ MCP server → audit log (E3).

---

## 3. Workspace server — calendar and email drafts

`mcp_servers/workspace/` · stdio or `:8703` · [server.py](../mcp_servers/workspace/server.py)

Backed by a **local mock** by default, which A3 permits, so the graded path runs for a
reviewer with no Google account. `WORKSPACE_BACKEND=google` moves the same five tools onto
Calendar and Gmail; the Client Liaison sees MCP tools either way.

| Tool | R/W | readOnly | destructive | idempotent | openWorld | Auth | Purpose |
|---|---|---|---|---|---|---|---|
| `free_interview_slots` | read | ✓ | ✗ | ✓ | ✗ | bearer (HTTP) | Open slots, weekdays in working hours, one per day |
| `list_interview_holds` | read | ✓ | ✗ | ✓ | ✗ | bearer (HTTP) | Current tentative holds |
| `workspace_health` | read | ✓ | ✗ | ✓ | ✗ | bearer (HTTP) | Backend usable? |
| `create_interview_hold` | **WRITE** | ✗ | ✗ | **✗** | ✗ | bearer (HTTP) | Tentative hold. **Not idempotent**: calling twice books twice |
| `release_interview_hold` | **WRITE** | ✗ | **✓** | ✓ | ✗ | bearer (HTTP) | Frees a slot. Destructive (undoes state), idempotent (re-release is a no-op) |
| `create_email_draft` | **WRITE** | ✗ | ✗ | ✗ | ✗ | bearer (HTTP) | Creates a draft. **Never sends** |

**This server has no send tool at all.** R4 expressed as an absence: no bug or prompt
injection upstream can cause an email to go out, because no code path exists that would.
Pinned by `test_this_server_has_no_way_to_send_anything`.

Holds are stored in SQLite, not memory, so a tentative hold survives the worker restart
C6 asks us to demonstrate — otherwise the agent would re-offer a slot it had promised.

---

## 4. Third-party servers (A5)

Neither is ours. Both are opt-in: absent credentials means the server is simply not
registered, rather than every run failing on a connection error.

| Server | URL | Transport | Used for | Trust rationale |
|---|---|---|---|---|
| **Upwork (official)** | `https://mcp.upwork.com/mcp` | Streamable HTTP, OAuth 2.1 | Live mode: job search, proposal and message **drafts** | Operated by Upwork itself, so it is the only live access the platform's own automation rules sanction. Every write is a draft the user confirms on Upwork, which bounds the damage any bug of ours can do. |
| **GitHub (official)** | `https://api.githubcopilot.com/mcp/` | Streamable HTTP, PAT | Ingesting repository READMEs and code into the portfolio corpus (B1) | Operated by GitHub. Scoped to a read-only PAT over public repositories, so the worst case is reading something already public. |

### Live-mode safety, enforced in code

In [`adapters/upwork_live.py`](../mcp_servers/marketplace/adapters/upwork_live.py), not in
a prompt:

- **Writes are drafts only.** `submit_proposal` calls Upwork's *draft* tool. There is no
  code path in the adapter that completes a submission.
- **Reads are paced.** Minimum 3s between calls (`UPWORK_MIN_INTERVAL_SEC`), justified in
  [RUNBOOK.md](RUNBOOK.md). Far below a human browsing the site.
- **No browser, ever.** Only the official MCP endpoint. No cookies, no HTML, no scraping.
- **Tokens encrypted per user.** Fernet, keyed from `TOKEN_ENCRYPTION_KEY`. A missing key
  raises rather than falling back to plaintext. `__repr__` never renders a token.
- **Tool names are discovered, not hardcoded.** A third-party server may rename a tool; a
  clear "this marketplace cannot do that" beats a crash.

---

## 5. Per-agent tool allowlists (A6)

From [`agent/mcp_client.py`](../agent/mcp_client.py). Read the list as "what this agent
could do if a job post talked it into trying".

| Agent | Tools |
|---|---|
| Scout | `search_jobs`, `get_job`, `marketplace_health` |
| Job Analyst | *none* — works from the job text already in state |
| Risk Assessor | `get_client_history`, `get_job` |
| Fit Matcher | `search_evidence`, `get_case_study`, `list_skills`, `portfolio_health` |
| Pricing Strategist | `get_rate_stats`, `get_client_history`, `get_connects_balance` |
| Proposal Writer | `search_evidence`, `get_case_study`, `list_skills` |
| Critic | `search_evidence`, `get_case_study` |
| Compliance Guard | *none* — judges text against the policy pack |
| Client Liaison | thread reads, `send_message`, and the five workspace tools |
| Approval node | `create_proposal_draft`, `submit_proposal`, `get_connects_balance` |

**The Writer cannot submit anything.** The capability to spend Connects sits with the
approval node, which runs only after `interrupt()` returned a human decision — not with
the model that drafted the text. A Writer that *could* call `submit_proposal` is one
prompt injection away from doing it, so the tool is removed rather than discouraged.

Pinned by `test_only_the_marketplace_can_write`, which asserts the write surface of the
entire system in one place. A new write tool anywhere fails that test and has to be
justified.

## 6. Failure handling (A6)

| Failure | Behaviour |
|---|---|
| 429 or 5xx from the marketplace | Bounded retries, exponential backoff with jitter, honours `Retry-After` |
| 4xx that is not 429 | **Not** retried. `MarketplaceRejected` surfaces the reason to the user |
| Connection refused, timeout | `MarketplaceUnavailable` after 3 attempts → `MARKETPLACE_UNAVAILABLE:` tool error |
| Marketplace server killed | `marketplace_health` returns `unavailable`; UI shows the degraded banner, work queues |
| A broken RAG variant | `search_evidence` falls back to the built-in BM25 ranker rather than failing the graph |
| One MCP server unreachable | `GigBidMCPClient.unavailable` names it; the other servers keep working |

The rejected/unavailable distinction is the whole reason there are two exception classes.
Collapsing them is how a system ends up either retrying a 404 forever or dropping a job
because of one transient 503.

## 7. Tests

| What | Where |
|---|---|
| Portfolio server through an in-memory MCP client | `mcp_servers/portfolio/tests/` (15 tests) |
| Marketplace server against GigSim in-process | `mcp_servers/marketplace/tests/` (15 tests) |
| Workspace server | `mcp_servers/workspace/tests/` (11 tests) |
| Schema snapshots + write-surface invariants | `tests/contract/` (7 tests) |
| The port, both adapters, retry and idempotency behaviour | `tests/integration/test_marketplace_port.py` (14 tests) |

All of them go through the MCP protocol rather than calling Python functions directly,
because a server whose functions work but whose schema is malformed is broken for every
host, and only a protocol-level test notices.

MCP Inspector screenshots: `make inspect-portfolio`, `make inspect-marketplace` →
[docs/images/](images/).
