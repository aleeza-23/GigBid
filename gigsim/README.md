# GigSim

A stand-in freelance marketplace for developing and grading GigBid. No simulator shipped
with the assignment, so this one is ours, built over the seed data in `data/gigsim_seed/`.

```bash
make gigsim                 # http://localhost:8081, interactive docs at /docs
curl localhost:8081/health
```

Every route except `/health` needs `Authorization: Bearer $GIGSIM_API_TOKEN`.

## What it simulates

**Clients who decide.** Each of the 1,554 clients carries preferences the agent never
sees: the skills they actually care about, what they will pay, whether a stock opening
annoys them, how much they will read. `POST /admin/tick` scores every pending proposal
against them and returns ignored, viewed, replied, interview or hired. That is where
GigBid's reply rate and hire rate come from, rather than from an LLM grading its own work.

Scam accounts reply to everyone and hire nobody, which gives the Risk Assessor something
real to be wrong about.

**Connects.** Each job has a cost. Submitting spends it, and a freelancer's weekly budget
is enforced, so a runaway agent runs out of money instead of bidding on everything.

**Pushback.** A token bucket per caller returns 429 with `Retry-After` above
`GIGSIM_RATE_LIMIT_PER_MIN`. Set `GIGSIM_ERROR_RATE=0.05` to have one read in twenty fail
with a 503, which is how you exercise the degraded-mode banner before the viva asks for it.

## What it hides

The simulator draws a hard line between what a marketplace shows you and what it knows.
Crossing it would make several metrics meaningless, so it is enforced in `store.py` and
tested in `tests/integration/test_gigsim.py`:

| Hidden | Why |
|---|---|
| `client.hidden_preferences` | Otherwise the Proposal Writer writes to the answer key |
| `client.is_scam_account` | The Risk Assessor has to infer scams from history and post text |
| `job.duplicate_of` | The Scout has to detect reposts itself |

Gold labels (`job_kind`, `is_scam`, `injection_goal`, bid or skip) live only in
`evals/datasets/dev_set.jsonl` and never travel through the API.

## Determinism

Client decisions are a pure function of the client, the job and the proposal text, keyed
by `GIGSIM_SEED`. The same proposal always gets the same outcome, so an evaluation run is
reproducible and an ablation measures the change you made rather than the weather.
`POST /admin/reset` returns the marketplace to its starting state; seed data is never
mutated.

Set `GIGSIM_CLIENT_ENGINE=llm` to have a model decide instead, against any
OpenAI-compatible endpoint. It falls back to the rule engine if the model is unreachable,
so a run never dies because Ollama is not up.

## API

| Method | Path | Notes |
|---|---|---|
| GET | `/health` | Open, no token |
| GET | `/api/v1/jobs` | `q`, `skills`, `category`, `country`, `is_hourly`, `min_hourly`, `max_hourly`, `posted_after`, `limit`, `offset` |
| GET | `/api/v1/jobs/{id}` | |
| GET | `/api/v1/jobs/historical` | 1,374 closed jobs, for pricing statistics |
| GET | `/api/v1/clients/{id}` | Public profile |
| GET | `/api/v1/clients/{id}/history` | Open jobs, past jobs, reviews |
| GET | `/api/v1/stats/rates` | Rate benchmarks and price bands by domain or category |
| GET | `/api/v1/freelancers/{id}/connects` | Balance, weekly budget, spent |
| POST | `/api/v1/proposals` | Write. `dry_run: true` previews; otherwise needs `Idempotency-Key` |
| GET | `/api/v1/proposals` | Filter by `freelancer_id`, `status` |
| GET | `/api/v1/threads` | |
| GET | `/api/v1/threads/{id}/messages` | |
| POST | `/api/v1/messages` | Write. Client answers immediately. Needs `Idempotency-Key` |
| POST | `/api/v1/admin/tick` | Clients decide on pending proposals |
| POST | `/api/v1/admin/reset` | Clean slate |
| GET | `/api/v1/admin/events` | Append-only audit log |

### Writes are idempotent

Every write needs an `Idempotency-Key`. Replaying a key returns the original result with
`idempotent_replay: true` and spends no Connects, which is what lets a worker crash
mid-submission and recover without double-bidding.

```bash
TOKEN=gigsim-dev-token
curl -X POST localhost:8081/api/v1/proposals \
  -H "Authorization: Bearer $TOKEN" \
  -H "Idempotency-Key: job-123-attempt-1" \
  -H "Content-Type: application/json" \
  -d '{"freelancer_id":"ai_ml_engineer","job_id":"upw-...","cover_letter":"...","bid_rate":55}'
```

### Dry runs

`dry_run: true` returns the exact payload that would be sent, the Connects cost, the
balance afterwards and any warnings, and changes nothing. This is what the approval
screen shows the user before they commit.

## Data

Jobs and client histories are built from three Kaggle datasets; personas, preferences,
scams and injected posts are synthetic. Every record carries `data_origin`, so real and
generated data stay distinguishable. Provenance, licensing and limitations go in
`docs/DATA_CARD.md`.

## Tests

```bash
pytest tests/integration/test_gigsim.py tests/integration/test_marketplace_port.py -q
```

The port tests run the adapter against this app in-process over an ASGI transport, so
they need no running server.
