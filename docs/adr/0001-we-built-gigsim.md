# ADR 0001: We wrote our own GigSim

**Status:** accepted · **Date:** 2026-11-11

## Context

The assignment lists `gigsim/` as PROVIDED, says "do not modify", and expects CI to
enforce that nothing under it changes. The starter kit shipped with no simulator: its
README describes `data/sample/` as "tiny synthetic examples ... not a replacement for the
instructor-provided GigSim data", and no simulator code or seed data was included.

Without a marketplace there is nothing to build Parts A, C or E against: no jobs to scout,
no Connects to spend, no client to reply, and no reply rate or hire rate to report.

## Decision

Write our own GigSim over the seed data in `data/gigsim_seed/`, matching the behaviours the
assignment describes: jobs, clients with histories, Connects, proposals, messages, rate
limits and occasional errors.

It deliberately reproduces the properties the grading depends on:

- **Clients decide.** 1,554 clients carry hidden preferences the agent never sees.
  `POST /admin/tick` scores pending proposals against them and returns ignored, viewed,
  replied, interview or hired. Reply rate and hire rate come from there, not from a model
  grading its own work.
- **A hard visible/hidden boundary.** `hidden_preferences` and `is_scam_account` never
  leave the store. Enforced in `gigsim/store.py`, tested in
  `tests/integration/test_gigsim.py`, and re-tested at the MCP boundary in
  `test_client_history_never_leaks_hidden_preferences`. Without this, scam precision and
  reply rate would measure a system reading the answer key.
- **Pushback.** A per-caller token bucket returns 429 with `Retry-After`;
  `GIGSIM_ERROR_RATE` injects 503s, which is how the degraded-mode banner gets exercised
  before a viva asks for it.
- **Reproducibility.** `GIGSIM_SEED` fixes client decisions and error injection, so an
  evaluation run can be repeated exactly.

## Consequences

**Good.** Everything downstream is testable in-process over an ASGI transport: no server
to start, no sockets, and the real validation and Connects accounting. The whole test suite
runs in about ten seconds.

**Bad, and worth saying plainly.** Our simulator is not the instructor's. If the hidden
test run targets a different GigSim with different routes or payload shapes, the adapter in
`mcp_servers/marketplace/adapters/gigsim.py` is what needs changing. That is one file
behind the marketplace port, which is the main reason the port exists (R1), but it is still
a real risk rather than a theoretical one.

Our client simulator is also almost certainly more forgiving than an LLM-driven one. Reply
rates measured here should be read as relative comparisons between our own variants, not as
absolute numbers comparable to another student's.

**CI.** The "nothing under `gigsim/` changed" rule cannot be checked against an upstream
copy that does not exist, so `.github/workflows/ci.yml` warns on changes to `gigsim/` in a
pull request instead of failing against a non-existent baseline.
