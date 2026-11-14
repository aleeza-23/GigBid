# ADR 0002: MCP tools sit above the marketplace port, not beside it

**Status:** accepted · **Date:** 2026-11-14

## Context

Two requirements pull in different directions. R1 wants one internal interface with two
adapters switched by config. A2 wants an MCP server wrapping the marketplace with schemas,
annotations, idempotency, auth and an audit log. It is not obvious which wraps which, and
getting it backwards is hard to undo later.

Three layouts were possible:

1. **MCP server per adapter.** Two servers, one for GigSim and one for Upwork, with the
   graph choosing between them.
2. **Port above MCP.** The port speaks to MCP clients, which speak to marketplaces.
3. **MCP above the port.** One MCP server; the port beneath it picks the adapter.

## Decision

Layout 3:

```
nodes -> MCP marketplace server -> MarketplacePort -> GigSim | Upwork live
```

## Rationale

Layout 1 fails R1 outright: the graph would know which marketplace it was using, and every
node would need both code paths.

Layout 2 puts the schemas, idempotency and audit log *below* the switch, so each adapter
would need its own copy. The audit log is the clearest argument against it — "who spent
what, under which trace id" should be recorded once, in the place every write passes
through, not twice in two adapters that can drift.

Layout 3 puts each concern where there is exactly one of it:

| Concern | Layer | Why there |
|---|---|---|
| Schemas, annotations, tool descriptions | MCP server | What the agent sees; one definition |
| Auth, rate limiting, audit log | MCP server (HTTP middleware) | Every write passes through; a new tool inherits all three |
| Idempotency claim/complete | MCP server | Must stop a retry before it reaches *any* marketplace |
| Retries, backoff, rejected vs unavailable | adapter | Protocol-specific; GigSim's 429 is not Upwork's |
| Which marketplace | port | One reader of `marketplace.mode` in the system |

## Consequences

Switching to live mode changes `marketplace.mode` in `agent/config.yaml` and nothing above
`adapters/`. That is the answer to viva question 7, and
`test_both_adapters_satisfy_one_interface` fails if the two adapters drift apart.

The cost is one extra hop, and a dict crossing the MCP boundary that was a Pydantic model
on either side. `_as_proposal()` absorbs that: tools accept either a validated model (how
FastMCP delivers it) or a plain dict (how a direct Python caller does), so the constraints
hold on both paths. Worth it for having one audit log instead of two.

The adapters live under `mcp_servers/marketplace/adapters/` because that is where the
required directory structure puts them, with `base.py` at the bottom of the dependency
graph so the port above can import them without a cycle.
