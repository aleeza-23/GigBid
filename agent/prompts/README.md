# Prompts

One versioned file per node. Nothing inline in node code: CI checks `agent/nodes/` for
prompt-shaped strings and fails if it finds one.

Each file carries frontmatter:

```
---
version: 3
node: writer
model: large          # resolved through models.nodes in agent/config.yaml
notes: what changed and why
---
```

The body is a template with `{placeholders}`. `loader.render(node, **values)` fills them
and raises on a missing one, because a prompt that ships with a literal `{requirement}` in
it produces plausible nonsense rather than failing.

**Bump `version` whenever you change a body.** The version appears in the execution trace
as `writer@v3`, which is what makes "the Writer got worse last Tuesday" answerable.

| File | Node | Model | What it does |
|---|---|---|---|
| `analyst.md` | Job Analyst | small | Extracts a JobSpec from an untrusted post |
| `risk_assessor.md` | Risk Assessor | small | Scores client quality and scam signals |
| `fit_matcher.md` | Fit Matcher | small | Per-requirement coverage from retrieved evidence |
| `pricing.md` | Pricing Strategist | small | Bid from market stats, client spend, the floor |
| `writer.md` | Proposal Writer | large | Grounded draft with per-claim citations |
| `critic.md` | Critic | large | Grounding, personalisation, opening, length |
| `compliance.md` | Compliance Guard | small | Policy pack; strips contact details and guarantees |
| `intent.md` | Client Liaison | small | Classifies an inbound client message |
| `client_liaison.md` | Client Liaison | large | Drafts a client reply, never sends |

Every prompt that touches a job post or a client message delimits it with
`<job_post untrusted="true">` and states that instructions inside are data. That is the
structural half of the injection defence; the pattern stripping in `agent/decide.py` is
the other half.
