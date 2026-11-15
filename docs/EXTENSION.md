# X1 Agency mode

One GigBid serving a small team instead of one freelancer. Before any analysis runs, a
routing node asks which member's portfolio actually has evidence for the job's
requirements, and the rest of the thread proceeds as that member: their corpus is what the
Fit Matcher retrieves from, their rate floor is what pricing respects, their promoted style
rules are what the Writer follows.

Off by default. `agency.enabled` in `agent/config.yaml` turns it on, with the member list
and the two rules below. The single-freelancer product is what Parts A to E are measured
on, and routing is only meaningful when every member has a real portfolio behind them.

## Design

```
scout -> route_member -> fan_out ==> [analyst | risk_assessor | fit_matcher] -> ...
```

`route_member` sits between the scout and the fan-out, in `agent/nodes/agency.py`. It has
to be there rather than inside the fan-out: every downstream node reads `persona`, so
choosing the member in a parallel branch would race the branch that retrieves from that
member's portfolio. This is the same mistake that made the fan-out decorative in an
earlier version, and it is easier to make the second time.

**Routing is by retrieval, not by a model.** The job is decomposed into requirements once,
and each member is scored on how many of those requirements their portfolio can support,
using the same V8 sufficiency grading the bid decision already runs on. Two reasons:

1. It reuses evidence the system has to gather anyway, so routing costs one retrieval pass
   per member and no model call. At three members that is about 5 seconds per job with the
   configured retrieval, and it would be minutes with the cross-encoder, which is one more
   consequence of the retrieval choice in Part B.
2. It leaves a number a person can check. The trace says "routed to the data analyst, 5 of
   6 requirements covered from 3 projects, next best 2 of 6". A model asked who should take
   a job answers from the post's wording, which sends every job mentioning Python to
   whichever profile says Python most often rather than to the member with matching work.

**Two rules stop it pretending to decide.**

*A win needs a margin.* Two members within `agency.min_margin` (0.15) of each other is not
a routing decision. The job goes to the configured default member, the thread is flagged
in `degraded`, and a person sees why. At this corpus size a one-requirement difference is
noise, and routing on noise produces a proposal drafted from the wrong portfolio.

*Nobody clearing `agency.min_coverage` (0.34) means the team cannot do this job.* The
thread falls back to the default member, whose low coverage the decision node then sees and
skips on. The alternative — routing to the best of three bad options — makes a skip look
like a choice.

## What it measures

`make agency-eval`, written to `evals/results/x1_agency_routing.json`. The dev set labels
every job with the persona it was written for, so the router's choice can be compared with
a label it never sees.

Over the first 30 dev jobs, hybrid retrieval, three members:

| | |
|---|---|
| Routed (committed to a member) | 24 of 30 |
| Declined to route | 6 of 30 |
| Routing accuracy when committed | 0.917 |
| Routing accuracy over all jobs | 0.733 |
| Mean coverage, routed | 0.897 |
| Mean coverage, best fixed member | 0.878 (`ai_ml_engineer`) |
| Coverage gained | +1.95 points |
| Cost | 3 retrieval passes per job, 5.2 s/job, no model call |

Both wrong choices sent an AI/ML job elsewhere, one to the web developer and one to the
data analyst. Nothing was routed *into* AI/ML wrongly, which fits how the routing errs:
it follows evidence, and a job whose requirements are generic enough ("Python", "API")
finds support in more than one portfolio.

## Is it worth it?

On this dev set, barely. Routing reaches 0.897 mean coverage against 0.878 for simply
always using the AI/ML engineer, a gain of 1.95 points for three times the retrieval. The
reason is the sample: the dev feed is dominated by AI/ML work, so the best fixed member is
already close to the ceiling, and the measurement says so rather than flattering the
feature.

The number that does justify it is the 0.917 accuracy when the router commits, together
with the six jobs it declined. On a feed with a real mix of work — which is what an agency
has, and what the first 30 rows of this dev set are not — the fixed-member baseline falls
to whatever fraction of the feed suits that one member, while routing keeps committing
correctly. The honest claim is therefore conditional: agency mode pays off in proportion to
how mixed the feed is, it costs one retrieval pass per member, and on a single-domain feed
it is overhead. The full-dev-set run in the results file is the one to read for the mixed
case, since the first 30 rows are all in-domain AI/ML jobs.

What would change the verdict rather than the numbers: giving each member a portfolio of
genuinely different work, which this synthetic corpus only partly does, and measuring reply
rate per member instead of coverage. GigSim does not model client replies, so that
measurement does not exist here and is not estimated.

## Where the code is

| | |
|---|---|
| Routing node, scoring and the two rules | `agent/nodes/agency.py` |
| Graph wiring | `agent/graphs/bid_graph.py` (`route_member`) |
| Configuration | `agent/config.yaml`, `agency:` block |
| Evaluation | `evals/runners/agency_eval.py` |
| Tests | `tests/unit/test_agency.py` |
