# AI usage disclosure

## Tool

Claude Code (Anthropic) was used as a coding assistant throughout this project, in the
VS Code terminal. No other AI coding tool was used. Model-generated text inside the
running system (proposals, critiques, intent classification) comes from Groq-hosted
open-weight models, which `agent/config.yaml` lists per agent; that is part of the
submitted system rather than part of how it was written.

## What it was used for

- Writing first drafts of most Python modules, including the MCP servers, the retrieval
  variants, the LangGraph nodes and graphs, the FastAPI routers and the evaluation
  runners.
- Writing the Next.js frontend and the Playwright specs.
- Writing the documentation in `docs/`, this file included.
- Diagnosing failures: most of the bugs listed below were found by running the code,
  reading the output and asking it to explain a number that looked wrong.

Design decisions, the choice of what to measure, and every judgement about whether a
result was believable were mine. Several of the assistant's structural suggestions were
rejected, and the ones below were wrong in ways that would have invalidated results if
they had shipped.

## Incorrect suggestions found and corrected

**The retrieval metric scored 76% of the corpus as a miss.** `_dedupe_by_project` in
`evals/runners/rag_eval.py` attributed a hit to a portfolio project through
`EvidenceItem.project_id`. Of roughly 197 items per persona, 150 are past proposals,
which carry no `project_id` of their own and instead cite the projects they drew on in
`cited_project_ids`. Every variant's Recall@5 was therefore capped near 0.2 by the metric
and not by the retriever, and the first full run reported V3 at 0.129 while the ablation
reported a comparable pipeline at 0.563. Two tables that disagreed by a factor of four is
what exposed it. A hit is now attributed to its own project, or, lacking one, to the
projects it cites, and Recall@5 for the same configurations moved to 0.45-0.64.

**The parallel fan-out was decorative.** `agent/nodes/fit_matcher.py` read `job_spec`,
which the Job Analyst writes in a sibling branch of the same `Send` fan-out. The Analyst
had not finished when the Fit Matcher ran, so `job_spec` was empty and coverage came back
as 0% on every job without raising anything. The Fit Matcher now derives its own
requirements from the post with the same rule-based `decompose` that V1 uses.

**Coverage used a score threshold that could not survive a change of variant.** The
suggested check was `score >= 2.0`, which is sensible for BM25 scores and meaningless for
reciprocal rank fusion, whose scores run from about 0.01 to 0.05. Switching the configured
variant silently marked every requirement uncovered. Coverage now comes from V8's
sufficiency grading, which is scale-free.

**The V0 baseline used different embeddings from the variants it was the baseline for.**
`rag/variants/v0_naive.py` did not pass the retrieval config through, so it fell back to
the LSA embedder while V1 and V2 used BGE. Every "points over V0" figure was comparing two
things at once. V0 now takes the same config, and the reported deltas are a comparison of
pipeline shape at a fixed embedding backend.

**Tool input constraints were declared where FastMCP does not read them.** Minimum and
maximum values were written into the `tools.py` metadata dictionaries, but FastMCP builds
its schema from the registered function signature, so none of them were enforced and the
contract tests passed anyway. The constraints are now shared `Annotated` aliases on the
parameters themselves.

**The Next.js rewrite was resolved at build time.** The API address reached the frontend
through a `rewrites()` entry reading `process.env.API_BASE_URL`. Next evaluates `rewrites()`
during `next build` and bakes the destination into the route manifest, so setting
`API_BASE_URL` at start-up did nothing and requests went to the compiled-in `:8000`. In the
e2e run the proxy failure surfaced as a 500, and the sign-in test reported
"Internal Server Error" for a wrong password, which looks like an authentication bug and is
not one. Requests now go through `apps/web/src/app/api/[...path]/route.ts`, which resolves
the target per request.

**A synchronous checkpointer was paired with asynchronous graphs.** `SqliteSaver` and
`PostgresSaver` raise `NotImplementedError` from every async method. Both graphs run
through `ainvoke`, so this worked for as long as the configured backend was in-memory and
broke at the moment durability was turned on, which is the moment it matters.
`agent/checkpointer.py` now provides `acheckpointer()` alongside the sync factory.

## Responsibility

I have read every file in this repository, I can explain what each part does and why, and
I am responsible for all of it, including anything an assistant drafted.
