# The reranker made my retrieval worse, and a broken metric hid it for an hour

*Written for: the technical write-up deliverable, a post for LinkedIn, Medium or a blog.
Publish the body below; this header line is for you, not the reader.*

---

I spent a week building a retrieval pipeline for a freelance-proposal agent, and the
component I was most confident about turned out to be the one making it worse. Removing
the cross-encoder reranker raised Recall@5 by 9.8 points. Finding that out took longer
than it should have, because the first evaluation I ran was wrong in a way that looked
plausible.

## The system, briefly

GigBid reads a freelance job feed, decides which jobs are worth bidding on, and drafts
proposals that cite portfolio evidence for every claim. The retrieval problem is specific:
given a job post listing six requirements, find the case studies, repository READMEs,
client reviews and past proposals that prove the freelancer has done each one. The corpus
per persona is about 197 items. Every claim in the draft has to point at an item, and
anything that cannot be supported gets removed before a human sees the draft.

So retrieval quality decides what the agent is allowed to say. I built five variants to
compare: naive dense retrieval, requirement decomposition, hybrid BM25 plus dense with
reciprocal rank fusion, cross-encoder reranking on top of the hybrid candidates, and a
corrective loop that re-queries when evidence looks thin.

My hypothesis, written before running anything: decomposition plus hybrid retrieval would
raise Recall@5 by 15 points over the baseline on jobs with four or more requirements, and
reranking would add precision on top.

## The first run said everything was terrible

The first full evaluation gave Recall@5 figures between 0.09 and 0.24. Low, but the
ordering looked sensible, with decomposition ahead and reranking behind, so I started
writing it up as "retrieval is hard on a small heterogeneous corpus".

Then the ablation disagreed. The ablation removes one component at a time from the full
pipeline and reports what each one contributes, and its baseline arm, the full pipeline on the same
rows through the same metric function, scored 0.563. The variant table said 0.129 for a
comparable configuration. A factor of four apart, from two scripts importing the same
`recall_at_k`.

One of them had to be wrong, and the disagreement is the only reason I looked. A single
table of plausible-looking low numbers would have gone into the report unchallenged.

## The metric was scoring three quarters of the corpus as a miss

The gold labels name portfolio *projects*: `ai-p07`, `ai-p12`. Each retrieved item was
attributed to a project through its `project_id`, and a hit counted if that id was in the
gold set.

Of the ~197 items per persona, 150 are past proposals. A past proposal has no `project_id`
of its own. It has `cited_project_ids`: the projects it argued from. So every proposal retrieved,
76% of the corpus, attributed to nothing and was scored as a miss no matter how good it
was.

This matters beyond the arithmetic. A past proposal citing `ai-p07` is often the best
evidence for `ai-p07`, because it is the text that already convinced a client. The metric
was blind to precisely the items the Writer most wants to quote.

Attributing a hit to its own project, or, lacking one, to the projects it cites, moved
Recall@5 for the same configurations from 0.09–0.24 to 0.45–0.65. Nothing about the
retrieval changed. I had been measuring the join between two id namespaces.

## With the metric fixed, the reranker had nowhere to hide

Here is the corrected table, 142 dev rows with gold evidence, `BAAI/bge-small-en-v1.5`
embeddings:

| Variant | Recall@5 | nDCG@10 | ms/query |
|---|---|---|---|
| V0 naive dense | 0.582 | 0.750 | 754 |
| V1 requirement decomposition | 0.651 | 0.774 | 56 |
| V2 hybrid BM25 + dense RRF | 0.581 | 0.768 | 4 |
| V3 cross-encoder rerank | 0.592 | 0.744 | 11,737 |
| V8 corrective loop | 0.571 | 0.747 | 65 |

Decomposition works: 6.9 points over the baseline for 56 ms. That is the one part of my
hypothesis that survived, though at half the size I predicted.

The cross-encoder spends **11.7 seconds a query** to land 1.0 point above the baseline. On
a six-requirement job that is over a minute of reranking against a declared 25-second
analysis budget. And the ablation, run with a cheaper lexical reranker in the same slot, is
blunter still: removing the reranker raised Recall@5 from 0.563 to 0.661 and nDCG@10 from
0.670 to 0.825. Both rerankers hurt. One of them also costs twelve seconds.

## Why, and the part I had not thought about

The explanation that fits both results is about pool depth rather than model quality.

Reranking is a precision tool for deep candidate pools. The usual shape is a cheap retriever
proposing 100 or 1,000 candidates from millions of documents, after which the
cross-encoder, which reads query and document jointly, reorders the top slice. Its value
comes from judging pairs the cheap retriever could only score independently.

My corpus has 197 items and the hybrid retriever scores all of them. There is no deep pool
and nothing hidden. A reranker here cannot surface anything new. It can only reorder what
is already ranked, and on this data it reorders worse. `bge-reranker-base` is trained on
web and QA passages, where relevance means "answers this question". My items are portfolio
artefacts, where relevance means "this is evidence the freelancer did this work". A case
study about Snowflake modelling and a job post asking for Snowflake modelling are not a
question-answer pair, and the model's notion of relevance is subtly wrong for the task.

The cheap signals are better suited than I expected. Exact library names carry most of the
information: a job asking for Celery wants the project that used Celery, and both a dense
embedding and a cross-encoder blur "Celery" into "task queue". BM25 does not blur it.

## What shipped

The configured variant is decomposition over hybrid retrieval with no reranking. There is
no model in the retrieval path at all, so retrieval cost grows by milliseconds per
requirement instead of seconds, and the 25-second analysis budget holds with room to
spare — measured p95 is 7.6 seconds.

The cross-encoder is still in the repository behind an environment variable, because the
result is corpus-specific. At 10,000 items I would expect it to start earning its place.
At 197 it does not.

## Three things I would do differently

**Make two independent measurements of anything that matters, early.** The bug survived a
full evaluation run and was caught by a disagreement, not by inspection. Had I written only
the variant table, I would have reported numbers a third of their true size and concluded
that retrieval on a small corpus is hopeless.

**Check that the metric's key space matches the data's.** The failure was not in retrieval
or in the metric formula. It was in the join: gold ids were project ids, retrieved ids were
item ids, and 76% of the corpus crossed that boundary through a field the metric never
read. That class of bug produces smooth, believable, wrong numbers.

**Measure latency as early as accuracy.** I treated 11.7 seconds a query as an
implementation detail to optimise later. It is the result. A component that cannot run inside
the latency budget is not a candidate, whatever its accuracy, and knowing that on day one
would have changed the order I built things in.

The negative result is the one I would keep. "We added a cross-encoder and recall improved"
is a sentence I could have written without measuring anything. "We removed the
cross-encoder, recall rose 9.8 points, and here is why that follows from the corpus size"
is the one that changed what shipped.

---

*Numbers from `evals/results/rag_eval.json` and `evals/results/ablation.json` in the
project repository; the evaluation runners are in `evals/runners/`.*
