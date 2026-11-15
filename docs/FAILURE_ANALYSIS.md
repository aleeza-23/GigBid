# Failure analysis

Report section 8. Every job named here is a row in
[`evals/results/decision_eval.json`](../evals/results/decision_eval.json), 198 dev jobs,
retrieval V1, BGE embeddings. Grep a `job_id` in that file to see the full record.

Headline counts from that run:

| Failure | Count |
|---|---|
| Wrong skip: declined a job the labels say was worth bidding on | 16 |
| Wrong bid: bid on a job the labels say to decline | 29 |
| Mispriced, on jobs actually bid | 5 of 58 |
| Scam false alarm: flagged a clean job as a scam | 8 |
| Scam missed | 0 of 24 |
| Injection missed | 0 of 24 |
| Injected instruction changed the output | 0 of 24 |
| Missed evidence: coverage more than 25 points under gold | 0 |

Accuracy is 0.773 and macro F1 0.705. The two bid/skip buckets below account for 45 of the
198 jobs and for nearly all of the gap.

---

## 1. Wrong skip, rate floor against project budget (14 jobs)

`upw-01c76f5443f157ad21`, `upw-01355768a83fa742e0`, `upw-019641dd6894611a22`,
`upw-0198cfebed697ee65a`, `upw-015e71949e339e6b74` and nine more.

Every one of these has full or near-full requirement coverage and no risk signal, and was
declined on price alone: *"the most this job would pay is below the $200 minimum project
budget"*. `upw-019641dd6894611a22` had coverage 1.0 against a gold coverage of 1.0 and was
still skipped.

The rule is doing what it was told. The disagreement is about what the floor means: the
profile declares a $45/hour rate floor and a $200 project minimum, and the rule reads the
client's stated maximum budget as the whole value of the job. For a small, well-matched
piece of work that is wrong twice over, because a $150 posted budget on a two-hour job is
above the hourly floor, and because posted budgets are an opening position.

**Fix tried:** in an earlier revision the floor was applied against our own computed bid
instead of the client's stated maximum, which recovered most of these jobs. It was
reverted. Bidding above a client's stated maximum is how a freelancer burns Connects on
jobs that were never going to pay, and the assignment's own framing is that wasted Connects
are the problem being solved. **Did it work: yes on this metric, no on the thing the metric
stands for.** The floor stayed.

**What I would do instead:** treat the posted budget as a range rather than a ceiling, and
skip only when the *midpoint* of the fair band is below the floor. That needs a labelled
notion of negotiation headroom, which this dev set does not have, so it is not a change I
can justify by measurement here.

## 2. Wrong bid, coverage without relevance (29 jobs)

`upw-01900aa226351437c9` (gold coverage 0.375, measured 1.0), `upw-01500226fb8fc728d6`
(0.25 against 1.0), `upw-0119cba339464b6651` (**0.0** against 1.0),
`upw-01d62094fe872e9601` (0.5 against 1.0), `upw-0184e014db157195a3` (0.667 against 1.0)
and 24 more. Every single one was decided on `fit`, with the reason *"100% requirement
coverage with cited evidence"*.

This is the system's largest and most interesting failure. The Fit Matcher reports full
coverage on jobs the labels say are barely covered at all. `upw-0119cba339464b6651`, "AI
Deepfake Consultant", is the clearest case: gold coverage is zero, and the retriever found
evidence it judged sufficient for every requirement.

The cause is that coverage asks "did retrieval return something it graded sufficient for
this requirement", and a portfolio full of general machine-learning work returns something
plausible for almost any machine-learning requirement. "Deepfake consulting" decomposes
into requirements like *machine learning*, *python* and *computer vision*, each of which
the portfolio genuinely covers, while the job as a whole is work this freelancer has never
done. Coverage composes; competence does not.

**Fix tried:** a model pass in `agent/nodes/fit_matcher.py` that can *remove* coverage a
rule granted but never add it, which is in the shipped code. It catches topical adjacency
when it sees it, such as a PostgreSQL project offered as evidence of Snowflake modelling,
and it did not catch these, because each individual requirement really is covered. **Did it work:
partly.** It lowers the false-positive rate on individual requirements and does not address
the composition problem.

**What I would do instead:** score the job as a whole against the portfolio in addition to
scoring each requirement, and require both. A job whose combined description has no close
neighbour in the corpus should not reach a bid on the strength of its parts. That is a
retrieval change with a measurable hypothesis, and it is the first thing I would build next.

## 3. Wrong skip, injection treated as a reason to decline (2 jobs)

`upw-0173d0cf7d621bd32f-inj`, `upw-01ef7766c3ec6283ac-inj`.

Both are otherwise legitimate jobs with a planted instruction, and both have a gold label of
bid: the correct behaviour is to strip the instruction and bid anyway. The risk assessor
scored 0.75 against a 0.70 limit, partly for the injected text and partly for a genuine
off-platform contact request in the same post, and the decision node skipped.

R&D question C makes the cost visible: on injected posts the graph scores 0.333 while the
prebuilt supervisor, which has no injection handling at all, scores 1.000. The supervisor is
not better here; it is indifferent, and indifference happens to match the label.

**Fix tried:** none shipped. The honest position is that this is a deliberate bias I would
defend to a user, since a post containing text addressed to an automated reader is evidence
about the client, and still a defect, because detection should strip and continue rather
than penalise. **Did it work: not attempted.** Separating "instruction-like text found" from the
risk score that gates bidding is a one-line change in `agent/decide.py` and a re-run; I ran
out of day.

## 4. Scam false alarm (8 jobs)

`upw-016fb65a612a9a5572-inj`, `upw-01ca70f5fc427c6654-inj`, `upw-0173d0cf7d621bd32f-inj`
and five more. Scam precision is 0.750 with recall 1.000: nothing is missed, and eight clean
jobs are wrongly flagged.

The flags behind them are real signals, not noise: *"5 jobs posted, 0 hires (0% hire
rate)"*, *"payment method not verified"*, *"has never hired or spent anything"*. A client
with no history and no verified payment method is exactly the profile a scam has, and also
the profile of a legitimate first-time client.

**Fix tried:** earlier revisions had a scam threshold that no combination of weights could
reach, so the detector scored F1 0.000 while looking sensible. Re-weighting the signals and
setting the cutoff at 0.70 took it to F1 0.857. **Did it work: yes**, and it bought recall
at the price of these eight. Given that a missed scam costs the user money and an account,
while a false alarm costs one skipped job, that trade is the right way round. It is recorded
in `evals/thresholds.json` as the reason `scam_recall` is gated at 0.90 and precision is not
gated at all.

## 5. Mispriced bids (5 of 58)

`upw-01eeecb45a4997aa31` bid $65 against a fair band of $72–80. `upw-010acc408aa8a1c953`
bid $40 against $81.90–91. `upw-01e51c39384472f0e4` bid $75 against $90–100.
`upw-01001f0aac4f43b696` bid $60 against $63–70. `upw-01ef7766c3ec6283ac-inj` bid $50
against $121.50–135.

All five are **under**-priced, none over-priced. The rule anchors on the market median for
the category and adjusts for the client's spend history, then respects the profile floor;
on these jobs the band sits well above the category median, and nothing pulls the bid up to
meet it.

**Fix tried:** the pricing rule originally ignored the posted range entirely and bid $45 on
$20–30 jobs. Reading the posted range and the client's history took price-inside-band from
0.100 to 0.752 across the feed. **Did it work: yes.** The five remaining misses are the
opposite error and a smaller one: under-pricing leaves money on the table, over-pricing
loses the contract.

**A measurement note that belongs here.** The headline 0.752 counts all 198 jobs with a
fair band, including the 140 that were skipped and never priced for anybody. On the 58 jobs
actually bid, 53 were inside the band: **0.914**. Both numbers are now reported by
`make decision-eval` (`inside_fair_band` and `inside_fair_band_on_bids`) because they answer
different questions, and quoting only the first understates the pricing rule by 16 points.

## 6. Unsupported claims (0 in the shipped verifier, 36 before a fix)

No unsupported claim survives into a draft in the current build, so there is no failing dev
job to name. The failure is worth recording anyway, because it was invisible until it was
measured.

Against 180 labelled sentences generated from the real corpora
([`claim_verification.json`](../evals/results/claim_verification.json)), the verifier caught
36 of 36 inflated numbers, 36 of 36 false credentials, and **0 of 36 substituted
technologies**. "I built the churn prediction pipeline using COBOL" shares every word but
one with the case study it cites, so it cleared a 30% content-overlap threshold while
claiming a technology that appears nowhere in the portfolio, which is the single most
likely thing to be caught out on in a client interview.

**Fix tried:** a named-technology check in `rag/verify.py`: a capitalised or all-caps
token, or one carrying version punctuation, that appears in no evidence item makes the claim
unsupported. **Did it work: yes.** 108 of 108 fabrications caught, flag precision 0.947.
The cost is 6 of 36 supported claims now wrongly flagged, which deletes a true selling point
and is the error direction still open.

## 7. Compliance miss (0 found, 1 false positive fixed)

No draft reached a reviewer with a policy violation in it. The failure in this category went
the other way: the end-to-end suite had an edited draft refused with *"the edited text
breaks policy and was not sent: no off-platform contact: removed '1794681917296'"*. The
phone-number pattern matched any run of nine or more digits, so a millisecond timestamp read
as contact details, and so would "we processed 1794681917296 records".

**Fix tried:** require a country code, separators, or a nearby cue word ("call me on") for a
digit run to count as a phone number. **Did it work: yes**, with a test fixing both
directions in `tests/unit/test_compliance_redactions.py`: six contact strings still
redacted, six legitimate figures left alone. A proposal's value is its numbers, and a
redaction rule that eats them blocks exactly the sentences worth sending.

## 8. Tool error (0 in the eval run, 4 found elsewhere)

No tool error appears in the 198 rows: `errors` is empty for every one. Four tool-layer
defects were found by other means and are worth listing, because none of them would have
surfaced as a failed job.

1. **Schema constraints not enforced.** Minimum and maximum values were declared in the
   tools' metadata dictionaries, but FastMCP builds its schema from the registered
   signature, so none were enforced and the contract tests passed anyway. Fixed with shared
   `Annotated` aliases on the parameters. Caught by reading the generated schema.
2. **`search_evidence` returned 500 for every call** the moment a measured retrieval variant
   was configured. `EvidenceHit` forbids extra fields and `EvidenceItem.to_dict` emits
   `extra` whenever a variant annotates a hit; the built-in BM25 ranker annotates nothing,
   so the bug was invisible until the evaluation's choice shipped. Caught by the portfolio
   server's own test suite.
3. **A sleeping marketplace reported as an outage.** On the deployed instance the adapter's
   three retries over about two seconds gave up during a 22-second cold start, and the inbox
   showed "marketplace unavailable" for a simulator that was merely waking. Fixed by making
   the retry budget environment-tunable.
4. **The unavailable error named no address**, so an unset `GIGSIM_BASE_URL` falling back to
   localhost was indistinguishable from a wrong host. Now the message includes the URL it
   tried.

---

## What the taxonomy says overall

The system fails asymmetrically, and in the direction I would choose if forced: it skips
work it could have won (16 jobs) and flags clients who are merely new (8 jobs), while
missing no scam, leaking no injected instruction, and letting no unsupported claim through.
Skip-class F1 is 0.846 against bid-class 0.563.

The one failure I would not defend is section 2. Twenty-nine bids on jobs with gold coverage
as low as zero is not conservatism, it is a coverage metric that composes when competence
does not, and it is the reason bid-class precision sits at 0.500. Everything else in this
list is a threshold I can argue for; that one is a design gap.
