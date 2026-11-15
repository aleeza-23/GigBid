# The change request

Report section 6.

## What arrived

Nothing. The brief says a platform policy update would be released for GigSim on
2 November 2026, introducing a new rule or cost affecting bidding. No such update reached
this repository. `data/policy_pack/marketplace_policy.md` is still the seven-rule starter
file, and git confirms it: the only commits that touch `data/policy_pack/` are the initial
commit and the baseline checkpoint.

So there is no real edit to report, and inventing one would be worse than saying so. What
the section is actually asking is whether the design absorbs a rule change or fights it, and
that can be answered by running the exercise. I picked three changes of the kind the brief
describes, implemented them, and timed them. The commands and the diffs are below; nothing
here is an estimate.

## Rehearsal B: a new prohibited practice

**The rule.** "Proposals must disclose when they were drafted with AI assistance."

**What it took.** Two lines appended to `data/policy_pack/marketplace_policy.md`. No code.

**Verification.** A draft containing *"I personally wrote every word of this proposal myself
with no tools of any kind"* was put through the Compliance Guard with the real model. It
returned:

```
violations ['I personally wrote every word of this proposal myself with no tools of any kind.']
rules      ['Rule 8: Proposals must disclose AI assistance; presenting AI-drafted text
            as unassisted is a misrepresentation.']
```

It quoted the offending sentence and cited the rule by number, fourteen seconds after the
rule was written, with no code touched and no restart.

**Elapsed: 14 seconds** (10:09:55 to 10:10:09, both timestamps from the terminal).

**Why it was that cheap.** `agent/nodes/compliance.py` reads the policy file at request time
and the compliance prompt interpolates it as `{policy}`. The rules are data the guard is
handed, not logic the guard contains. A rule written in English is enforced in English.

## Rehearsal C: the gap that rehearsal B exposed

Rehearsal B is also where the exercise earned its keep, because the guard reported
`status: pass` while listing that violation. The draft would have gone to the approval
screen with the offending sentence intact.

The cause was in `agent/nodes/compliance.py`:

```python
blocked = bool(violations) and (
    isinstance(parsed, dict) and str(parsed.get("status")) == "blocked"
)
```

A violation only blocked if the model *also* labelled its own verdict `blocked`. On this
draft it labelled it `fixed` and returned the text unchanged, so a detected, quoted, cited
policy breach passed. Detection without enforcement, which is the worst kind of guard
because the trace looks correct.

**The fix.** Remediation decides it, not the model's label: if the text the guard objected
to is still in the cleaned draft, the draft is blocked. Extracted as `is_blocked()` so it
can be tested without a model, since the model's wording varies between runs and a test
that depends on it is not a test. Four cases in
`tests/unit/test_compliance_redactions.py`: still present blocks, removed does not, no
violations never blocks, and a model that says `blocked` is still believed.

**Elapsed: 34 seconds** (10:10:54 to 10:11:28). 32 changed lines in `compliance.py` plus 37
lines of test.

## Rehearsal A: a new cost rule

**The rule.** "Keep a reserve of Connects rather than spending the balance to zero."

I chose this because the obvious cost change, a rise in the per-proposal Connects price,
needs no edit at all: the price comes from the job and
`agent/decide.py` already turns a bid into a `watch` when the week's remaining balance
cannot cover it. A reserve is the nearest rule that genuinely was not implemented.

**What it took.** One threshold in `agent/config.yaml` (`thresholds.connects_reserve: 8`),
16 lines in `agent/decide.py` beside the existing budget guard, and four tests in
`tests/unit/test_decide.py`. One mistake on the way: the new line read `limits.get(...)`
where the surrounding function calls the dict `th`, so the first run raised `NameError` and
three of the four tests failed. Caught by the tests in the same minute.

**Elapsed: 97 seconds** (10:11:34 to 10:13:11), including the failed run and the fix.

**Verification.** `pytest tests/unit/test_decide.py` — 4 passed. A 12-Connect bid against a
16-Connect balance is now a `watch` with `deciding_factor: connects_reserve`; against a
40-Connect balance it still bids; with the reserve set to 0 the previous behaviour is
restored from config alone.

## What the three say about the design

| Rehearsal | Rule | Config / data | Code | Tests | Elapsed |
|---|---|---|---|---|---|
| B | New prohibited practice | 2 lines | none | none needed | 14 s |
| C | Enforce what the guard detects | none | 32 lines | 37 lines | 34 s |
| A | New cost rule (Connects reserve) | 4 lines | 16 lines | 4 tests | 97 s |

Total: 87 changed lines across 5 files, about two and a half minutes, and the full suite
went from 209 to 217 tests, all passing.

The claim the design makes is that rules and thresholds live in configuration rather than
in code, and two of the three rehearsals support it. A new prohibited practice was pure
data. A new cost rule needed a guard, but the *number* is config, so tightening or relaxing
the reserve later is an environment change rather than a deployment.

The third rehearsal is the useful one, and it argues the opposite way. Rule 8 was read,
understood and cited within fourteen seconds, and then not enforced, because enforcement
lived in a conjunction nobody had tested. Configuration made the rule arrive instantly and
did nothing to make it bite. The lesson I would carry into the next change is that a policy
rule needs a test asserting a draft that breaks it cannot pass, written at the same time as
the rule, and that the natural place for it is a fixture in the policy pack rather than a
hand-written unit test.

## Honest limits of this exercise

These are rules I chose, so they are rules the architecture happens to suit. A real change
request can land somewhere the design did not anticipate: a rule about *when* bidding is
allowed would need the worker's schedule and the ledger rather than a threshold, and a rule
requiring a new field on every submission would reach the tool schemas, the snapshot tests
and the marketplace adapters at once. Neither of those is a config edit, and I would not
claim otherwise on the strength of three rehearsals that took two and a half minutes.
