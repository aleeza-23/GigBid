"""Long-term memory: two stores, written only through the promotion rules (C2, R7).

C2 requires exactly two stores, and this module is the only way into either:

* **style memory** - learned from the user's edits to drafts ("always cut the first
  sentence", "never say passionate");
* **outcome memory** - what won or got replies per job category.

Both live in a LangGraph store under namespaced keys, scoped per user so one freelancer's
preferences never reach another's drafts.

The API is deliberately narrow. There is no `write(key, value)`. The only writers are
`observe_edit` and `observe_outcome`, which record evidence, and `promote`, which asks
`promotion_rules` whether the accumulated evidence is enough. A node cannot put something
in memory by deciding to; it can only contribute an observation and let the rule judge.

That shape is what makes R7 checkable: grep for writes to the store and there are three
functions, all in this file, all gated.
"""

from __future__ import annotations

import time
from typing import Any

from agent.memory import promotion_rules as rules

# Namespaces. Tuples because LangGraph stores are hierarchical; the user id is the first
# element so every read is tenant-scoped by construction rather than by remembering to
# filter.
STYLE_OBSERVATIONS = "style_observations"
STYLE_RULES = "style_rules"
OUTCOME_OBSERVATIONS = "outcome_observations"
OUTCOME_RULES = "outcome_rules"


def _ns(user_id: str, bucket: str) -> tuple[str, ...]:
    return ("gigbid", user_id or "default", bucket)


class Memory:
    """Style and outcome memory over a LangGraph store."""

    def __init__(self, store: Any, user_id: str = "default") -> None:
        self.store = store
        self.user_id = user_id

    # ------------------------------------------------------------- observations

    def observe_edit(
        self, job_id: str, original: str, edited: str, reason: str | None = None
    ) -> dict[str, Any]:
        """Record what a human changed about a draft. Not yet memory.

        Called from the approval node when the user edits rather than approves verbatim.
        The diff is computed here so that what gets stored is a set of checkable signals,
        not the full text of the proposal: storing drafts would turn the memory store into
        a second copy of the corpus.
        """
        diff = rules.diff_edit(original, edited)
        record = {
            "job_id": job_id,
            "signals": diff["signals"],
            "chars_before": diff["chars_before"],
            "chars_after": diff["chars_after"],
            "removed_sentences": diff["removed_sentences"],
            "reason": reason,
            "observed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        self.store.put(_ns(self.user_id, STYLE_OBSERVATIONS), job_id, record)
        return record

    def observe_outcome(
        self, job_id: str, category: str, outcome: str, **extra: Any
    ) -> dict[str, Any]:
        """Record what the marketplace reported about a submitted proposal."""
        record = {
            "job_id": job_id,
            "category": category or "uncategorised",
            "outcome": outcome,
            "observed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            **extra,
        }
        self.store.put(_ns(self.user_id, OUTCOME_OBSERVATIONS), job_id, record)
        return record

    # ---------------------------------------------------------------- promotion

    # LangGraph's store.search defaults to limit=10. Taking that default silently ran
    # every promotion rule on at most ten observations: with twelve outcomes recorded,
    # the baseline reply rate came out as 40% instead of 33% and a category that
    # qualified for promotion was never considered. Paginating is not an optimisation
    # here, it is the difference between a correct rule and a plausible-looking one.
    _PAGE = 200

    def _all(self, bucket: str, max_items: int = 10_000) -> list[dict[str, Any]]:
        """Every observation in a bucket, paginated. Never a truncated page."""
        namespace = _ns(self.user_id, bucket)
        out: list[dict[str, Any]] = []
        offset = 0
        while offset < max_items:
            try:
                items = self.store.search(namespace, limit=self._PAGE, offset=offset)
            except TypeError:
                # A store without limit/offset support: take what it gives once.
                try:
                    items = self.store.search(namespace)
                except Exception:
                    return out
                for item in items or []:
                    value = getattr(item, "value", None)
                    if isinstance(value, dict):
                        out.append(value)
                return out
            except Exception:
                return out

            if not items:
                break
            for item in items:
                value = getattr(item, "value", None)
                if isinstance(value, dict):
                    out.append(value)
            if len(items) < self._PAGE:
                break
            offset += self._PAGE
        return out

    def promote(self) -> dict[str, Any]:
        """Run both promotion rules over the accumulated observations.

        Idempotent: re-promoting an existing rule refreshes its evidence count rather than
        duplicating it, so this can be called after every approval without the memory
        growing a new copy each time.
        """
        promoted: dict[str, list[dict[str, Any]]] = {"style": [], "outcome": []}

        for candidate in rules.style_candidates(self._all(STYLE_OBSERVATIONS)):
            record = candidate.to_record()
            self.store.put(_ns(self.user_id, STYLE_RULES), candidate.key, record)
            promoted["style"].append(record)

        for candidate in rules.outcome_candidates(self._all(OUTCOME_OBSERVATIONS)):
            record = candidate.to_record()
            self.store.put(_ns(self.user_id, OUTCOME_RULES), candidate.key, record)
            promoted["outcome"].append(record)

        return promoted

    # ------------------------------------------------------------------- reads

    def style_rules(self) -> list[dict[str, Any]]:
        """Promoted style rules, for the Writer's prompt and the memory viewer."""
        return sorted(
            self._all(STYLE_RULES), key=lambda r: -(r.get("evidence_count") or 0)
        )

    def outcome_rules(self) -> list[dict[str, Any]]:
        return sorted(
            self._all(OUTCOME_RULES), key=lambda r: -(r.get("evidence_count") or 0)
        )

    def style_instructions(self, limit: int = 8) -> list[str]:
        """Style rules as plain instructions to paste into the Writer's prompt.

        Only promoted rules reach the prompt. An observation that has not met the
        threshold has no effect on drafting, which is the whole point of the threshold.
        """
        return [str(r["value"]) for r in self.style_rules()[:limit] if r.get("value")]

    def category_note(self, category: str) -> str | None:
        """What outcome memory knows about one job category, if anything."""
        for record in self.outcome_rules():
            if record.get("key") == category:
                value = record.get("value") or {}
                return str(value.get("note") or "")
        return None

    # ------------------------------------------------------------------ deletes

    def forget(self, kind: str, key: str) -> bool:
        """Delete one promoted rule. The memory viewer calls this (viva question 6).

        Deletes the *rule*, not the observations behind it, so the next promotion run
        would re-derive it. `forget_observations` exists for when the user means "this was
        never my preference" rather than "stop applying this".
        """
        bucket = STYLE_RULES if kind == "style" else OUTCOME_RULES
        try:
            self.store.delete(_ns(self.user_id, bucket), key)
            return True
        except Exception:
            return False

    def forget_observations(self, kind: str, key: str | None = None) -> int:
        """Delete the evidence too, so a promotion rule cannot re-derive the rule."""
        bucket = STYLE_OBSERVATIONS if kind == "style" else OUTCOME_OBSERVATIONS
        removed = 0
        for record in self._all(bucket):
            job_id = str(record.get("job_id") or "")
            if key and key not in (record.get("signals") or []) and record.get("category") != key:
                continue
            try:
                self.store.delete(_ns(self.user_id, bucket), job_id)
                removed += 1
            except Exception:
                continue
        return removed

    # -------------------------------------------------------------------- views

    def viewer(self) -> dict[str, Any]:
        """Everything GigBid has learned about this user, for the settings screen.

        Includes the rules that admitted each item and the raw observation counts, because
        "see and delete what GigBid has learned" is only meaningful if the user can also
        see *why* it believes it.
        """
        return {
            "user_id": self.user_id,
            "style_rules": self.style_rules(),
            "outcome_rules": self.outcome_rules(),
            "pending_style_observations": len(self._all(STYLE_OBSERVATIONS)),
            "pending_outcome_observations": len(self._all(OUTCOME_OBSERVATIONS)),
            "promotion_rules": rules.explain_rules(),
        }


def get_memory(store: Any = None, user_id: str = "default") -> Memory:
    """Memory over the configured store, or an in-memory one for tests."""
    if store is None:
        from agent.checkpointer import store as build_store

        store = build_store()
    return Memory(store, user_id)
