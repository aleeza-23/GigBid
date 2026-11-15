"""Model access, routed by node and chosen entirely in config.

Three rules from the assignment are enforced here:

* **Every node's model is set in config, never in code.** `chat("writer", ...)` asks for
  the Writer's model, and `config.yaml` decides what that is. No node names a model.
* **Open-weight first.** The `open` profile is the default and is what the deployed link
  runs on. Any OpenAI-compatible endpoint works, so Groq, OpenRouter, Together and a
  local Ollama are all a `base_url` change.
* **Costs are measured, not estimated.** Every call records tokens and latency, so the
  per-job and per-proposal figures in the report come from counters rather than guesses.

There is also an offline mode. With no API key the client returns deterministic canned
structures instead of raising, so tests, CI and `make demo` all work on a laptop with no
credentials. Offline results are stamped `offline: true` and the cost report says so; a
number produced offline is never reported as a measured one.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import yaml

# Importing this loads .env before any os.getenv call below runs.
from agent.env import first_env

CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"


class LLMError(RuntimeError):
    """The provider refused, or returned something unusable."""


@dataclass
class Usage:
    """What one call cost. Summed per thread for the budget report."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    calls: int = 0
    latency_ms: float = 0.0
    offline: bool = False

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def add(self, other: Usage) -> None:
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        self.calls += other.calls
        self.latency_ms += other.latency_ms
        self.offline = self.offline or other.offline

    def to_dict(self) -> dict[str, Any]:
        return {
            "calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "latency_ms": round(self.latency_ms, 1),
            "offline": self.offline,
        }


@dataclass
class Reply:
    text: str
    usage: Usage
    model: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    offline: bool = False

    def json(self) -> Any:
        """Parse the reply as JSON, tolerating the fences models insist on adding."""
        raw = self.text.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1]
            if raw.rstrip().endswith("```"):
                raw = raw.rstrip()[:-3]
        raw = raw.strip()
        # Some models prepend a sentence before the object. Salvage the outermost braces
        # rather than failing a whole run over a stray "Here is the JSON:".
        if not raw.startswith(("{", "[")):
            start = min((i for i in (raw.find("{"), raw.find("[")) if i != -1), default=-1)
            if start == -1:
                raise LLMError(f"no JSON in model reply: {self.text[:200]!r}")
            raw = raw[start:]
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            start = max(0, exc.pos - 80)
            end = min(len(self.text), exc.pos + 80)
            context = self.text[start:end]
            raise LLMError(
                f"model reply was not valid JSON ({exc.msg}) at character {exc.pos}: "
                f"{context!r}"
            ) from exc


def load_config(path: Path | None = None) -> dict[str, Any]:
    return yaml.safe_load((path or CONFIG_PATH).read_text(encoding="utf-8"))


# ------------------------------------------------------- tokens-per-minute budget


class TokenBudget:
    """A sliding-window tokens-per-minute budget, shared by every client in the process.

    Free tiers meter tokens per minute, not requests, so a request-per-second limiter does
    not help: one 2,000-token call can eat a quarter of the window. This tracks what was
    spent in the last 60 seconds and blocks until there is room.

    Shared at module level rather than per-client because the provider's limit is per
    organisation. Two LLM instances in one process draw on the same allowance, and a
    per-instance limiter would let them collectively blow through it.
    """

    def __init__(self) -> None:
        self._spent: list[tuple[float, int]] = []  # (timestamp, tokens)
        self._lock = threading.Lock()

    def _prune(self, now: float) -> int:
        self._spent = [(t, n) for t, n in self._spent if now - t < 60.0]
        return sum(n for _, n in self._spent)

    def reserve(self, tokens: int, limit: int) -> None:
        """Block until `tokens` fit in the current window, then record them."""
        if limit <= 0:
            return
        # Leave 10% headroom: our estimate of the prompt size is approximate, and
        # overshooting costs a 429 for every concurrent caller.
        ceiling = limit * 0.9
        while True:
            with self._lock:
                now = time.monotonic()
                used = self._prune(now)
                if used + tokens <= ceiling or not self._spent:
                    self._spent.append((now, tokens))
                    return
                # Sleep until the oldest entry leaves the window.
                oldest = self._spent[0][0]
                wait = max(0.2, 60.0 - (now - oldest) + 0.1)
            time.sleep(min(wait, 60.0))

    def settle(self, reserved: int, actual: int) -> None:
        """Replace the reservation with the real figure."""
        with self._lock:
            for i in range(len(self._spent) - 1, -1, -1):
                if self._spent[i][1] == reserved:
                    self._spent[i] = (self._spent[i][0], actual)
                    return


_budget = TokenBudget()


def _estimate_prompt_tokens(
    messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None
) -> int:
    """Rough token count for budgeting. Four characters per token is close enough.

    Tool schemas count: they are sent on every request and in this study they are most of
    the prompt, which is precisely why the fine-grained arm is expensive.
    """
    chars = sum(len(str(m.get("content") or "")) for m in messages)
    if tools:
        chars += len(json.dumps(tools))
    return chars // 4 + 16


def _is_daily_quota(response: httpx.Response) -> bool:
    """Is this 429 a per-day cap rather than a per-minute one?

    Groq says "tokens per day (TPD)" in the body; other providers word it differently, so
    a few spellings are matched. Getting this wrong in the permissive direction only costs
    a few wasted retries, so the check is deliberately narrow.
    """
    text = response.text.lower()
    return any(
        phrase in text
        for phrase in ("per day", "tpd", "daily limit", "quota exceeded", "rpd")
    )


def _quota_detail(response: httpx.Response) -> str:
    """The provider's own numbers, for a message a human can act on."""
    match = re.search(r"(Limit \d+, Used \d+)", response.text)
    if match:
        return match.group(1)
    return response.text[:160]


def _retry_after_seconds(response: httpx.Response) -> float:
    """How long the provider asked us to wait, from the header or the message body."""
    header = response.headers.get("Retry-After") or response.headers.get("retry-after")
    if header:
        try:
            return float(header)
        except ValueError:
            pass
    # Groq puts it in prose: "Please try again in 5.0175s."
    match = re.search(r"try again in ([\d.]+)\s*(ms|s)\b", response.text)
    if match:
        value = float(match.group(1))
        return value / 1000.0 if match.group(2) == "ms" else value
    return 0.0


class LLM:
    """An OpenAI-compatible chat client with per-node model routing."""

    def __init__(self, config: dict[str, Any] | None = None, profile: str | None = None) -> None:
        self.config = config or load_config()
        models = self.config.get("models") or {}
        self.profile_name = profile or os.getenv("MODEL_PROFILE") or models.get("profile", "open")
        profiles = models.get("profiles") or {}
        if self.profile_name not in profiles:
            raise LLMError(
                f"unknown model profile {self.profile_name!r}; config defines {sorted(profiles)}"
            )
        self.profile = profiles[self.profile_name]
        self.node_map: dict[str, str] = models.get("nodes") or {}
        self.temperature = float(models.get("temperature", 0.2))
        self.writer_temperature = float(models.get("writer_temperature", 0.5))
        # A profile may cap its own output: providers set output-per-minute limits per
        # model, and exceeding one is refused as "request too large" rather than queued.
        self.max_output_tokens = int(
            self.profile.get("max_output_tokens") or models.get("max_output_tokens", 2000)
        )
        self.timeout = float(models.get("request_timeout_sec", 60))
        # Free tiers meter tokens per minute. Set to the provider's limit so evaluations
        # pace themselves instead of collecting 429s.
        self.tpm_limit = int(os.getenv("LLM_TPM_LIMIT") or models.get("tpm_limit", 8000))
        self.max_attempts = int(models.get("max_attempts", 4))

        self.api_key, self.api_key_source = first_env(self.profile.get("api_key_env") or [])
        self.base_url = (self.profile.get("base_url") or "").rstrip("/")
        self.total = Usage()

    # ------------------------------------------------------------------ routing

    @property
    def offline(self) -> bool:
        """True when no credentials are available, so results are canned.

        `GIGBID_FORCE_OFFLINE=1` forces it even with a key present. Tests and CI set it:
        a graph test should exercise routing, reducers and interrupts, not a provider's
        rate limit, and a test whose result depends on what a model felt like returning
        is not a test.
        """
        if os.getenv("GIGBID_FORCE_OFFLINE", "").lower() in ("1", "true", "yes"):
            return True
        return not self.api_key

    def model_for(self, node: str) -> str:
        """The model this node runs on, resolved through the active profile.

        An unknown node falls back to the small model rather than the large one: a node
        nobody configured should not quietly start spending on the expensive model.
        """
        size = self.node_map.get(node, "small")
        model = self.profile.get(size)
        if not model:
            raise LLMError(
                f"profile {self.profile_name!r} defines no {size!r} model for node {node!r}"
            )
        return model

    def why(self, node: str) -> str:
        """A one-line explanation of a routing decision, for the run viewer and the viva."""
        size = self.node_map.get(node, "small")
        return f"{node} -> {size} -> {self.model_for(node)} (profile {self.profile_name})"

    # -------------------------------------------------------------------- calling

    def chat(
        self,
        node: str,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        json_mode: bool = False,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Reply:
        """One chat completion for one node."""
        model = self.model_for(node)
        if self.offline:
            reply = _offline_reply(node, messages, model, tools=tools)
            self.total.add(reply.usage)
            return reply

        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": (
                temperature
                if temperature is not None
                else (self.writer_temperature if node in ("writer", "client_liaison") else self.temperature)
            ),
            "max_tokens": max_tokens or self.max_output_tokens,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        # Wait for room in the token-per-minute budget before spending any of it. Without
        # this an evaluation sends 40 requests in 12 seconds, the provider refuses most of
        # them, and the resulting table measures the rate limiter rather than the thing
        # under test.
        estimated = _estimate_prompt_tokens(messages, tools) + int(payload["max_tokens"])
        _budget.reserve(estimated, self.tpm_limit)

        started = time.perf_counter()
        response = None
        last_error = ""
        for attempt in range(1, self.max_attempts + 1):
            try:
                response = httpx.post(
                    f"{self.base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                    timeout=self.timeout,
                )
            except httpx.HTTPError as exc:
                last_error = f"{self.profile.get('provider')} unreachable: {exc}"
                if attempt < self.max_attempts:
                    time.sleep(min(30.0, 2.0**attempt))
                    continue
                raise LLMError(last_error) from exc

            if response.status_code == 429:
                # Two very different 429s wear the same status code, and retrying the
                # second one is pointless:
                #
                #   per-minute (TPM) -- wait a few seconds and continue;
                #   per-day (TPD)    -- the quota is gone until it resets. Retrying ten
                #                       times just turns one clear failure into ten
                #                       confusing ones, and an evaluation that retries
                #                       through it reports zeros as if they were results.
                if _is_daily_quota(response):
                    raise LLMError(
                        f"DAILY_QUOTA_EXHAUSTED for {model}: {_quota_detail(response)}. "
                        "Retrying will not help until the quota resets. Either wait, or "
                        "switch models.profile in agent/config.yaml to a profile using a "
                        "different model."
                    )
                wait = _retry_after_seconds(response) or min(30.0, 2.0 * attempt)
                last_error = f"429 rate limited, waited {wait:.1f}s"
                if attempt < self.max_attempts:
                    time.sleep(wait + 0.25)
                    continue
                raise LLMError(f"429 from provider after {attempt} attempts: {response.text[:200]}")

            if response.status_code >= 500 and attempt < self.max_attempts:
                last_error = f"{response.status_code} from provider"
                time.sleep(min(30.0, 2.0**attempt))
                continue

            # Strict JSON mode rejects output it cannot validate, and the reasoning models
            # in the open profile spend output tokens on reasoning before emitting the
            # object, so a long answer gets truncated and the provider returns 400 rather
            # than partial JSON. Retrying without `response_format` lets the lenient
            # parser in Reply.json() recover the object from fenced or prefixed text,
            # which it handles anyway.
            if (
                response.status_code == 400
                and "response_format" in payload
                and "json" in response.text.lower()
            ):
                payload.pop("response_format", None)
                payload["max_tokens"] = min(8000, int(payload["max_tokens"]) * 2)
                last_error = "400 on strict JSON mode; retrying without it"
                if attempt < self.max_attempts:
                    continue

            break

        if response is None:
            raise LLMError(last_error or "no response from provider")

        elapsed_ms = (time.perf_counter() - started) * 1000

        if response.status_code >= 400:
            raise LLMError(f"{response.status_code} from provider: {response.text[:300]}")

        body = response.json()
        choice = (body.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        usage_body = body.get("usage") or {}

        usage = Usage(
            prompt_tokens=int(usage_body.get("prompt_tokens") or 0),
            completion_tokens=int(usage_body.get("completion_tokens") or 0),
            calls=1,
            latency_ms=elapsed_ms,
        )
        self.total.add(usage)
        # Correct the reservation with what was actually spent, so the budget tracks
        # reality rather than our estimate drifting over a long run.
        _budget.settle(estimated, usage.total_tokens)

        return Reply(
            text=message.get("content") or "",
            usage=usage,
            model=model,
            tool_calls=[
                {
                    "id": tc.get("id"),
                    "name": (tc.get("function") or {}).get("name"),
                    "arguments": _parse_args((tc.get("function") or {}).get("arguments")),
                }
                for tc in (message.get("tool_calls") or [])
            ],
        )

    def cost_report(self) -> dict[str, Any]:
        """Tokens, latency and USD for this client's lifetime.

        USD is reported as null offline and whenever the provider's price is not in the
        table below, rather than printed as zero. A fabricated zero in a budget table is
        worse than an honest blank.
        """
        report = self.total.to_dict()
        report["profile"] = self.profile_name
        report["provider"] = self.profile.get("provider")
        report["api_key_from"] = self.api_key_source or None
        report["usd"] = (
            None if self.offline else _estimate_usd(self.profile, self.node_map, self.total)
        )
        return report


# --------------------------------------------------------------- price estimates

# USD per million tokens, (prompt, completion). Published list prices as of the run;
# these are inputs to a report, so they belong in one visible place rather than inline.
_PRICES: dict[str, tuple[float, float]] = {
    "openai/gpt-oss-20b": (0.10, 0.50),
    "openai/gpt-oss-120b": (0.15, 0.75),
    "claude-haiku-5-5": (1.00, 5.00),
    "claude-opus-5-5": (15.00, 75.00),
}


def _estimate_usd(profile: dict[str, Any], node_map: dict[str, str], usage: Usage) -> float | None:
    """Blended estimate using the large model's price as the conservative bound."""
    model = profile.get("large") or profile.get("small") or ""
    price = _PRICES.get(model)
    if price is None:
        return None
    prompt_rate, completion_rate = price
    return round(
        usage.prompt_tokens / 1_000_000 * prompt_rate
        + usage.completion_tokens / 1_000_000 * completion_rate,
        6,
    )


def _parse_args(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {"_unparsed": raw}
    return {}


# ------------------------------------------------------------------- offline mode


def _offline_reply(
    node: str, messages: list[dict[str, Any]], model: str, tools: list[dict[str, Any]] | None = None
) -> Reply:
    """A deterministic stand-in, so nothing requires credentials to run.

    Shaped to match what each node expects, so the graph's plumbing can be exercised and
    tested without a provider. These are not model outputs and are never reported as
    measured results: `offline` is True all the way into the cost report.
    """
    prompt = "\n".join(str(m.get("content") or "") for m in messages)
    usage = Usage(
        prompt_tokens=len(prompt) // 4,
        completion_tokens=40,
        calls=1,
        latency_ms=1.0,
        offline=True,
    )

    # Shaped to match what each node expects, so the graph's routing, reducers, loop
    # bounds and interrupts can all be exercised without a provider. These are not model
    # outputs and are never reported as measurements: `offline` stays True all the way
    # into the cost report, and the eval runners refuse to write results from them.
    canned: dict[str, Any] = {
        "analyst": {
            "deliverables": ["[offline] run with a provider key for real extraction"],
            "required_skills": [],
            # Empty, so the node falls back to rule-based decomposition from the post and
            # the graph still has requirements to work with.
            "requirements": [],
            "screening_questions": [],
            "hidden_needs": [],
        },
        "risk_assessor": {
            "score": 0.5,
            "flags": ["offline_mode"],
            "explanation": "[offline] no model explanation available",
        },
        "fit_matcher": {"requirement_verdicts": [], "missing_skills": [], "overall_note": "[offline]"},
        "pricing": {"amount_usd": 0, "rationale": "[offline]", "connects_worth_it": True},
        "writer": {
            "cover_letter": (
                "[offline mode] This is a placeholder draft, not generated text. "
                "Set GROQ_API_KEY in .env to produce a real proposal."
            ),
            "screening_answers": [],
            "claim_citations": {},
        },
        "critic": {
            "verdict": "approve",
            "reasons": ["[offline] no critique performed"],
            "grounding_ok": True,
            "personalised": True,
            "length_ok": True,
            "opening_is_generic": False,
            "suggested_fixes": [],
        },
        "compliance": {
            "status": "pass",
            "violations": [],
            "redactions": [],
            "policy_rules_cited": [],
            "explanation": "[offline] rule-based checks still ran",
            "cleaned_text": None,
        },
        "intent": {"intent": "question", "confidence": 0.5, "risk_flags": [], "wants_slots": False},
        "client_liaison": {
            "reply": "[offline mode] Placeholder reply, not generated text.",
            "offers_slots": False,
            "notes_for_user": "[offline]",
        },
        "verifier": {"claims": []},
    }

    if node in canned:
        return Reply(
            text=json.dumps(canned[node]), usage=usage, model=model, offline=True
        )
    return Reply(
        text=(
            "[offline mode] No model credentials are configured, so this text is a "
            "placeholder rather than a generated draft. Set GROQ_API_KEY (or switch "
            "models.profile in agent/config.yaml) to produce real output."
        ),
        usage=usage,
        model=model,
        offline=True,
    )


_default: LLM | None = None


def get_llm(config: dict[str, Any] | None = None, profile: str | None = None) -> LLM:
    """The process-wide client, so token counters accumulate across nodes."""
    global _default
    if _default is None or config is not None or profile is not None:
        _default = LLM(config=config, profile=profile)
    return _default
