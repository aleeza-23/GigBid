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
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import yaml

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
            raise LLMError(f"model reply was not valid JSON: {self.text[:300]!r}") from exc


def load_config(path: Path | None = None) -> dict[str, Any]:
    return yaml.safe_load((path or CONFIG_PATH).read_text(encoding="utf-8"))


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
        self.max_output_tokens = int(models.get("max_output_tokens", 2000))
        self.timeout = float(models.get("request_timeout_sec", 60))

        self.api_key = os.getenv(self.profile.get("api_key_env", "") or "", "").strip()
        self.base_url = (self.profile.get("base_url") or "").rstrip("/")
        self.total = Usage()

    # ------------------------------------------------------------------ routing

    @property
    def offline(self) -> bool:
        """True when no credentials are available, so results are canned."""
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

        started = time.perf_counter()
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
            raise LLMError(f"{self.profile.get('provider')} unreachable: {exc}") from exc
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

    canned: dict[str, Any] = {
        "analyst": {
            "deliverables": ["[offline] run with a provider key for real extraction"],
            "required_skills": [],
            "screening_questions": [],
            "hidden_needs": [],
        },
        "risk_assessor": {"score": 0.5, "flags": ["offline_mode"], "explanation": "[offline]"},
        "fit_matcher": {"score": 0.5, "matched_skills": [], "missing_skills": [], "evidence_ids": []},
        "pricing": {"amount_usd": 0, "rationale": "[offline]", "within_floor": True},
        "critic": {"verdict": "approve", "reasons": ["[offline] no critique performed"]},
        "compliance": {"verdict": "pass", "violations": []},
        "intent": {"intent": "question", "confidence": 0.5},
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
