"""Shared node plumbing: model calls, tracing, budgets and retries.

Every node does the same four things around its actual work, so they live here once:

* **render a prompt from disk** and record its version in the trace, so `writer@v3`
  appears next to the output it produced;
* **count tokens and steps** against the per-thread budget (C4), and mark the thread
  `needs_attention` rather than looping forever;
* **retry a model call** with a fallback to the small model (C7), because a cheap node
  failing should degrade rather than kill a run;
* **never raise into the graph.** A node that throws takes the thread down; a node that
  returns an error entry leaves a thread a human can look at. The only exception is the
  approval interrupt, which must propagate.

The division of labour between rules and models is deliberate and worth stating, because
the report has to defend it: risk, fit coverage, pricing and claim verification are
**deterministic**, measured in Part B at 0.783 bid/skip accuracy, 0.857 scam F1 and 0.752
price-in-band. Models do extraction, drafting and critique, where there is no rule to
write. A model that merely re-scores a number a rule already computes adds cost, latency
and a failure mode for nothing.
"""

from __future__ import annotations

import asyncio
import json
import time
from functools import lru_cache
from typing import Any

from agent.llm import LLMError, get_llm
from agent.prompts import loader
from agent.state import trace


class BudgetExceeded(RuntimeError):
    """The thread hit its step or token ceiling (C4)."""


@lru_cache(maxsize=1)
def _yaml_config() -> dict[str, Any]:
    """agent/config.yaml, read once per process."""
    from agent.marketplace_port import load_config

    return load_config() or {}


def app_config(runnable_config: Any = None) -> dict[str, Any]:
    """GigBid's own configuration, given whatever LangGraph passed as `config`.

    This exists because of a bug worth describing. LangGraph passes each node a
    `RunnableConfig` as its second argument -- `{"configurable": {"thread_id": ...}}` --
    and the nodes were handing that straight to `load_thresholds()`, which looked for a
    `thresholds` key, found none, and silently fell back to the hardcoded defaults in
    `agent/decide.py`. Behaviour happened to match because those defaults equalled the
    YAML values, so nothing looked wrong; but **editing `agent/config.yaml` had no effect
    inside the graph at all**. That defeats the point of keeping rules in configuration,
    which is exactly what the change request tests.

    So: the app config comes from the file, and a run may override it by putting an
    `app_config` dict in `configurable` -- which is how a test changes a threshold, and
    how a per-user autonomy setting will eventually reach the nodes.
    """
    base = _yaml_config()
    if isinstance(runnable_config, dict):
        override = (runnable_config.get("configurable") or {}).get("app_config")
        if isinstance(override, dict):
            return {**base, **override}
    return base


def thresholds(runnable_config: Any = None) -> dict[str, Any]:
    return app_config(runnable_config).get("thresholds") or {}


def retrieval_config(runnable_config: Any = None) -> dict[str, Any]:
    cfg = app_config(runnable_config).get("retrieval") or {}
    return {"embeddings": cfg.get("embeddings") or {}, **cfg}


def budgets(runnable_config: Any = None) -> dict[str, Any]:
    return app_config(runnable_config).get("budgets") or {}


def check_budget(state: dict[str, Any], runnable_config: Any = None) -> str | None:
    """Return a reason string when the thread is over budget, else None.

    Checked at the top of every node rather than by a central supervisor, so a runaway
    loop stops at the next node boundary instead of after the whole graph drains.
    """
    limits = budgets(runnable_config)
    steps = int(state.get("steps_used") or 0)
    tokens = int(state.get("tokens_used") or 0)

    max_steps = int(limits.get("max_graph_steps") or 30)
    max_tokens = int(limits.get("max_tokens_per_job") or 12000)

    if steps >= max_steps:
        return f"step budget exhausted: {steps} of {max_steps}"
    if tokens >= max_tokens:
        return f"token budget exhausted: {tokens} of {max_tokens}"
    return None


def call_model(
    node: str,
    *,
    prompt_values: dict[str, Any],
    user_content: str | None = None,
    json_mode: bool = True,
    llm: Any = None,
    trace_id: str | None = None,
    thread_id: str | None = None,
    job_id: str | None = None,
    max_tokens: int | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Render this node's prompt, call its model, parse JSON. Returns (parsed, meta).

    `meta` carries tokens, latency, the model used and the prompt label, which the caller
    folds into the trace. On failure `parsed` is None and `meta["error"]` says why; the
    caller decides whether that is fatal for its node.

    `trace_id` is passed explicitly rather than read from the ContextVar, and that is not
    belt-and-braces. LangGraph runs each node in its own task, and a task copies the
    context when it is *created*, so a `set_trace_id` call inside one node is invisible to
    its siblings. Relying on the ContextVar produced a trace split across several ids and
    `spans: 0` under the id the thread actually carried — E3's "one trace id end to end"
    quietly false. The thread's id lives in state; nodes pass it in from there.
    """
    client = llm or get_llm()
    prompt = loader.load(node)
    try:
        system = prompt.render(**prompt_values)
    except loader.PromptError as exc:
        return None, {"error": str(exc), "prompt": prompt.label, "tokens": 0}

    messages = [{"role": "system", "content": system}]
    if user_content:
        messages.append({"role": "user", "content": user_content})
    else:
        # Some providers will not return structured output without a user turn.
        messages.append({"role": "user", "content": "Produce the JSON described above."})

    started = time.perf_counter()
    try:
        reply = client.chat(node, messages, json_mode=json_mode, max_tokens=max_tokens)
    except LLMError as exc:
        return None, {
            "error": str(exc),
            "prompt": prompt.label,
            "tokens": 0,
            "duration_ms": (time.perf_counter() - started) * 1000,
        }

    meta = {
        "prompt": prompt.label,
        "model": reply.model,
        "tokens": reply.usage.total_tokens,
        "duration_ms": reply.usage.latency_ms,
        "offline": reply.offline,
    }

    # E3: a span per model call, with tokens and cost attributed to the node that made
    # it. Written locally as well as to any hosted backend, so "tokens and cost per node"
    # is inspectable by a reviewer with no account.
    try:
        from agent.tracing import write_span
        from mcp_servers._common.trace import get_trace_id

        write_span(
            trace_id or get_trace_id(),
            node,
            kind="model",
            thread_id=thread_id,
            job_id=job_id,
            model=reply.model,
            prompt_tokens=reply.usage.prompt_tokens,
            completion_tokens=reply.usage.completion_tokens,
            duration_ms=reply.usage.latency_ms,
            prompt_version=prompt.label,
            offline=reply.offline,
        )
    except Exception:
        # Tracing must never be the reason a node fails.
        pass

    if not json_mode:
        return reply.text, meta

    try:
        return reply.json(), meta
    except LLMError as exc:
        meta["error"] = f"unparseable JSON: {exc}"
        return None, meta


async def call_model_async(node: str, **kwargs: Any) -> tuple[Any, dict[str, Any]]:
    """Run blocking provider HTTP and token pacing off the API event loop."""
    return await asyncio.to_thread(call_model, node, **kwargs)


def node_trace(node: str, kind: str, meta: dict[str, Any] | None = None, **fields: Any):
    """Trace entry for a node, folding in a model call's metadata."""
    payload = dict(fields)
    if meta:
        payload.setdefault("tokens", meta.get("tokens"))
        payload.setdefault("duration_ms", meta.get("duration_ms"))
        if meta.get("error"):
            payload["error"] = meta["error"]
        if meta.get("prompt"):
            payload["prompt"] = meta["prompt"]
        if meta.get("model"):
            payload["model"] = meta["model"]
    return trace(node, kind, **payload)


def step(state: dict[str, Any]) -> int:
    return int(state.get("steps_used") or 0) + 1


def error_entry(node: str, exc: BaseException | str, fatal: bool = False) -> dict[str, Any]:
    return {
        "node": node,
        "error": str(exc) if isinstance(exc, str) else f"{type(exc).__name__}: {exc}",
        "fatal": fatal,
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def compact(value: Any, limit: int = 600) -> str:
    """Shorten a structure for a prompt or a trace entry."""
    text = value if isinstance(value, str) else json.dumps(value, default=str, ensure_ascii=False)
    return text if len(text) <= limit else text[:limit] + f"...[+{len(text) - limit}]"


def evidence_block(evidence: list[dict[str, Any]], limit: int = 6, chars: int = 500) -> str:
    """Format retrieved evidence for a prompt.

    Each item leads with its id, because the Writer has to cite that id and a model will
    not invent a handle it was never shown. Keeping the id adjacent to the text is the
    cheapest way to make grounded citation the path of least resistance.
    """
    if not evidence:
        return "(no evidence retrieved - do not claim any past work)"
    lines: list[str] = []
    for item in evidence[:limit]:
        body = str(item.get("text") or "").replace("\n", " ")[:chars]
        lines.append(
            f"- id: {item.get('id')}\n"
            f"  citation: {item.get('citation')}\n"
            f"  type: {item.get('type')} | project: {item.get('project')} | "
            f"date: {item.get('date')} | outcome: {item.get('outcome')}\n"
            f"  text: {body}"
        )
    return "\n".join(lines)
