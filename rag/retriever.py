"""Variant registry and factory: the retrieval design is chosen by config (R6).

`mcp_servers/portfolio/search.py` looks for `get_ranker` here on every call. When this
module is importable and the configured variant resolves, the Portfolio server's
`search_evidence` tool starts using it — with no change to the tool, its schema, or any
node that calls it. That is the whole reason the swap point was built before the variants
existed.

Which variant ships is `retrieval.variant` in `agent/config.yaml`, and R6 requires that
value to be justified by the evaluation in `evals/results/`, not by preference. The
default stays `builtin_bm25` until a measured run says otherwise.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import yaml

from mcp_servers.portfolio.corpus import EvidenceItem

CONFIG_PATH = Path(__file__).resolve().parents[1] / "agent" / "config.yaml"


def _load_variants() -> dict[str, dict[str, Any]]:
    """Import every variant module and index it by id and by name.

    Imported lazily and defensively: a variant with a broken import should not take the
    tool layer down with it, because `search_evidence` must keep answering.
    """
    registry: dict[str, dict[str, Any]] = {}
    modules = [
        ("rag.variants.v0_naive", "V0"),
        ("rag.variants.v1_decompose", "V1"),
        ("rag.variants.v2_hybrid", "V2"),
        ("rag.variants.v3_rerank", "V3"),
        ("rag.variants.v8_corrective", "V8"),
        # V1's queries over V2's retriever, no reranking: what Part B measured as best.
        ("rag.variants.v12_decompose_hybrid", "V12"),
    ]
    import importlib

    for module_path, expected_id in modules:
        try:
            module = importlib.import_module(module_path)
            spec = getattr(module, "VARIANT", None)
            if not spec:
                continue
            registry[spec["id"]] = spec
            registry[spec["name"]] = spec
            registry[spec["id"].lower()] = spec
        except Exception as exc:  # pragma: no cover - defensive
            registry[f"{expected_id}:error"] = {
                "id": expected_id,
                "name": f"{expected_id}_unavailable",
                "error": f"{type(exc).__name__}: {exc}",
            }
    return registry


_registry: dict[str, dict[str, Any]] | None = None


def variants() -> dict[str, dict[str, Any]]:
    global _registry
    if _registry is None:
        _registry = _load_variants()
    return _registry


def list_variants() -> list[dict[str, Any]]:
    """Every working variant, deduplicated, for the comparison runner and the UI."""
    seen: dict[str, dict[str, Any]] = {}
    for spec in variants().values():
        if "error" in spec:
            continue
        seen[spec["id"]] = spec
    return [seen[key] for key in sorted(seen)]


def load_config(path: Path | None = None) -> dict[str, Any]:
    try:
        return yaml.safe_load((path or CONFIG_PATH).read_text(encoding="utf-8")) or {}
    except Exception:
        return {}


def get_variant(name: str | None = None, config: dict[str, Any] | None = None) -> dict[str, Any] | None:
    """Resolve a variant by id or name, or from config. None means the built-in ranker."""
    cfg = config if config is not None else load_config()
    wanted = name or os.getenv("RAG_VARIANT") or (cfg.get("retrieval") or {}).get("variant")
    if not wanted or wanted == "builtin_bm25":
        return None
    spec = variants().get(wanted) or variants().get(str(wanted).upper())
    return spec if spec and "error" not in spec else None


def get_ranker(name: str | None = None, config: dict[str, Any] | None = None):
    """The ranking callable the Portfolio server expects, or None for the built-in.

    Signature contract, matching `mcp_servers.portfolio.search`:

        rank(requirement, items, k, **opts) -> list[EvidenceItem] with .score set
    """
    cfg = config if config is not None else load_config()
    spec = get_variant(name, cfg)
    if spec is None:
        return None

    retrieval_cfg = cfg.get("retrieval") or {}
    full_cfg = {"embeddings": retrieval_cfg.get("embeddings") or {}, **retrieval_cfg}

    def rank(
        requirement: str, items: Sequence[EvidenceItem], k: int = 5, **opts: Any
    ) -> list[EvidenceItem]:
        # The tool layer passes outcome_aware; variants take it as a keyword or ignore it.
        merged = {**full_cfg, **opts}
        return spec["retrieve"](requirement, items, k, config=full_cfg, **_filter(merged))

    rank.__qualname__ = f"variant:{spec['id']}:{spec['name']}"  # shows up in tool output
    return rank


_ACCEPTED = {
    "llm", "max_rounds", "candidate_depth", "bm25_weight", "dense_weight",
    "requirements", "base_retrieve", "index", "outcome_aware",
}


def _filter(options: dict[str, Any]) -> dict[str, Any]:
    """Pass through only options a variant's retrieve() understands.

    Variants accept **_ so an unexpected key is harmless, but filtering keeps the call
    readable in a trace and stops config keys like `k` colliding with the positional
    argument.
    """
    return {key: value for key, value in options.items() if key in _ACCEPTED}


def describe() -> dict[str, Any]:
    """What is registered and what is configured. Printed by the eval runners."""
    cfg = load_config()
    configured = (cfg.get("retrieval") or {}).get("variant", "builtin_bm25")
    broken = [spec for spec in variants().values() if "error" in spec]
    return {
        "configured": configured,
        "available": [{"id": s["id"], "name": s["name"], "group": s["group"]} for s in list_variants()],
        "unavailable": [{"id": s["id"], "error": s["error"]} for s in broken],
        "embeddings_backend": ((cfg.get("retrieval") or {}).get("embeddings") or {}).get(
            "backend", "lsa"
        ),
    }
