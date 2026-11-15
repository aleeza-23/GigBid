"""Dense vector index over portfolio evidence.

Small enough to be exact: 592 items means a brute-force matrix multiply beats any
approximate index, and introducing FAISS here would add a dependency and a recall question
to buy nothing. If the corpus grew past a few hundred thousand items this is the file that
would change, and nothing above it.

Indexes are cached per (persona set, backend, text revision) so a variant sweep encodes
the corpus once rather than once per variant. With a local transformer that is the
difference between a two-minute sweep and a twenty-minute one.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from mcp_servers.portfolio.corpus import EvidenceItem
from rag.embeddings import get_embedder, stable_hash


def item_text(item: EvidenceItem, include_metadata: bool = True) -> str:
    """The text that represents one item in the index.

    Project name and skills are prepended because a README's prose often never names the
    library it demonstrates, while the indexed skill list does. Leaving them out makes the
    dense baseline worse for a reason that has nothing to do with embeddings.
    """
    if not include_metadata:
        return item.text
    head = " ".join(filter(None, [item.project or "", " ".join(item.skills)]))
    return f"{head}\n{item.text}".strip()


@dataclass
class DenseIndex:
    """Encoded corpus plus the embedder that produced it."""

    items: list[EvidenceItem]
    matrix: np.ndarray
    embedder: Any
    _query_cache: dict[str, np.ndarray] = field(default_factory=dict, repr=False)

    @property
    def backend(self) -> str:
        return getattr(self.embedder, "name", "unknown")

    @property
    def is_pretrained(self) -> bool:
        """False when LSA produced these vectors. Travels into every results file."""
        return bool(getattr(self.embedder, "is_pretrained", False))

    @classmethod
    def build(
        cls,
        items: Sequence[EvidenceItem],
        config: dict[str, Any] | None = None,
        embedder: Any = None,
    ) -> DenseIndex:
        rows = list(items)
        texts = [item_text(i) for i in rows]
        # LSA needs the corpus to fit its projection, so it is passed in at construction.
        model = embedder or get_embedder(config, corpus=texts)
        matrix = model.encode(texts) if rows else np.zeros((0, 1), dtype=np.float32)
        return cls(items=rows, matrix=matrix, embedder=model)

    def encode_query(self, text: str) -> np.ndarray:
        """Encode a query, cached. V1 issues one query per requirement and they repeat."""
        key = stable_hash(text)
        cached = self._query_cache.get(key)
        if cached is None:
            cached = self.embedder.encode([text], is_query=True)[0]
            self._query_cache[key] = cached
        return cached


_cache: dict[str, DenseIndex] = {}


def get_dense_index(
    items: Sequence[EvidenceItem], config: dict[str, Any] | None = None
) -> DenseIndex:
    """A cached index for this corpus and backend."""
    rows = list(items)
    backend = (config or {}).get("backend", "lsa")
    key = stable_hash(f"{backend}:{len(rows)}:" + "|".join(i.id for i in rows[:200]))
    index = _cache.get(key)
    if index is None:
        index = DenseIndex.build(rows, config)
        _cache[key] = index
    return index


def clear_cache() -> None:
    _cache.clear()
