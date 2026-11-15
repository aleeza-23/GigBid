"""Embedding backends, chosen by config.

The assignment requires open-weight embedding and reranker models in both profiles, and
notes they run acceptably on CPU. Three backends implement the same interface:

* **local** - sentence-transformers with an open model (BGE, nomic-embed-text, Qwen3
  Embedding). The spec-compliant default. Needs a one-off model download.
* **api** - any OpenAI-compatible `/embeddings` endpoint serving an open model, for when
  the deployed host should not carry model weights. Note Groq serves no embeddings
  endpoint, so this needs a Together or OpenRouter key.
* **lsa** - TF-IDF reduced by truncated SVD, numpy only, no download. This is **not** an
  open-weight embedding model and must not be reported as one. It exists so the tests,
  CI and offline development all work, and so a variant comparison can be exercised
  before anyone downloads anything.

Every backend reports `is_pretrained`. Anything that writes a results file reads that flag
and records which backend produced the numbers, because "V2 beat V0 by 15 points" means
something different when the dense half was LSA rather than BGE.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
from pathlib import Path
from typing import Any, Protocol

import numpy as np

CACHE_DIR = Path(
    os.getenv("EMBED_CACHE_DIR", Path(__file__).resolve().parents[1] / "data" / "processed" / "embed")
)

_WORD = re.compile(r"[a-z0-9+#.]+")


class Embedder(Protocol):
    """What the dense variants need."""

    name: str
    dim: int
    is_pretrained: bool
    """False for LSA. Results from a non-pretrained backend are not spec-compliant."""

    def encode(self, texts: list[str], is_query: bool = False) -> np.ndarray:
        """Return an (n, dim) L2-normalised float32 array."""
        ...


def _normalise(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (matrix / norms).astype(np.float32)


# --------------------------------------------------------------------------- LSA


class LsaEmbedder:
    """TF-IDF plus truncated SVD. numpy only, no download, deterministic.

    Latent semantic analysis is a real retrieval technique, just a 1990s one: it captures
    co-occurrence structure, so it is a meaningful dense baseline, but it has no knowledge
    beyond this 592-item corpus. A transformer embedding knows that "Celery" and "task
    queue" are related from pretraining; LSA can only know it if they co-occur here.

    That difference is exactly what Part B is measuring, which is why this backend is for
    development and must be labelled wherever its numbers appear.
    """

    name = "lsa-tfidf-svd"
    is_pretrained = False

    def __init__(self, dim: int = 192) -> None:
        self.dim = dim
        self._vocab: dict[str, int] = {}
        self._idf: np.ndarray | None = None
        self._components: np.ndarray | None = None
        self._fitted = False

    def _tokenise(self, text: str) -> list[str]:
        return _WORD.findall(text.lower())

    def fit(self, texts: list[str]) -> LsaEmbedder:
        """Learn the vocabulary, idf weights and SVD basis from the corpus."""
        docs = [self._tokenise(t) for t in texts]
        vocab: dict[str, int] = {}
        for doc in docs:
            for token in doc:
                if token not in vocab:
                    vocab[token] = len(vocab)
        self._vocab = vocab
        if not vocab:
            self._fitted = True
            return self

        counts = np.zeros((len(docs), len(vocab)), dtype=np.float32)
        for i, doc in enumerate(docs):
            for token in doc:
                counts[i, vocab[token]] += 1.0

        df = (counts > 0).sum(axis=0)
        self._idf = np.log((1 + len(docs)) / (1 + df)).astype(np.float32) + 1.0
        tfidf = _normalise(counts * self._idf)

        # Economy SVD, then keep the top components as the projection basis.
        k = min(self.dim, min(tfidf.shape) - 1) or 1
        try:
            _, _, vt = np.linalg.svd(tfidf, full_matrices=False)
            self._components = vt[:k].astype(np.float32)
        except np.linalg.LinAlgError:
            # Degenerate corpus; fall back to raw TF-IDF space.
            self._components = None
        self.dim = k if self._components is not None else len(vocab)
        self._fitted = True
        return self

    def _vectorise(self, texts: list[str]) -> np.ndarray:
        if not self._vocab or self._idf is None:
            return np.zeros((len(texts), max(1, self.dim)), dtype=np.float32)
        rows = np.zeros((len(texts), len(self._vocab)), dtype=np.float32)
        for i, text in enumerate(texts):
            for token in self._tokenise(text):
                index = self._vocab.get(token)
                if index is not None:
                    rows[i, index] += 1.0
        return _normalise(rows * self._idf)

    def encode(self, texts: list[str], is_query: bool = False) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("LsaEmbedder.fit(corpus) must be called before encode()")
        tfidf = self._vectorise(texts)
        if self._components is None:
            return tfidf
        return _normalise(tfidf @ self._components.T)


# ------------------------------------------------------------------------- local


class SentenceTransformerEmbedder:
    """An open-weight model via sentence-transformers. The spec-compliant backend."""

    is_pretrained = True

    def __init__(self, model: str | None = None) -> None:
        self.name = model or os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "the local embedding backend needs sentence-transformers:\n"
                "  pip install sentence-transformers\n"
                "or set rag.embeddings.backend to 'api' or 'lsa' in agent/config.yaml"
            ) from exc
        self._model = SentenceTransformer(self.name)
        self.dim = int(self._model.get_sentence_embedding_dimension())

    def encode(self, texts: list[str], is_query: bool = False) -> np.ndarray:
        # BGE models want an instruction prefix on queries but not on documents; getting
        # this backwards costs several points of recall and is silent.
        prepared = texts
        if is_query and "bge" in self.name.lower():
            prefix = "Represent this sentence for searching relevant passages: "
            prepared = [prefix + t for t in texts]
        vectors = self._model.encode(
            prepared, batch_size=32, show_progress_bar=False, convert_to_numpy=True
        )
        return _normalise(np.asarray(vectors, dtype=np.float32))


# --------------------------------------------------------------------------- api


class ApiEmbedder:
    """Any OpenAI-compatible /embeddings endpoint serving an open model."""

    is_pretrained = True

    def __init__(
        self, model: str | None = None, base_url: str | None = None, api_key_env: Any = None
    ) -> None:
        from agent.env import first_env

        self.name = model or os.getenv("EMBED_MODEL", "BAAI/bge-base-en-v1.5")
        self.base_url = (
            base_url or os.getenv("EMBED_BASE_URL", "https://api.together.xyz/v1")
        ).rstrip("/")
        self.api_key, self.api_key_source = first_env(
            api_key_env or ["TOGETHER_API_KEY", "OPENROUTER_API_KEY", "EMBED_API_KEY"]
        )
        if not self.api_key:
            raise RuntimeError(
                "the api embedding backend found no key. Groq serves no embeddings "
                "endpoint, so this needs TOGETHER_API_KEY or OPENROUTER_API_KEY. Use the "
                "'local' or 'lsa' backend instead if you have neither."
            )
        self.dim = 0  # learned from the first response

    def encode(self, texts: list[str], is_query: bool = False) -> np.ndarray:
        import httpx

        out: list[list[float]] = []
        # Batched: a 592-item corpus in one request tends to exceed payload limits.
        for start in range(0, len(texts), 64):
            batch = texts[start : start + 64]
            response = httpx.post(
                f"{self.base_url}/embeddings",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={"model": self.name, "input": batch},
                timeout=120,
            )
            if response.status_code >= 400:
                raise RuntimeError(f"embeddings {response.status_code}: {response.text[:200]}")
            data = sorted(response.json()["data"], key=lambda d: d.get("index", 0))
            out.extend(d["embedding"] for d in data)
        matrix = np.asarray(out, dtype=np.float32)
        self.dim = matrix.shape[1] if matrix.size else 0
        return _normalise(matrix)


# ------------------------------------------------------------------------ factory


def get_embedder(config: dict[str, Any] | None = None, corpus: list[str] | None = None) -> Any:
    """Build the backend named in config, falling back loudly rather than silently.

    `corpus` is needed only by LSA, which has nothing to project onto until it has seen
    the documents.
    """
    cfg = config or {}
    backend = os.getenv("EMBED_BACKEND") or cfg.get("backend", "lsa")

    if backend == "local":
        return SentenceTransformerEmbedder(cfg.get("model"))
    if backend == "api":
        return ApiEmbedder(cfg.get("model"), cfg.get("base_url"), cfg.get("api_key_env"))
    if backend == "lsa":
        embedder = LsaEmbedder(int(cfg.get("dim", 192)))
        if corpus:
            embedder.fit(corpus)
        return embedder
    raise ValueError(f"unknown embedding backend {backend!r}: expected local, api or lsa")


def cosine_scores(query_vector: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Cosine similarity against pre-normalised rows, which is just a dot product."""
    if matrix.size == 0:
        return np.zeros(0, dtype=np.float32)
    return matrix @ query_vector.astype(np.float32)


def stable_hash(text: str) -> str:
    """A content hash, for caching embeddings across runs."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def dcg(relevances: list[float]) -> float:
    return sum(r / math.log2(i + 2) for i, r in enumerate(relevances))
