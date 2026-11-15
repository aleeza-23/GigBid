"""Test-wide defaults, so the suite tests this system rather than its dependencies.

Three environment defaults, each fixing a specific way the suite would otherwise measure
something it does not mean to:

* **`GIGBID_FORCE_OFFLINE`** — graph and API tests check routing, reducers, loop bounds and
  interrupts. None of that should depend on what a model returned, and a test whose result
  varies with a provider's mood is not a test.

* **`EMBED_BACKEND=lsa`** — the configured backend is a real transformer, which is correct
  for production and wrong here: it downloads 130MB on a cold CI runner and encodes the
  corpus on every session. LSA is numpy-only and deterministic.

* **`RERANK_BACKEND=lexical`** — a cross-encoder scoring 30 candidates per query on CPU
  took the RAG test file from 2 seconds to 119. The tests check that reranking is wired in
  and reorders candidates, not that BGE ranks well.

All three use `setdefault`, so a deliberate `EMBED_BACKEND=local pytest ...` still runs
against the real models when that is the point.
"""

from __future__ import annotations

import os

os.environ.setdefault("GIGBID_FORCE_OFFLINE", "1")
os.environ.setdefault("EMBED_BACKEND", "lsa")
os.environ.setdefault("RERANK_BACKEND", "lexical")

# Keep runtime state out of the repository and out of each other's way. Individual tests
# override these with tmp_path where they need isolation per test.
os.environ.setdefault("MCP_IDEMPOTENCY_DB", ":memory:")
os.environ.setdefault("APP_SECRET_KEY", "test-secret-not-a-real-key")
