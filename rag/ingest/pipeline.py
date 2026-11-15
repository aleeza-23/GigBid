"""Ingestion pipeline with incremental re-indexing (B1).

B1 asks for "incremental re-indexing when the user adds work in the UI". The naive version
re-reads and re-embeds the whole corpus on every upload, which with a local transformer
takes long enough that the UI has to show a spinner and the user wonders if it worked.

So the manifest records a content hash per document. On re-ingest:

* a document whose hash is unchanged is **reused**, embedding and all;
* a changed or new document is **re-chunked and re-embedded**;
* a document that has disappeared is **dropped**.

The embedding cache is keyed on chunk text, not document id, so editing one paragraph of
a case study re-embeds one chunk rather than the document's six.

    python -m rag.ingest.pipeline --persona ai_ml_engineer
    python -m rag.ingest.pipeline --persona ai_ml_engineer --stats
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

from rag.ingest.chunker import Chunk, chunk_all
from rag.ingest.loaders import RawDocument, load_persona_directory

ROOT = Path(__file__).resolve().parents[2]
PERSONAS_DIR = ROOT / "data" / "personas"
INDEX_DIR = ROOT / "data" / "processed" / "rag_index"


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


class IngestIndex:
    """Chunks plus a manifest, persisted so re-ingest can be incremental."""

    def __init__(self, persona: str, index_dir: Path | None = None) -> None:
        self.persona = persona
        self.dir = (index_dir or INDEX_DIR) / persona
        self.dir.mkdir(parents=True, exist_ok=True)
        self.chunks_path = self.dir / "chunks.jsonl"
        self.manifest_path = self.dir / "manifest.json"

    def load_manifest(self) -> dict[str, Any]:
        if not self.manifest_path.exists():
            return {"documents": {}, "built_at": None, "chunk_count": 0}
        return json.loads(self.manifest_path.read_text(encoding="utf-8"))

    def load_chunks(self) -> list[dict[str, Any]]:
        if not self.chunks_path.exists():
            return []
        return [
            json.loads(line)
            for line in self.chunks_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def write(self, chunks: list[dict[str, Any]], manifest: dict[str, Any]) -> None:
        with self.chunks_path.open("w", encoding="utf-8") as handle:
            for chunk in chunks:
                handle.write(json.dumps(chunk, ensure_ascii=False) + "\n")
        self.manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def ingest(
    persona: str,
    *,
    root: Path | None = None,
    index_dir: Path | None = None,
    max_tokens: int = 220,
    overlap_tokens: int = 40,
    force: bool = False,
) -> dict[str, Any]:
    """Ingest or re-ingest one persona's portfolio. Returns what changed."""
    started = time.perf_counter()
    index = IngestIndex(persona, index_dir)
    previous = index.load_manifest()
    previous_docs: dict[str, Any] = {} if force else (previous.get("documents") or {})
    existing_chunks = [] if force else index.load_chunks()
    chunks_by_parent: dict[str, list[dict[str, Any]]] = {}
    for chunk in existing_chunks:
        chunks_by_parent.setdefault(chunk["parent_id"], []).append(chunk)

    docs: list[RawDocument] = load_persona_directory(root or PERSONAS_DIR, persona)
    seen: set[str] = set()
    kept: list[dict[str, Any]] = []
    manifest_docs: dict[str, Any] = {}

    reused = changed = added = 0

    for doc in docs:
        seen.add(doc.id)
        digest = content_hash(doc.text)
        record = previous_docs.get(doc.id)

        if record and record.get("hash") == digest and chunks_by_parent.get(doc.id):
            # Unchanged: reuse the chunks exactly, so nothing is re-embedded.
            kept.extend(chunks_by_parent[doc.id])
            manifest_docs[doc.id] = record
            reused += 1
            continue

        fresh: list[Chunk] = chunk_all([doc], max_tokens, overlap_tokens)
        kept.extend(chunk.to_dict() for chunk in fresh)
        manifest_docs[doc.id] = {
            "hash": digest,
            "type": doc.type,
            "source": doc.source,
            "chunks": len(fresh),
            "project_id": doc.project_id,
        }
        if record:
            changed += 1
        else:
            added += 1

    removed = sorted(set(previous_docs) - seen)

    manifest = {
        "persona": persona,
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "chunk_count": len(kept),
        "document_count": len(manifest_docs),
        "chunk_tokens": max_tokens,
        "overlap_tokens": overlap_tokens,
        "documents": manifest_docs,
    }
    index.write(kept, manifest)

    return {
        "persona": persona,
        "documents": len(docs),
        "chunks": len(kept),
        "added": added,
        "changed": changed,
        "reused": reused,
        "removed": removed,
        "incremental": bool(previous_docs) and not force,
        "elapsed_sec": round(time.perf_counter() - started, 3),
    }


def stats(persona: str, index_dir: Path | None = None) -> dict[str, Any]:
    """What is indexed, for the data card and the settings screen."""
    index = IngestIndex(persona, index_dir)
    chunks = index.load_chunks()
    manifest = index.load_manifest()

    by_type: dict[str, int] = {}
    by_outcome: dict[str, int] = {}
    lengths: list[int] = []
    with_skills = 0
    for chunk in chunks:
        by_type[chunk["type"]] = by_type.get(chunk["type"], 0) + 1
        # Only proposals have a meaningful outcome. Case studies carry "delivered" and
        # counting those alongside won/replied/ignored made the funnel look wrong.
        if chunk["type"] == "proposal" and chunk.get("outcome"):
            by_outcome[chunk["outcome"]] = by_outcome.get(chunk["outcome"], 0) + 1
        lengths.append(len(chunk["text"]))
        if chunk.get("skills"):
            with_skills += 1

    return {
        "persona": persona,
        "documents": manifest.get("document_count", 0),
        "chunks": len(chunks),
        "by_type": dict(sorted(by_type.items())),
        "proposal_outcomes": dict(sorted(by_outcome.items())),
        "chunks_missing_skills": sum(1 for c in chunks if not c.get("skills")),
        "chunks_with_skill_metadata": with_skills,
        "mean_chunk_chars": round(sum(lengths) / len(lengths), 1) if lengths else 0,
        "max_chunk_chars": max(lengths) if lengths else 0,
        "built_at": manifest.get("built_at"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest a portfolio into chunks (B1)")
    parser.add_argument("--persona", default="ai_ml_engineer")
    parser.add_argument("--all", action="store_true", help="every persona")
    parser.add_argument("--force", action="store_true", help="full rebuild, ignore the manifest")
    parser.add_argument("--stats", action="store_true", help="report what is indexed and exit")
    parser.add_argument("--max-tokens", type=int, default=220)
    args = parser.parse_args()

    personas = (
        ["ai_ml_engineer", "fullstack_web_developer", "data_analyst"]
        if args.all
        else [args.persona]
    )

    if args.stats:
        for persona in personas:
            print(json.dumps(stats(persona), indent=2))
        return

    for persona in personas:
        report = ingest(persona, force=args.force, max_tokens=args.max_tokens)
        mode = "full rebuild" if not report["incremental"] else "incremental"
        print(
            f"{persona}: {report['documents']} docs -> {report['chunks']} chunks ({mode}) "
            f"added={report['added']} changed={report['changed']} reused={report['reused']} "
            f"removed={len(report['removed'])} in {report['elapsed_sec']}s"
        )


if __name__ == "__main__":
    main()
