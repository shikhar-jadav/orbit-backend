"""Chunked semantic + lexical RAG for large uploaded attachments (PDFs, docs, text)."""

from __future__ import annotations

import logging
import math
import re
import uuid
from collections import Counter
from typing import Any, Dict, List, Optional

logger = logging.getLogger("orbit.attachment_rag")

LARGE_ATTACHMENT_RAG_THRESHOLD_CHARS = 20_000
ATTACHMENTS_COLLECTION_NAME = "orbit_attachments"
_TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)

_ATTACHMENTS_COLLECTION_READY = False


def _tokens(text: str) -> List[str]:
    return _TOKEN_RE.findall((text or "").lower())


def _bm25_rank_indices(query: str, chunks: List[str], limit: int) -> List[int]:
    """Rank chunk indices using BM25 lexical scoring as a zero-dependency fallback."""
    if not chunks:
        return []
    q_tokens = _tokens(query)
    if not q_tokens:
        return list(range(min(limit, len(chunks))))

    doc_tokens = [_tokens(c) for c in chunks]
    avg_len = max(1.0, sum(len(d) for d in doc_tokens) / len(doc_tokens))
    df: Counter[str] = Counter()
    for d in doc_tokens:
        df.update(set(d))

    n_docs = len(doc_tokens)
    k1 = 1.5
    b = 0.75
    scores = []
    for idx, d in enumerate(doc_tokens):
        if not d:
            continue
        counts = Counter(d)
        dl = len(d)
        score = 0.0
        for tok in q_tokens:
            tf = counts.get(tok, 0)
            if not tf:
                continue
            idf = math.log(1.0 + (n_docs - df.get(tok, 0) + 0.5) / (df.get(tok, 0) + 0.5))
            denom = tf + k1 * (1.0 - b + b * (dl / avg_len))
            score += idf * (tf * (k1 + 1.0) / denom)
        if score > 0:
            scores.append((score, idx))

    scores.sort(key=lambda item: (item[0], -item[1]), reverse=True)
    selected = [idx for _, idx in scores[:limit]]
    if not selected:
        return list(range(min(limit, len(chunks))))
    return selected


def _ensure_attachments_collection(qdrant_store: Any) -> bool:
    global _ATTACHMENTS_COLLECTION_READY
    if _ATTACHMENTS_COLLECTION_READY:
        return True
    if not qdrant_store or not getattr(qdrant_store, "supports_semantic_search", False):
        return False
    try:
        from qdrant_client import models
        from qdrant_vector import DEFAULT_VECTOR_SIZE

        client = qdrant_store.client
        existing = {c.name for c in client.get_collections().collections}
        if ATTACHMENTS_COLLECTION_NAME not in existing:
            client.create_collection(
                collection_name=ATTACHMENTS_COLLECTION_NAME,
                vectors_config={
                    qdrant_store.vector_name: models.VectorParams(
                        size=DEFAULT_VECTOR_SIZE,
                        distance=models.Distance.COSINE,
                    )
                },
            )
        for field in ("thread_id", "attachment_id", "user_id"):
            try:
                client.create_payload_index(
                    collection_name=ATTACHMENTS_COLLECTION_NAME,
                    field_name=field,
                    field_schema=models.PayloadSchemaType.KEYWORD,
                )
            except Exception:
                pass
        _ATTACHMENTS_COLLECTION_READY = True
        return True
    except Exception as exc:
        logger.warning(f"[AttachmentRAG] Qdrant collection setup skipped: {exc}")
        return False


def index_attachment_chunks(
    *,
    thread_id: str,
    attachment_id: str,
    user_id: str,
    filename: str,
    chunks: List[str],
    qdrant_store: Optional[Any] = None,
) -> int:
    """Embed and upsert document chunks into Qdrant when available. Returns indexed count."""
    if not chunks or not _ensure_attachments_collection(qdrant_store):
        return 0
    try:
        from qdrant_client import models

        embedder = qdrant_store._get_embedder()
        if embedder is None:
            return 0

        vectors = [vec.tolist() for vec in embedder.embed(chunks)]
        points = []
        for idx, (chunk_text, vec) in enumerate(zip(chunks, vectors)):
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{thread_id}::{attachment_id}::{idx}"))
            points.append(
                models.PointStruct(
                    id=point_id,
                    vector={qdrant_store.vector_name: vec},
                    payload={
                        "thread_id": str(thread_id),
                        "attachment_id": str(attachment_id),
                        "user_id": str(user_id),
                        "filename": str(filename),
                        "chunk_index": idx,
                        "total_chunks": len(chunks),
                        "text": chunk_text,
                    },
                )
            )

        batch_size = 64
        for start in range(0, len(points), batch_size):
            qdrant_store.client.upsert(
                collection_name=ATTACHMENTS_COLLECTION_NAME,
                points=points[start : start + batch_size],
            )
        return len(points)
    except Exception as exc:
        logger.warning(f"[AttachmentRAG] Failed to index chunks for {filename}: {exc}")
        return 0


def retrieve_relevant_chunks(
    *,
    query: str,
    filename: str,
    chunks: List[str],
    thread_id: Optional[str] = None,
    attachment_id: Optional[str] = None,
    limit: int = 8,
    max_chars: int = 24_000,
    qdrant_store: Optional[Any] = None,
) -> str:
    """Retrieve the most relevant chunks via Qdrant vector search with BM25 fallback."""
    if not chunks:
        return ""

    selected_indices: List[int] = []
    used_semantic = False

    if (
        thread_id
        and attachment_id
        and query.strip()
        and _ensure_attachments_collection(qdrant_store)
    ):
        try:
            from qdrant_client import models

            query_vec = qdrant_store.embed_text(query)
            flt = models.Filter(
                must=[
                    models.FieldCondition(key="thread_id", match=models.MatchValue(value=str(thread_id))),
                    models.FieldCondition(key="attachment_id", match=models.MatchValue(value=str(attachment_id))),
                ]
            )
            res = qdrant_store.client.query_points(
                collection_name=ATTACHMENTS_COLLECTION_NAME,
                query=query_vec,
                using=qdrant_store.vector_name,
                query_filter=flt,
                limit=limit,
                with_payload=True,
            )
            for point in getattr(res, "points", []) or []:
                payload = point.payload or {}
                idx = payload.get("chunk_index")
                if isinstance(idx, int) and 0 <= idx < len(chunks) and idx not in selected_indices:
                    selected_indices.append(idx)
            used_semantic = bool(selected_indices)
        except Exception as exc:
            logger.warning(f"[AttachmentRAG] Semantic retrieval failed, using BM25 fallback: {exc}")

    if not selected_indices:
        selected_indices = _bm25_rank_indices(query, chunks, limit=limit)

    # Always include chunk 0 for document header/context, and sort in reading order
    if 0 not in selected_indices:
        selected_indices.insert(0, 0)
    ordered_indices = sorted(set(selected_indices[:limit]))

    mode_label = "semantic RAG" if used_semantic else "lexical RAG"
    parts: List[str] = [
        f"Attached document: {filename} ({len(chunks)} chunks total; showing {len(ordered_indices)} via {mode_label})"
    ]
    used_chars = len(parts[0])
    for idx in ordered_indices:
        block = f"[Chunk {idx + 1}/{len(chunks)}]\n{chunks[idx]}"
        if used_chars + len(block) + 2 > max_chars:
            remaining = max_chars - used_chars - 2
            if remaining > 200:
                parts.append(block[:remaining])
            break
        parts.append(block)
        used_chars += len(block) + 2

    return "\n\n".join(parts)
