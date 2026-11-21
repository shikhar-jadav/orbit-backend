"""
Qdrant Cloud Vector Store Integration for Orbit.

Provides production-grade vector storage, embedding generation, payload indexing,
and high-speed semantic similarity search for agent memories, instructions, and few-shot episodes.
Compatible with the official open-source Qdrant MCP server schema and collections.
"""

from __future__ import annotations

import logging
import os
import uuid
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

logger = logging.getLogger("orbit.qdrant_vector")

try:
    from qdrant_client import QdrantClient, models
    _QDRANT_CLIENT_AVAILABLE = True
except ImportError:
    _QDRANT_CLIENT_AVAILABLE = False
    QdrantClient = None
    models = None

try:
    from fastembed import TextEmbedding
    _FASTEMBED_AVAILABLE = True
except ImportError:
    _FASTEMBED_AVAILABLE = False
    TextEmbedding = None


# Match the vector specification used by mcp-server-qdrant for collection interoperability
DEFAULT_COLLECTION_NAME = "orbit_memories"
DEFAULT_VECTOR_NAME = "fast-all-minilm-l6-v2"
DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_VECTOR_SIZE = 384


class QdrantVectorStore:
    """
    Manages vector storage and similarity retrieval using Qdrant Cloud.
    """

    _instance: Optional["QdrantVectorStore"] = None
    _embedder: Optional[Any] = None

    def __init__(
        self,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        collection_name: Optional[str] = None,
    ):
        load_dotenv(override=False)
        self.url = (url or os.getenv("QDRANT_URL") or "").strip()
        self.api_key = (api_key or os.getenv("QDRANT_API_KEY") or "").strip()
        self.collection_name = (
            collection_name
            or os.getenv("QDRANT_COLLECTION_NAME")
            or DEFAULT_COLLECTION_NAME
        ).strip()
        self.vector_name = DEFAULT_VECTOR_NAME
        self.client: Optional[QdrantClient] = None
        self._is_ready = False

        if not _QDRANT_CLIENT_AVAILABLE:
            logger.warning("[Qdrant] qdrant-client package is not installed.")
            return

        if not self.url or not self.api_key:
            logger.debug("[Qdrant] QDRANT_URL or QDRANT_API_KEY not provided.")
            return

        try:
            self.client = QdrantClient(
                url=self.url,
                api_key=self.api_key,
                timeout=10.0,
            )
            self._ensure_collection_and_indexes()
            self._is_ready = True
            logger.info(f"[Qdrant] Initialized vector store at {self.url} (collection: {self.collection_name}).")
        except Exception as e:
            logger.warning(f"[Qdrant] Failed to connect to Qdrant cluster: {e}")
            self.client = None
            self._is_ready = False

    @property
    def is_ready(self) -> bool:
        return self._is_ready and self.client is not None

    def _get_embedder(self):
        """Lazy-initialize singleton FastEmbed model."""
        if QdrantVectorStore._embedder is None and _FASTEMBED_AVAILABLE:
            try:
                QdrantVectorStore._embedder = TextEmbedding(model_name=DEFAULT_MODEL_NAME)
            except Exception as e:
                logger.error(f"[Qdrant] Error initializing FastEmbed: {e}")
        return QdrantVectorStore._embedder

    def embed_text(self, text: str) -> List[float]:
        """Convert a string into a 384-dimensional dense float vector."""
        embedder = self._get_embedder()
        if embedder is None:
            raise RuntimeError("FastEmbed embedder is not available.")
        vectors = list(embedder.embed([text]))
        if not vectors:
            raise RuntimeError("Failed to generate embedding vector.")
        return vectors[0].tolist()

    def _ensure_collection_and_indexes(self) -> None:
        """Create the collection and required payload keyword indexes if not existing."""
        if not self.client:
            return

        collections_res = self.client.get_collections()
        existing_names = {c.name for c in collections_res.collections}

        if self.collection_name not in existing_names:
            logger.info(f"[Qdrant] Creating collection '{self.collection_name}'...")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    self.vector_name: models.VectorParams(
                        size=DEFAULT_VECTOR_SIZE,
                        distance=models.Distance.COSINE,
                    )
                },
            )

        # Ensure essential payload fields are indexed for rapid metadata filtering
        indexed_fields = [
            "metadata.user_id",
            "metadata.category",
            "metadata.namespace",
            "metadata.key",
            "metadata.memory_type",
        ]
        for f in indexed_fields:
            try:
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=f,
                    field_schema=models.PayloadSchemaType.KEYWORD,
                )
            except Exception:
                pass  # Index already exists or supported

    @staticmethod
    def _deterministic_point_id(user_id: str, namespace_str: str, key: str) -> str:
        """Generate a stable UUID-v5 for idempotent upserts and updates."""
        combined = f"{user_id}::{namespace_str}::{key}"
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, combined))

    def upsert_memory(
        self,
        key: str,
        text: str,
        user_id: str,
        namespace: Tuple[str, ...],
        memory_type: str = "fact",
        category: str = "general",
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Embed and upsert a memory record into Qdrant Cloud.
        """
        if not self.is_ready or not text.strip():
            return False

        try:
            vec = self.embed_text(text)
            ns_list = list(namespace)
            ns_str = ".".join(ns_list)
            point_id = self._deterministic_point_id(user_id, ns_str, key)

            metadata_payload = {
                "user_id": str(user_id),
                "namespace": ns_list,
                "namespace_str": ns_str,
                "key": str(key),
                "memory_type": str(memory_type),
                "category": str(category),
            }
            if extra_metadata and isinstance(extra_metadata, dict):
                metadata_payload.update(extra_metadata)

            payload = {
                "document": text.strip(),
                "metadata": metadata_payload,
            }

            self.client.upsert(
                collection_name=self.collection_name,
                points=[
                    models.PointStruct(
                        id=point_id,
                        vector={self.vector_name: vec},
                        payload=payload,
                    )
                ],
            )
            return True
        except Exception as e:
            logger.warning(f"[Qdrant] Error upserting memory '{key}': {e}")
            return False

    def search_memories(
        self,
        query: str,
        user_id: Optional[str] = None,
        namespace: Optional[Tuple[str, ...]] = None,
        memory_type: Optional[str] = None,
        category: Optional[str] = None,
        limit: int = 5,
        score_threshold: float = 0.25,
    ) -> List[Dict[str, Any]]:
        """
        Perform vector cosine similarity search in Qdrant Cloud with metadata filtering.
        """
        if not self.is_ready or not query.strip():
            return []

        try:
            vec = self.embed_text(query)
            must_conditions: List[models.FieldCondition] = []

            if user_id:
                must_conditions.append(
                    models.FieldCondition(
                        key="metadata.user_id",
                        match=models.MatchValue(value=str(user_id)),
                    )
                )

            if namespace:
                ns_list = list(namespace)
                must_conditions.append(
                    models.FieldCondition(
                        key="metadata.namespace",
                        match=models.MatchAny(any=ns_list),
                    )
                )

            if memory_type and memory_type != "all":
                must_conditions.append(
                    models.FieldCondition(
                        key="metadata.memory_type",
                        match=models.MatchValue(value=str(memory_type)),
                    )
                )

            if category:
                must_conditions.append(
                    models.FieldCondition(
                        key="metadata.category",
                        match=models.MatchValue(value=str(category)),
                    )
                )

            query_filter = models.Filter(must=must_conditions) if must_conditions else None

            res = self.client.query_points(
                collection_name=self.collection_name,
                query=vec,
                using=self.vector_name,
                query_filter=query_filter,
                score_threshold=score_threshold,
                limit=limit,
            )

            hits = []
            for p in res.points:
                payload = p.payload or {}
                meta = payload.get("metadata", {})
                hits.append({
                    "id": p.id,
                    "score": round(float(p.score), 4),
                    "document": payload.get("document", ""),
                    "key": meta.get("key", ""),
                    "user_id": meta.get("user_id", ""),
                    "namespace": tuple(meta.get("namespace", [])),
                    "memory_type": meta.get("memory_type", "fact"),
                    "category": meta.get("category", "general"),
                    "metadata": meta,
                })
            return hits
        except Exception as e:
            logger.warning(f"[Qdrant] Error searching memories for '{query}': {e}")
            return []

    def delete_memory(
        self,
        key: str,
        user_id: str,
        namespace: Tuple[str, ...],
    ) -> bool:
        """Delete a memory point by key and namespace from Qdrant."""
        if not self.is_ready:
            return False

        try:
            ns_str = ".".join(list(namespace))
            point_id = self._deterministic_point_id(user_id, ns_str, key)
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.PointIdsList(points=[point_id]),
            )
            return True
        except Exception as e:
            logger.warning(f"[Qdrant] Error deleting memory '{key}': {e}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """Return collection and cluster statistics."""
        if not self.is_ready:
            return {"ready": False, "error": "Not connected to Qdrant cluster"}

        try:
            info = self.client.get_collection(self.collection_name)
            return {
                "ready": True,
                "url": self.url,
                "collection": self.collection_name,
                "status": str(info.status.value) if hasattr(info.status, "value") else str(info.status),
                "points_count": info.points_count,
                "indexed_vectors_count": info.indexed_vectors_count,
                "vector_name": self.vector_name,
                "vector_size": DEFAULT_VECTOR_SIZE,
            }
        except Exception as e:
            return {"ready": False, "error": str(e)}
