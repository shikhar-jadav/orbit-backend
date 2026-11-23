"""
LangGraph Production-Grade Long-Term Memory Store Module.

Provides cross-thread persistent memory using LangGraph's official Store abstraction:
- Direct PostgreSQL backing via `langgraph.store.postgres.PostgresStore` (zero in-memory drift across workers)
- Automatic schema creation and migrations via `store.setup()`
- Vector similarity search when embeddings are configured (e.g. OpenAIEmbeddings + pgvector)
- Graceful keyword search fallback when running without embeddings
- User-scoped namespaces: (user_id, "memories"), (user_id, "preferences")
- Cross-user shared knowledge namespaces: ("shared", topic)
- Backward compatibility: automatically migrates any records from legacy 'memory_store' table

Based on official LangGraph Stores documentation:
https://docs.langchain.com/oss/python/langgraph/stores
"""

from __future__ import annotations

import asyncio
import contextvars
import copy
import json
import logging
import os
import time
from threading import Lock
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Sequence
from uuid import uuid4

import psycopg
from psycopg.rows import dict_row

from langgraph.store.base import BaseStore, Item
from langgraph.store.memory import InMemoryStore
from langgraph.store.postgres import PostgresStore

from database import PostgresDatabase

logger = logging.getLogger("orbit.memory_store")

# Context vars for thread/user scoped tool and runtime access
active_user_id_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "active_user_id", default=None
)
active_memory_store_var: contextvars.ContextVar[Optional[Any]] = contextvars.ContextVar(
    "active_memory_store", default=None
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class MemoryStoreManager:
    """
    Production-grade Long-Term Memory Store Manager.

    Uses LangGraph's official PostgresStore for persistence directly in PostgreSQL,
    avoiding in-memory heap bloat and ensuring multi-worker cluster synchronization.
    Falls back gracefully to InMemoryStore when running without an active database.
    """

    def __init__(
        self,
        database: Optional[PostgresDatabase] = None,
        *,
        index_config: Optional[Any] = None,
    ):
        self.database = database
        self.index_config = index_config
        self.store: BaseStore
        self._dedicated_conn: Optional[psycopg.Connection] = None
        self._is_postgres = False

        # Qdrant is an optional search accelerator. Do not connect during
        # application startup: its remote network handshake can take seconds.
        # The server lifespan warms it in a background thread after startup.
        self._qdrant_vector = None
        self._qdrant_init_attempted = False
        self._qdrant_init_lock = Lock()
        self._agent_context_cache: Dict[str, Tuple[float, Dict[str, Any]]] = {}
        self._agent_context_cache_lock = Lock()

        if database is not None:
            self._init_postgres(database, index_config)
        else:
            self.store = InMemoryStore()
            self._is_postgres = False

    @property
    def qdrant_vector(self):
        """Lazily initialize the optional Qdrant accelerator on first use."""
        return self.initialize_qdrant()

    @qdrant_vector.setter
    def qdrant_vector(self, value):
        self._qdrant_vector = value
        self._qdrant_init_attempted = True

    def initialize_qdrant(self):
        """Connect to Qdrant once; safe to call from a background worker."""
        if self._qdrant_init_attempted:
            return self._qdrant_vector
        with self._qdrant_init_lock:
            if self._qdrant_init_attempted:
                return self._qdrant_vector
            self._qdrant_init_attempted = True
            try:
                from qdrant_vector import QdrantVectorStore

                vector_store = QdrantVectorStore()
                self._qdrant_vector = vector_store if vector_store.is_ready else None
                if self._qdrant_vector is not None:
                    print(
                        f"[INIT] Connected to Qdrant Cloud vector store "
                        f"({self._qdrant_vector.collection_name})."
                    )
            except Exception as q_err:
                logger.warning(
                    f"[MemoryStore] Qdrant unavailable; using PostgreSQL/keyword memory search ({q_err})"
                )
                self._qdrant_vector = None
            return self._qdrant_vector

    def _init_postgres(self, database: PostgresDatabase, index_config: Optional[Any]) -> None:
        """Initialize PostgresStore with dedicated connection and migrations."""
        try:
            # Dedicated autocommit connection for DDL migrations and PostgresStore
            self._dedicated_conn = database.connect(
                autocommit=True, row_factory=dict_row
            )

            # Try initializing with vector index if provided
            if index_config is not None:
                try:
                    self.store = PostgresStore(self._dedicated_conn, index=index_config)
                    self.store.setup()
                    self._is_postgres = True
                    logger.info("[MemoryStore] Initialized PostgresStore with vector search index.")
                except Exception as vec_err:
                    logger.warning(
                        f"[MemoryStore] Vector index setup failed ({vec_err}); falling back to standard PostgresStore."
                    )
                    self.store = PostgresStore(self._dedicated_conn, index=None)
                    self.store.setup()
                    self._is_postgres = True
            else:
                self.store = PostgresStore(self._dedicated_conn, index=None)
                self.store.setup()
                self._is_postgres = True
                logger.info("[MemoryStore] Initialized standard PostgresStore with JSONB indexing.")

            # Migrate any legacy records from 'memory_store' table
            self._migrate_legacy_data()

        except Exception as e:
            logger.error(f"[MemoryStore] Failed to initialize PostgresStore ({e}); falling back to InMemoryStore.")
            if self._dedicated_conn is not None:
                try:
                    self._dedicated_conn.close()
                except Exception:
                    pass
                self._dedicated_conn = None
            self.store = InMemoryStore()
            self._is_postgres = False

    def _migrate_legacy_data(self) -> None:
        """Migrate existing records from legacy 'memory_store' table to LangGraph 'store'."""
        if not self._is_postgres or not self._dedicated_conn:
            return
        try:
            with self._dedicated_conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT EXISTS (
                        SELECT 1 FROM information_schema.tables 
                        WHERE table_schema = current_schema() AND table_name = 'memory_store'
                    ) AS exists
                    """
                )
                row = cur.fetchone()
                if not row or not row["exists"]:
                    return

                cur.execute("SELECT namespace, key, value FROM memory_store")
                legacy_rows = cur.fetchall()
                if not legacy_rows:
                    return

                migrated = 0
                for r in legacy_rows:
                    try:
                        ns = tuple(json.loads(r["namespace"]))
                        val = r["value"]
                        if isinstance(val, str):
                            val = json.loads(val)
                        if isinstance(val, dict):
                            self.store.put(ns, str(r["key"]), val)
                            migrated += 1
                    except Exception:
                        continue
                if migrated > 0:
                    logger.info(f"[MemoryStore] Migrated {migrated} memories from legacy 'memory_store' table.")
        except Exception as e:
            logger.debug(f"[MemoryStore] Legacy migration skipped: {e}")

    def close(self) -> None:
        """Close dedicated database connection if open."""
        if self._dedicated_conn is not None:
            try:
                self._dedicated_conn.close()
            except Exception:
                pass
            self._dedicated_conn = None

    # ── Core LangGraph BaseStore API ────────────────────────────────────

    def put(
        self,
        namespace: Tuple[str, ...],
        key: str,
        value: Dict[str, Any],
        index: Optional[Any] = None,
    ) -> None:
        """Store or overwrite a memory item in PostgreSQL."""
        self.store.put(namespace, str(key), value, index=index)
        self._invalidate_agent_context_cache(namespace)
        if getattr(self, "qdrant_vector", None) and self.qdrant_vector.supports_semantic_search and value:
            try:
                user_id = namespace[0] if namespace else "default_user"
                mem_type = namespace[1] if len(namespace) > 1 else "fact"
                text_to_embed = ""
                category = "general"
                if isinstance(value, dict):
                    text_to_embed = (
                        value.get("fact")
                        or value.get("rule")
                        or value.get("task")
                        or value.get("content")
                        or value.get("solution_steps")
                        or value.get("value")
                        or json.dumps(value, ensure_ascii=False)
                    )
                    category = value.get("category", "general")
                else:
                    text_to_embed = str(value)

                if text_to_embed:
                    self.qdrant_vector.upsert_memory(
                        key=str(key),
                        text=str(text_to_embed),
                        user_id=user_id,
                        namespace=namespace,
                        memory_type=mem_type,
                        category=category,
                        extra_metadata=value if isinstance(value, dict) else {},
                    )
            except Exception as e:
                logger.debug(f"[MemoryStore] Qdrant sync put skipped: {e}")

    def get(
        self, namespace: Tuple[str, ...], key: str
    ) -> Optional[Dict[str, Any]]:
        """Get a single memory item's value by key. Returns None if not found."""
        item = self.store.get(namespace, str(key))
        if item is None:
            return None
        return item.value if hasattr(item, "value") else item

    def get_item(
        self, namespace: Tuple[str, ...], key: str
    ) -> Optional[Item]:
        """Get the full Item object including timestamps, namespace, key, and value."""
        return self.store.get(namespace, str(key))

    def delete(self, namespace: Tuple[str, ...], key: str) -> None:
        """Delete a single memory item from PostgreSQL."""
        self.store.delete(namespace, str(key))
        self._invalidate_agent_context_cache(namespace)
        if getattr(self, "qdrant_vector", None) and self.qdrant_vector.is_ready:
            try:
                user_id = namespace[0] if namespace else "default_user"
                self.qdrant_vector.delete_memory(str(key), user_id, namespace)
            except Exception as e:
                logger.debug(f"[MemoryStore] Qdrant sync delete skipped: {e}")

    def search(
        self,
        namespace_prefix: Tuple[str, ...],
        query: Optional[str] = None,
        filter: Optional[Dict[str, Any]] = None,
        limit: int = 10,
        offset: int = 0,
    ) -> List[Any]:
        """
        Search memories under a namespace prefix.

        If a vector index is configured, semantic vector search is executed.
        If no vector index is active, performs natural keyword filtering on content.
        """
        candidate_limit = min(max(int(limit), 50), 200) if query else limit
        kwargs: Dict[str, Any] = {"limit": candidate_limit, "offset": offset}
        if filter:
            kwargs["filter"] = filter

        # Vector search if index configured
        # 1. Semantic Vector Search via Qdrant Cloud
        if query and getattr(self, "qdrant_vector", None) and self.qdrant_vector.supports_semantic_search:
            try:
                user_id = namespace_prefix[0] if namespace_prefix else None
                mem_type = namespace_prefix[1] if len(namespace_prefix) > 1 else None
                hits = self.qdrant_vector.search_memories(
                    query=query,
                    user_id=user_id,
                    namespace=namespace_prefix,
                    memory_type=mem_type,
                    limit=limit,
                )
                if hits:
                    matched_items = []
                    for h in hits:
                        hit_key = h.get("key")
                        if hit_key:
                            pg_item = self.store.get(namespace_prefix, str(hit_key))
                            if pg_item:
                                matched_items.append(pg_item)
                                continue
                        matched_items.append(
                            Item(
                                value={"content": h["document"], **h.get("metadata", {})},
                                key=h.get("key") or h.get("id", ""),
                                namespace=namespace_prefix,
                                created_at=datetime.now(timezone.utc),
                                updated_at=datetime.now(timezone.utc),
                            )
                        )
                    if matched_items:
                        return matched_items
            except Exception as q_err:
                logger.warning(f"[MemoryStore] Qdrant search fallback: {q_err}")

        # 2. Vector search if store index configured
        if query and getattr(self.store, "index_config", None):
            kwargs["query"] = query
            return self.store.search(namespace_prefix, **kwargs)

        # Standard search or keyword fallback
        results = self.store.search(namespace_prefix, **kwargs)
        if query:
            q_lower = query.lower().strip()
            q_tokens = [t for t in q_lower.split() if len(t) > 1]
            filtered = []
            for item in results:
                val = getattr(item, "value", item)
                val_text = (
                    json.dumps(val, ensure_ascii=False).lower()
                    if isinstance(val, (dict, list))
                    else str(val).lower()
                )
                if q_lower in val_text or any(token in val_text for token in q_tokens):
                    filtered.append(item)
            return filtered[:limit]
        return results

    def list_namespaces(
        self,
        prefix: Optional[Tuple[str, ...]] = None,
        max_depth: Optional[int] = None,
    ) -> List[Tuple[str, ...]]:
        """List all namespaces matching optional prefix."""
        kwargs: Dict[str, Any] = {}
        if prefix:
            kwargs["prefix"] = prefix
        if max_depth:
            kwargs["max_depth"] = max_depth
        return self.store.list_namespaces(**kwargs)

    # ── Async Counterparts conforming to BaseStore Contract ─────────────

    async def aput(
        self,
        namespace: Tuple[str, ...],
        key: str,
        value: Dict[str, Any],
        index: Optional[Any] = None,
    ) -> None:
        """Async store or overwrite a single item in PostgreSQL."""
        await self.store.aput(namespace, str(key), value, index=index)

    async def aget(
        self, namespace: Tuple[str, ...], key: str
    ) -> Optional[Dict[str, Any]]:
        """Async retrieve a single item by key."""
        item = await self.store.aget(namespace, str(key))
        if item is None:
            return None
        return item.value if hasattr(item, "value") else item

    async def adelete(self, namespace: Tuple[str, ...], key: str) -> None:
        """Async delete a single item from PostgreSQL."""
        await self.store.adelete(namespace, str(key))

    async def asearch(
        self,
        namespace_prefix: Tuple[str, ...],
        *,
        query: Optional[str] = None,
        filter: Optional[Dict[str, Any]] = None,
        limit: int = 10,
        offset: int = 0,
    ) -> List[Any]:
        """Async search items under a namespace prefix with vector or keyword search."""
        # 1. Semantic Vector Search via Qdrant Cloud
        if query and getattr(self, "qdrant_vector", None) and self.qdrant_vector.supports_semantic_search:
            try:
                user_id = namespace_prefix[0] if namespace_prefix else None
                mem_type = namespace_prefix[1] if len(namespace_prefix) > 1 else None
                hits = self.qdrant_vector.search_memories(
                    query=query,
                    user_id=user_id,
                    namespace=namespace_prefix,
                    memory_type=mem_type,
                    limit=limit,
                )
                if hits:
                    matched_items = []
                    for h in hits:
                        hit_key = h.get("key")
                        if hit_key:
                            pg_item = self.store.get(namespace_prefix, str(hit_key))
                            if pg_item:
                                matched_items.append(pg_item)
                                continue
                        matched_items.append(
                            Item(
                                value={"content": h["document"], **h.get("metadata", {})},
                                key=h.get("key") or h.get("id", ""),
                                namespace=namespace_prefix,
                                created_at=datetime.now(timezone.utc),
                                updated_at=datetime.now(timezone.utc),
                            )
                        )
                    if matched_items:
                        return matched_items
            except Exception as q_err:
                logger.warning(f"[MemoryStore] Qdrant search fallback: {q_err}")

        # 2. Vector search if store index configured
        if query and getattr(self.store, "index_config", None):
            return await self.store.asearch(
                namespace_prefix, query=query, filter=filter, limit=limit, offset=offset
            )
        results = await self.store.asearch(
            namespace_prefix, filter=filter, limit=limit, offset=offset
        )
        if query:
            q_lower = query.lower().strip()
            q_tokens = [t for t in q_lower.split() if len(t) > 1]
            return [
                item
                for item in results
                if (
                    q_lower in json.dumps(getattr(item, "value", item), ensure_ascii=False).lower()
                    or any(token in json.dumps(getattr(item, "value", item), ensure_ascii=False).lower() for token in q_tokens)
                )
            ]
        return results

    async def alist_namespaces(
        self,
        *,
        prefix: Optional[Tuple[str, ...]] = None,
        suffix: Optional[Tuple[str, ...]] = None,
        max_depth: Optional[int] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Tuple[str, ...]]:
        """Async list namespaces matching prefix/suffix."""
        kwargs: Dict[str, Any] = {"limit": limit, "offset": offset}
        if prefix:
            kwargs["prefix"] = prefix
        if suffix:
            kwargs["suffix"] = suffix
        if max_depth:
            kwargs["max_depth"] = max_depth
        return await self.store.alist_namespaces(**kwargs)

    # ── Convenience Helpers for Agent Integration ───────────────────────

    def get_user_memories(
        self, user_id: str, limit: int = 20
    ) -> List[Dict[str, Any]]:
        """Get all memories for a user."""
        items = self.search((user_id, "memories"), limit=limit)
        return [
            {
                "key": item.key,
                "value": item.value,
                "namespace": list(item.namespace),
                "created_at": (
                    item.created_at.isoformat()
                    if hasattr(item, "created_at") and isinstance(item.created_at, datetime)
                    else str(getattr(item, "created_at", "") or "")
                ),
                "updated_at": (
                    item.updated_at.isoformat()
                    if hasattr(item, "updated_at") and isinstance(item.updated_at, datetime)
                    else str(getattr(item, "updated_at", "") or "")
                ),
            }
            for item in items
        ]

    def save_user_memory(
        self, user_id: str, memory_data: Dict[str, Any], key: Optional[str] = None
    ) -> str:
        """
        Save a memory for a user. Returns the unique memory key.
        """
        memory_key = key or str(uuid4())
        namespace = (user_id, "memories")
        self.put(namespace, memory_key, memory_data)
        return memory_key

    def delete_user_memory(self, user_id: str, key: str) -> None:
        """Delete a specific memory for a user."""
        self.delete((user_id, "memories"), key)

    def get_user_preferences(self, user_id: str) -> Dict[str, Any]:
        """Get user preferences from the store."""
        items = self.search((user_id, "preferences"), limit=50)
        prefs = {}
        for item in items:
            prefs[item.key] = item.value
        return prefs

    def save_user_preference(
        self, user_id: str, pref_key: str, pref_value: Any
    ) -> None:
        """Save a single user preference."""
        namespace = (user_id, "preferences")
        self.put(namespace, pref_key, {"value": pref_value})

    def get_shared_knowledge(self, topic: str, limit: int = 10) -> List[Any]:
        """Get shared knowledge items for a topic (cross-user)."""
        return self.search(("shared", topic), limit=limit)

    def save_shared_knowledge(
        self, topic: str, data: Dict[str, Any], key: Optional[str] = None
    ) -> str:
        """Save shared knowledge item. Returns the key."""
        knowledge_key = key or str(uuid4())
        self.put(("shared", topic), knowledge_key, data)
        return knowledge_key

    # ── Procedural Memory Helpers (Learned Rules & Instructions) ────────

    def save_user_instruction(
        self,
        user_id: str,
        rule: str,
        category: str = "general",
        key: Optional[str] = None,
    ) -> str:
        """
        Save or update a procedural rule / behavioral instruction learned from feedback.
        Stored under namespace: (user_id, "instructions").
        """
        rule_key = key or f"rule_{uuid4().hex[:8]}"
        namespace = (user_id, "instructions")
        data = {
            "rule": rule,
            "category": category,
            "updated_at": _utc_now(),
        }
        self.put(namespace, rule_key, data)
        return rule_key

    def get_agent_context(self, user_id: str, ttl_seconds: float = 30.0) -> Dict[str, Any]:
        """Return stable user memory context with a short per-user cache."""
        now = time.monotonic()
        with self._agent_context_cache_lock:
            cached = self._agent_context_cache.get(user_id)
            if cached and now < cached[0]:
                return copy.deepcopy(cached[1])

        context = {
            "memories": self.get_user_memories(user_id, limit=10),
            "preferences": self.get_user_preferences(user_id),
            "instructions": self.get_user_instructions(user_id, limit=20),
        }
        with self._agent_context_cache_lock:
            self._agent_context_cache[user_id] = (time.monotonic() + ttl_seconds, context)
        return copy.deepcopy(context)

    def _invalidate_agent_context_cache(self, namespace: Tuple[str, ...]) -> None:
        if len(namespace) < 2 or namespace[1] not in {"memories", "preferences", "instructions"}:
            return
        with self._agent_context_cache_lock:
            self._agent_context_cache.pop(namespace[0], None)

    def get_user_instructions(
        self, user_id: str, limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Retrieve all active procedural rules and instructions for a user.
        """
        items = self.search((user_id, "instructions"), limit=limit)
        return [
            {
                "key": item.key,
                "rule": item.value.get("rule", "") if isinstance(item.value, dict) else str(item.value),
                "category": item.value.get("category", "general") if isinstance(item.value, dict) else "general",
                "namespace": list(item.namespace),
                "created_at": (
                    item.created_at.isoformat()
                    if hasattr(item, "created_at") and isinstance(item.created_at, datetime)
                    else str(getattr(item, "created_at", "") or "")
                ),
                "updated_at": (
                    item.updated_at.isoformat()
                    if hasattr(item, "updated_at") and isinstance(item.updated_at, datetime)
                    else str(getattr(item, "updated_at", "") or "")
                ),
            }
            for item in items
        ]

    def delete_user_instruction(self, user_id: str, key: str) -> None:
        """Delete an outdated or contradicted procedural rule."""
        self.delete((user_id, "instructions"), key)

    # ── Episodic Memory Helpers (Problem-Solving & Few-Shot Ingestion) ──

    def save_episode(
        self,
        user_id: str,
        task: str,
        solution_steps: Any,
        outcome: str = "success",
        tools_used: Optional[List[str]] = None,
        key: Optional[str] = None,
    ) -> str:
        """
        Save a problem-solving episode (task, strategy, tools, outcome) for few-shot learning.
        Stored under namespace: (user_id, "episodes").
        """
        ep_key = key or f"ep_{uuid4().hex[:8]}"
        namespace = (user_id, "episodes")
        data = {
            "task": task,
            "solution_steps": solution_steps,
            "outcome": outcome,
            "tools_used": tools_used or [],
            "saved_at": _utc_now(),
        }
        self.put(namespace, ep_key, data)
        return ep_key

    def get_relevant_episodes(
        self, user_id: str, query: str = "", limit: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant past problem-solving episodes matching a task query.
        """
        items = self.search(
            (user_id, "episodes"),
            query=query if query else None,
            limit=limit,
        )
        return [
            {
                "key": item.key,
                "value": item.value,
                "namespace": list(item.namespace),
                "created_at": (
                    item.created_at.isoformat()
                    if hasattr(item, "created_at") and isinstance(item.created_at, datetime)
                    else str(getattr(item, "created_at", "") or "")
                ),
                "updated_at": (
                    item.updated_at.isoformat()
                    if hasattr(item, "updated_at") and isinstance(item.updated_at, datetime)
                    else str(getattr(item, "updated_at", "") or "")
                ),
            }
            for item in items
        ]

    def delete_episode(self, user_id: str, key: str) -> None:
        """Delete an episodic memory by key."""
        self.delete((user_id, "episodes"), key)

    def update_memory_item(
        self,
        namespace: Tuple[str, ...],
        key: str,
        patch_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Update or patch an existing item in any namespace.
        Merges new keys into existing dictionary value.
        """
        existing = self.get(namespace, key) or {}
        if isinstance(existing, dict):
            updated = {**existing, **patch_data, "updated_at": _utc_now()}
        else:
            updated = patch_data
        self.put(namespace, key, updated)
        return updated


# ── Factory Function ────────────────────────────────────────────────────

def create_memory_store(
    database: Optional[PostgresDatabase] = None,
) -> MemoryStoreManager:
    """
    Factory to create a production-grade MemoryStoreManager.
    Backed directly by Neon PostgreSQL without external vector stores or embeddings.
    """
    return MemoryStoreManager(database=database, index_config=None)
