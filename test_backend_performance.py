import asyncio
from unittest.mock import AsyncMock

import httpx

import mcp_client
from memory_store import MemoryStoreManager
from qdrant_vector import QdrantVectorStore


def test_qdrant_semantics_are_skipped_without_fastembed(monkeypatch):
    store = object.__new__(QdrantVectorStore)
    store._is_ready = True
    store.client = object()
    monkeypatch.setattr("qdrant_vector._FASTEMBED_AVAILABLE", False)
    monkeypatch.setattr(
        store,
        "embed_text",
        lambda _text: (_ for _ in ()).throw(AssertionError("embed_text should not run")),
    )

    assert store.supports_semantic_search is False
    assert store.search_memories("hello") == []
    assert store.upsert_memory(
        key="key",
        text="hello",
        user_id="user",
        namespace=("user", "memories"),
    ) is False


def test_agent_context_cache_is_reused_and_invalidated():
    store = MemoryStoreManager(database=None)
    store.qdrant_vector = None
    calls = {"memories": 0, "preferences": 0, "instructions": 0}

    def memories(_user_id, limit=10):
        calls["memories"] += 1
        return [{"value": {"fact": "prefers concise answers"}}]

    def preferences(_user_id):
        calls["preferences"] += 1
        return {"tone": {"value": "concise"}}

    def instructions(_user_id, limit=20):
        calls["instructions"] += 1
        return []

    store.get_user_memories = memories
    store.get_user_preferences = preferences
    store.get_user_instructions = instructions

    first = store.get_agent_context("user-1")
    second = store.get_agent_context("user-1")
    assert first == second
    assert calls == {"memories": 1, "preferences": 1, "instructions": 1}

    store.put(("user-1", "memories"), "memory-1", {"fact": "likes tables"})
    store.get_agent_context("user-1")
    assert calls == {"memories": 2, "preferences": 2, "instructions": 2}
