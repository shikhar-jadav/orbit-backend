"""
Integration test for Qdrant Cloud Vector Store with MemoryStoreManager.
Tests end-to-end storage, vector embedding generation, payload filtering, and semantic search.
"""

from dotenv import load_dotenv
load_dotenv(override=True)

import pytest
from memory_store import MemoryStoreManager
from qdrant_vector import QdrantVectorStore


def test_qdrant_vector_store_direct():
    vector_store = QdrantVectorStore()
    assert vector_store.is_ready is True

    test_user = "test_qdrant_user_123"
    test_key = "pref_code_style"
    text = "The user prefers PEP 8 formatting with 4-space indentation and type hints on every function."

    # 1. Upsert
    ok = vector_store.upsert_memory(
        key=test_key,
        text=text,
        user_id=test_user,
        namespace=(test_user, "preferences"),
        memory_type="preference",
        category="coding"
    )
    assert ok is True

    # 2. Semantic Search
    hits = vector_store.search_memories(
        query="python coding guidelines and spacing",
        user_id=test_user,
        limit=2
    )
    assert len(hits) >= 1
    top_hit = hits[0]
    assert top_hit["key"] == test_key
    assert top_hit["user_id"] == test_user
    assert "PEP 8" in top_hit["document"]
    assert top_hit["score"] > 0.3

    # 3. Cleanup
    del_ok = vector_store.delete_memory(
        key=test_key,
        user_id=test_user,
        namespace=(test_user, "preferences")
    )
    assert del_ok is True


def test_memory_store_manager_with_qdrant():
    store_mgr = MemoryStoreManager(database=None)  # Uses InMemoryStore + Qdrant
    assert store_mgr.qdrant_vector is not None
    assert store_mgr.qdrant_vector.is_ready is True

    user_id = "test_agent_user_456"

    # 1. Save an episodic memory
    ep_key = store_mgr.save_episode(
        user_id=user_id,
        task="Diagnose and resolve slow PostgreSQL queries using EXPLAIN ANALYZE",
        solution_steps="Ran EXPLAIN (ANALYZE, BUFFERS) and added a composite index on (user_id, created_at)",
        outcome="success",
        tools_used=["run_sql", "explain_query"]
    )
    assert ep_key is not None

    # 2. Semantic query matching the episode
    retrieved_episodes = store_mgr.get_relevant_episodes(
        user_id=user_id,
        query="how to optimize slow database queries with indexes",
        limit=2
    )
    assert len(retrieved_episodes) >= 1
    found_task = retrieved_episodes[0]["value"]["task"]
    assert "EXPLAIN ANALYZE" in found_task

    # 3. Clean up
    store_mgr.delete_episode(user_id=user_id, key=ep_key)
