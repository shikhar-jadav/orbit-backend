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


def test_mcp_session_cache_is_scoped_to_credentials():
    url = "https://mcp.example.com/mcp"
    user_a = mcp_client._mcp_session_cache_key(
        url, {"Authorization": "Bearer user-a-secret"}
    )
    user_b = mcp_client._mcp_session_cache_key(
        url, {"Authorization": "Bearer user-b-secret"}
    )

    assert user_a != user_b
    assert "user-a-secret" not in user_a
    assert "user-b-secret" not in user_b


def test_mcp_tool_call_reuses_cached_session(monkeypatch):
    async def run():
        url = "https://mcp.example.com/mcp"
        headers = mcp_client._base_mcp_headers({"Authorization": "Bearer test-user"})
        key = mcp_client._mcp_session_cache_key(url, headers)
        monkeypatch.setattr(mcp_client, "_MCP_SESSION_CACHE", {})
        mcp_client._cache_mcp_session(key, "session-123")
        monkeypatch.setattr(mcp_client, "_get_shared_mcp_http_client", lambda: object())

        response = {
            "result": {
                "content": [{"type": "text", "text": "tool result"}],
            }
        }
        post = AsyncMock(
            return_value=(200, "OK", httpx.Headers(), response)
        )
        initialize = AsyncMock()
        monkeypatch.setattr(mcp_client, "_post_mcp_jsonrpc", post)
        monkeypatch.setattr(mcp_client, "_init_streamable_session", initialize)

        result = await mcp_client.call_mcp_tool(
            url,
            "list_items",
            {},
            custom_headers={"Authorization": "Bearer test-user"},
        )

        assert "tool result" in result
        initialize.assert_not_awaited()
        post.assert_awaited_once()
        call = post.await_args
        assert call.args[2]["Mcp-Session-Id"] == "session-123"
        assert call.kwargs["rpc_id"] == 3

    asyncio.run(run())
