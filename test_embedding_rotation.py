"""
Unit tests for RoundRobinEmbeddingManager and RotatingEmbeddings.
Tests multi-provider discovery, round-robin rotation, rate-limit cooldown, and automatic failover.
"""

import asyncio
from unittest.mock import MagicMock, patch
import pytest

from embedding_manager import (
    EmbeddingProviderConfig,
    RoundRobinEmbeddingManager,
    RotatingEmbeddings,
    _is_rate_limit_error,
)


MOCK_EMBEDDING_ENV = {
    "EMBEDDING_PROVIDER_NAME1": "Voyage-AI",
    "EMBEDDING_API_KEY1": "pa-test-voyage-12345",
    "EMBEDDING_BASE_URL1": "https://api.voyageai.com/v1",
    "EMBEDDING_MODEL1": "voyage-3",
    "EMBEDDING_DIMS1": "1024",

    "EMBEDDING_PROVIDER_NAME2": "OpenAI-Account",
    "EMBEDDING_API_KEY2": "sk-test-openai-67890",
    "EMBEDDING_BASE_URL2": "https://api.openai.com/v1",
    "EMBEDDING_MODEL2": "text-embedding-3-small",
    "EMBEDDING_DIMS2": "1536",

    "EMBEDDING_PROVIDER_NAME3": "Together-Embed",
    "EMBEDDING_API_KEY3": "tog-test-together-99999",
    "EMBEDDING_BASE_URL3": "https://api.together.xyz/v1",
    "EMBEDDING_MODEL3": "BAAI/bge-large-en-v1.5",
}


def test_provider_discovery_and_metadata():
    manager = RoundRobinEmbeddingManager(env_dict=MOCK_EMBEDDING_ENV)
    assert manager.get_provider_count() == 3

    # Provider 1
    p1 = manager.providers[0]
    assert p1.name == "Voyage-AI"
    assert p1.api_key == "pa-test-voyage-12345"
    assert p1.base_url == "https://api.voyageai.com/v1"
    assert p1.default_model == "voyage-3"
    assert p1.dimensions == 1024
    assert p1.masked_key() == "pa-t...2345"

    # Provider 2
    p2 = manager.providers[1]
    assert p2.name == "OpenAI-Account"
    assert p2.api_key == "sk-test-openai-67890"
    assert p2.default_model == "text-embedding-3-small"
    assert p2.dimensions == 1536

    # Provider 3
    p3 = manager.providers[2]
    assert p3.name == "Together-Embed"
    assert p3.dimensions is None


def test_round_robin_rotation():
    manager = RoundRobinEmbeddingManager(env_dict=MOCK_EMBEDDING_ENV)

    p1 = manager.get_next_provider()
    assert p1.name == "Voyage-AI"
    assert p1.call_count == 1

    p2 = manager.get_next_provider()
    assert p2.name == "OpenAI-Account"
    assert p2.call_count == 1

    p3 = manager.get_next_provider()
    assert p3.name == "Together-Embed"
    assert p3.call_count == 1

    # Wraparound to Provider 1
    p4 = manager.get_next_provider()
    assert p4.name == "Voyage-AI"
    assert p4.call_count == 2


def test_rate_limit_cooldown_skips_provider():
    manager = RoundRobinEmbeddingManager(env_dict=MOCK_EMBEDDING_ENV)

    # Mark provider 1 as cooling down
    manager.mark_rate_limited(manager.providers[0], retry_after=10.0)
    assert manager.providers[0].cooldown_until > 0

    # Next call should skip Provider 1 and return Provider 2
    picked = manager.get_next_provider()
    assert picked.name == "OpenAI-Account"


def test_single_unnumbered_env():
    single_env = {
        "EMBEDDING_API_KEY": "sk-single-test",
        "EMBEDDING_BASE_URL": "https://custom.embed.api/v1",
        "EMBEDDING_MODEL": "bge-m3",
        "EMBEDDING_DIMS": "1024",
    }
    manager = RoundRobinEmbeddingManager(env_dict=single_env)
    assert manager.get_provider_count() == 1
    p = manager.providers[0]
    assert p.api_key == "sk-single-test"
    assert p.base_url == "https://custom.embed.api/v1"
    assert p.default_model == "bge-m3"
    assert p.dimensions == 1024


def test_status_endpoint_data():
    manager = RoundRobinEmbeddingManager(env_dict=MOCK_EMBEDDING_ENV)
    manager.get_next_provider()
    status = manager.get_status()
    assert status["total_providers"] == 3
    assert status["total_requests_served"] == 1
    assert len(status["providers"]) == 3
    assert status["providers"][0]["call_count"] == 1


def test_rotating_embeddings_normal():
    manager = RoundRobinEmbeddingManager(env_dict=MOCK_EMBEDDING_ENV)
    embeddings = RotatingEmbeddings(manager=manager)

    fake_client = MagicMock()
    fake_client.embed_documents.return_value = [[0.1, 0.2, 0.3]]
    fake_client.embed_query.return_value = [0.1, 0.2, 0.3]

    with patch.object(manager, "create_embeddings_client", return_value=fake_client):
        docs = embeddings.embed_documents(["Hello world"])
        assert docs == [[0.1, 0.2, 0.3]]

        query = embeddings.embed_query("Search text")
        assert query == [0.1, 0.2, 0.3]


def test_rotating_embeddings_failover_on_429():
    manager = RoundRobinEmbeddingManager(env_dict=MOCK_EMBEDDING_ENV)
    embeddings = RotatingEmbeddings(manager=manager)

    p1_client = MagicMock()
    p1_client.embed_documents.side_effect = RuntimeError("Error code: 429 - Rate limit reached")

    p2_client = MagicMock()
    p2_client.embed_documents.return_value = [[0.4, 0.5, 0.6]]

    clients = [p1_client, p2_client]

    with patch.object(manager, "create_embeddings_client", side_effect=lambda p, **kw: clients.pop(0)):
        res = embeddings.embed_documents(["Test doc"])

    assert res == [[0.4, 0.5, 0.6]]
    # Provider 1 must have been marked in cooldown
    assert manager.providers[0].cooldown_until > 0


@pytest.mark.asyncio
async def test_rotating_embeddings_async():
    manager = RoundRobinEmbeddingManager(env_dict=MOCK_EMBEDDING_ENV)
    embeddings = RotatingEmbeddings(manager=manager)

    fake_client = MagicMock()
    fake_client.embed_documents.return_value = [[0.7, 0.8, 0.9]]
    fake_client.embed_query.return_value = [0.7, 0.8, 0.9]

    with patch.object(manager, "create_embeddings_client", return_value=fake_client):
        docs = await embeddings.aembed_documents(["Async text"])
        assert docs == [[0.7, 0.8, 0.9]]

        query = await embeddings.aembed_query("Async query")
        assert query == [0.7, 0.8, 0.9]


def test_rate_limit_detection_helper():
    class Dummy429(Exception):
        status_code = 429

    assert _is_rate_limit_error(Dummy429("error")) is True
    assert _is_rate_limit_error(RuntimeError("429 Too Many Requests")) is True
    assert _is_rate_limit_error(RuntimeError("rate limit exceeded")) is True
    assert _is_rate_limit_error(ValueError("invalid arguments")) is False

