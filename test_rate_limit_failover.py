import asyncio
from unittest.mock import patch
import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool

from agent import ReActAgent, _normalize_messages_for_provider
from config_manager import RoundRobinProviderManager


MOCK_ENV = {
    "API_KEY1": "sk-test-mistral-12345",
    "BASE_URL1": "https://api.mistral.ai/v1",
    "MODEL1": "codestral-2508",
    "PROVIDER_NAME1": "mistral",
    "API_KEY2": "sk-test-zai-67890",
    "BASE_URL2": "https://api.z.ai/api/paas/v4",
    "MODEL2": "GLM-4.7-Flash",
    "PROVIDER_NAME2": "zai",
}


@tool
def dummy_echo(text: str) -> str:
    """Return the given text."""
    return f"ECHO:{text}"


def test_rate_limit_cooldown_skips_cooling_provider():
    manager = RoundRobinProviderManager(env_dict=MOCK_ENV)
    manager.mark_rate_limited(manager.providers[0], retry_after=5.0)
    picked = manager.get_next_provider()
    assert picked.name == "zai"


def test_create_chat_model_disables_hidden_sdk_retries():
    manager = RoundRobinProviderManager(env_dict=MOCK_ENV)
    llm = manager.create_chat_model(manager.providers[0])
    assert llm.max_retries == 0


def test_normalize_tool_call_ids_for_mistral():
    manager = RoundRobinProviderManager(env_dict=MOCK_ENV)
    mistral = manager.providers[0]
    msgs = [
        AIMessage(
            content="",
            tool_calls=[{"name": "manage_routine", "args": {"action": "list"}, "id": "call_239cf08362fe4e4088e6ee52"}],
        ),
        ToolMessage(content="ok", tool_call_id="call_239cf08362fe4e4088e6ee52", name="manage_routine"),
    ]
    out = _normalize_messages_for_provider(msgs, mistral)
    new_id = out[0].tool_calls[0]["id"]
    assert len(new_id) == 9 and new_id.isalnum()
    assert out[1].tool_call_id == new_id


@pytest.mark.asyncio
async def test_turn2_429_fails_over_cleanly_after_tool_call():
    manager = RoundRobinProviderManager(env_dict=MOCK_ENV)
    agent = ReActAgent(provider_manager=manager, tools=[dummy_echo])

    class FakeChunk:
        def __init__(self, content="", tool_calls=None):
            self.content = content
            self.tool_calls = tool_calls or []

        def __add__(self, other):
            return FakeChunk(
                self.content + other.content,
                self.tool_calls or other.tool_calls,
            )

    p1_calls = {"n": 0}
    p2_calls = {"n": 0}

    class Mistral429OnTurn2LLM:
        model_name = "codestral-2508"

        def bind_tools(self, tools):
            class Bound:
                async def astream(self_inner, messages):
                    p1_calls["n"] += 1
                    if p1_calls["n"] == 1:
                        yield FakeChunk(
                            "",
                            [{"name": "dummy_echo", "args": {"text": "hello"}, "id": "abc123456"}],
                        )
                    else:
                        raise RuntimeError("Error code: 429 - Rate limit exceeded")
            return Bound()

    class ZaiFallbackLLM:
        model_name = "GLM-4.7-Flash"

        def bind_tools(self, tools):
            class Bound:
                async def astream(self_inner, messages):
                    p2_calls["n"] += 1
                    yield FakeChunk("Echoed: hello")
            return Bound()

    seq = [
        (Mistral429OnTurn2LLM(), manager.providers[0], 1),
        (ZaiFallbackLLM(), manager.providers[1], 2),
    ]

    with patch.object(manager, "get_next_llm", side_effect=lambda **kw: seq.pop(0)):
        events = [ev async for ev in agent.astream_run(user_query="Echo hello")]

    assert p1_calls["n"] == 2
    assert p2_calls["n"] == 1
    assert manager.providers[0].cooldown_until > 0.0
    assert [e for e in events if e["type"] == "error"] == []
    done = [e for e in events if e["type"] == "done"]
    assert len(done) == 1
    assert "Echoed: hello" in done[0]["data"]["content"]
