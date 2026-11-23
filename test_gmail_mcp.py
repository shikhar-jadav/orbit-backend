"""Tests for the native Multi-User Gmail MCP Server co-hosted in Orbit backend."""

import pytest
from fastapi.testclient import TestClient

from server import app

client = TestClient(app)


def test_gmail_mcp_health():
    response = client.get("/api/mcp/gmail/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["service"] == "orbit-gmail-mcp"
    assert data["embedded"] is True
    assert data["multi_user"] is True


def test_gmail_mcp_initialize():
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {},
    }
    response = client.post("/api/mcp/gmail", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["jsonrpc"] == "2.0"
    assert data["id"] == 1
    assert "tools" in data["result"]["capabilities"]
    assert data["result"]["serverInfo"]["name"] == "orbit-gmail-mcp"


def test_gmail_mcp_tools_list():
    payload = {
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/list",
        "params": {},
    }
    response = client.post("/api/mcp/gmail", json=payload)
    assert response.status_code == 200
    data = response.json()
    tools = data["result"]["tools"]
    tool_names = [t["name"] for t in tools]
    assert "send_email" in tool_names
    assert "read_email" in tool_names
    assert "search_messages" in tool_names
    assert "create_draft" in tool_names
    assert "modify_email" in tool_names


def test_gmail_mcp_tools_call_unauthenticated():
    payload = {
        "jsonrpc": "2.0",
        "id": 3,
        "method": "tools/call",
        "params": {
            "name": "send_email",
            "arguments": {"to": "test@example.com", "subject": "Hi", "body": "Hello"},
        },
    }
    response = client.post("/api/mcp/gmail", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["result"]["isError"] is True
    assert "No Gmail account connected" in data["result"]["content"][0]["text"]


def test_deduplicate_message_parts():
    from server import deduplicate_message_parts

    # Case 1: Identical text repeated before and after a tool
    parts = [
        {"type": "text", "text": "I have set up an event trigger to monitor for new emails."},
        {"type": "tool", "tool": {"id": "1", "name": "manage_routine"}},
        {"type": "text", "text": "I have set up an event trigger to monitor for new emails."},
    ]
    cleaned = deduplicate_message_parts(parts)
    assert len(cleaned) == 2
    assert cleaned[0]["type"] == "tool"
    assert cleaned[1]["type"] == "text"
    assert cleaned[1]["text"] == "I have set up an event trigger to monitor for new emails."

    # Case 2: Post-tool text extends pre-tool text
    parts_subsumed = [
        {"type": "text", "text": "Checking your emails..."},
        {"type": "tool", "tool": {"id": "2", "name": "list_messages"}},
        {"type": "text", "text": "Checking your emails... You received 1 new email from Alex."},
    ]
    cleaned_subsumed = deduplicate_message_parts(parts_subsumed)
    assert len(cleaned_subsumed) == 2
    assert cleaned_subsumed[0]["type"] == "tool"
    assert cleaned_subsumed[1]["text"] == "Checking your emails... You received 1 new email from Alex."

    # Case 3: Distinct commentary and final answer should both be kept
    parts_distinct = [
        {"type": "text", "text": "Searching your inbox for invoices."},
        {"type": "tool", "tool": {"id": "3", "name": "search_messages"}},
        {"type": "text", "text": "Found invoice #1024 from Stripe."},
    ]
    cleaned_distinct = deduplicate_message_parts(parts_distinct)
    assert len(cleaned_distinct) == 3


@pytest.mark.asyncio
async def test_resolve_user_gmail_token_without_request():
    from gmail_mcp_server import resolve_user_gmail_token

    # Should safely return (None, None, []) or (None, user_id, []) without throwing
    token, user_id, scopes = await resolve_user_gmail_token(None, x_user_id="nonexistent_user")
    assert token is None
    assert user_id == "nonexistent_user"

