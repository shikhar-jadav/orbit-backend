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
