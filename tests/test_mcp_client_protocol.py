"""Protocol, SSRF, OAuth discovery, and session tests for mcp_client.py."""

from __future__ import annotations

import base64
import hashlib
import json
import sys
from pathlib import Path

import httpx
import pytest

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import mcp_client  # noqa: E402


@pytest.mark.parametrize(
    "url",
    [
        "http://10.0.0.1/mcp",
        "http://172.16.0.1/mcp",
        "http://192.168.1.1/mcp",
        "http://127.0.0.1/mcp",
        "http://localhost:8000/mcp",
        "http://169.254.169.254/latest/meta-data",
        "http://100.64.0.1/mcp",
        "http://metadata.google.internal/mcp",
        "http://metadata.goog/mcp",
        "file:///etc/passwd",
        "ftp://example.com/mcp",
        "ws://example.com/mcp",
        "",
    ],
)
def test_ssrf_blocks_unsafe_urls(url):
    with pytest.raises(ValueError):
        mcp_client.validate_mcp_url(url)


def test_require_https_blocks_plain_http():
    with pytest.raises(ValueError, match="https://"):
        mcp_client.validate_mcp_url("http://mcp.example.com/mcp", require_https=True)
    assert mcp_client.validate_mcp_url("https://mcp.example.com/mcp", require_https=True) == "https://mcp.example.com/mcp"


def test_domains_containing_ssrf_keywords_are_allowed():
    for url in (
        "https://getunblocked.com/api/mcpsse",
        "https://my-private-cloud.example.com/mcp",
        "https://reserved-seat.example.com/mcp",
    ):
        assert mcp_client.validate_mcp_url(url) == url


def test_pkce_pair_valid_s256():
    verifier, challenge = mcp_client.generate_pkce_pair()
    expected = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode("utf-8")).digest()).decode("utf-8").rstrip("=")
    assert challenge == expected
    assert len(verifier) >= 43


@pytest.mark.asyncio
async def test_streamable_http_init_pagination_and_tool_call(monkeypatch):
    seen_methods = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            body = json.loads(request.content.decode("utf-8"))
            method = body.get("method")
            seen_methods.append(method)
            if method == "initialize":
                return httpx.Response(
                    200,
                    headers={"Mcp-Session-Id": "sess-123"},
                    json={
                        "jsonrpc": "2.0",
                        "id": body["id"],
                        "result": {
                            "protocolVersion": "2025-03-26",
                            "capabilities": {"tools": {}},
                            "serverInfo": {"name": "mock-mcp", "version": "1.0"},
                        },
                    },
                )
            if method == "notifications/initialized":
                return httpx.Response(202)
            if method == "tools/list":
                cursor = (body.get("params") or {}).get("cursor")
                if not cursor:
                    return httpx.Response(
                        200,
                        json={
                            "jsonrpc": "2.0",
                            "id": body["id"],
                            "result": {
                                "tools": [{"name": "get_item", "description": "Get item", "inputSchema": {"type": "object", "properties": {"id": {"type": "string"}}, "required": ["id"]}}],
                                "nextCursor": "page-2",
                            },
                        },
                    )
                return httpx.Response(
                    200,
                    json={
                        "jsonrpc": "2.0",
                        "id": body["id"],
                        "result": {
                            "tools": [{"name": "create_item", "description": "Create item", "inputSchema": {"type": "object", "properties": {}}}],
                        },
                    },
                )
            if method == "tools/call":
                assert request.headers.get("Mcp-Session-Id") == "sess-123"
                args = body["params"]["arguments"]
                return httpx.Response(
                    200,
                    json={
                        "jsonrpc": "2.0",
                        "id": body["id"],
                        "result": {"content": [{"type": "text", "text": f"fetched:{args['id']}"}]},
                    },
                )
        return httpx.Response(404)

    transport = httpx.MockTransport(handler)
    real_async_client = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: real_async_client(transport=transport, **{k: v for k, v in kw.items() if k != "transport"}))

    try:
        res = await mcp_client.connect_and_list_mcp_tools("https://mock.mcp.test/mcp")
        assert res["ok"] is True
        assert len(res["tools"]) == 2
        assert res["tools"][0]["name"] == "get_item"
        assert res["tools"][0]["readOnlyHint"] is True
        assert res["tools"][1]["name"] == "create_item"
        assert res["tools"][1]["readOnlyHint"] is False

        out = await mcp_client.call_mcp_tool("https://mock.mcp.test/mcp", "get_item", {"id": "42"})
        assert out == "fetched:42"
        assert "notifications/initialized" in seen_methods
    finally:
        await mcp_client.close_shared_mcp_http_client()


@pytest.mark.asyncio
async def test_per_user_dynamic_registration_isolation():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(201, json={"client_id": f"cid-{len(calls)}", "client_secret": f"sec-{len(calls)}"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        mcp_client._DYN_REG_CACHE.clear()
        c1 = await mcp_client.register_mcp_oauth_client(client, "https://auth.test/register", "https://orbit.test/cb", user_id="user-a")
        c2 = await mcp_client.register_mcp_oauth_client(client, "https://auth.test/register", "https://orbit.test/cb", user_id="user-b")
        c1_again = await mcp_client.register_mcp_oauth_client(client, "https://auth.test/register", "https://orbit.test/cb", user_id="user-a")

    assert c1 != c2, "Different users must not share dynamically registered OAuth clients"
    assert c1 == c1_again
    assert len(calls) == 2
