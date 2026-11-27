"""End-to-end integration tests for Orbit MCP marketplace, OAuth CSRF, per-user isolation, and agent tool execution."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qs, urlparse

import httpx
import pytest
from fastapi.testclient import TestClient

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import mcp_client  # noqa: E402
import mcp_tool_search  # noqa: E402
import server  # noqa: E402
from agent import _execute_mcp_tool_with_auto_refresh  # noqa: E402


class _InMemoryConnectorChatStore:
    """Minimal in-memory chat_store double for testing personal connectors & marketplace settings."""

    def __init__(self) -> None:
        self._settings: Dict[str, Dict[str, str]] = {}
        self._connectors: Dict[str, Dict[str, Dict[str, Any]]] = {}
        self._seq = 0

    def get_setting(self, key: str, default: Any = None, user_id: Optional[str] = None) -> Any:
        uid = str(user_id or "global")
        return self._settings.get(uid, {}).get(key, default)

    def set_setting(self, key: str, value: str, user_id: Optional[str] = None) -> None:
        uid = str(user_id or "global")
        self._settings.setdefault(uid, {})[key] = value

    def get_settings_by_prefix(self, prefix: str, user_id: Optional[str] = None) -> Dict[str, str]:
        uid = str(user_id or "global")
        return {k: v for k, v in self._settings.get(uid, {}).items() if k.startswith(prefix)}

    def list_personal_connectors_for_user(self, user_id: str) -> List[Dict[str, Any]]:
        return list(self._connectors.get(user_id, {}).values())

    def get_personal_connectors(self, user_id: str, include_secrets: bool = False) -> List[Dict[str, Any]]:
        return self.list_personal_connectors_for_user(user_id)

    def get_personal_connector(self, user_id: str, connector_id: str, include_secrets: bool = False) -> Optional[Dict[str, Any]]:
        return self._connectors.get(user_id, {}).get(connector_id)

    def create_personal_connector(self, user_id: str, **kwargs: Any) -> Dict[str, Any]:
        self._seq += 1
        cid = f"pc-{self._seq}"
        rec = {
            "id": cid,
            "name": kwargs.get("name", ""),
            "mcp_url": kwargs.get("mcp_url", ""),
            "connector_type": kwargs.get("connector_type", "remote_mcp"),
            "auth_mode": kwargs.get("auth_mode", "oauth"),
            "headers": kwargs.get("headers") or [],
            "advanced": kwargs.get("advanced") or {},
            "tools": kwargs.get("tools") or [],
            "tool_permissions": kwargs.get("tool_permissions") or {},
            "is_connected": bool(kwargs.get("is_connected", False)),
            "last_status_code": kwargs.get("last_status_code"),
            "last_checked_at": "2026-11-26T00:00:00Z",
            "created_at": "2026-11-26T00:00:00Z",
            "updated_at": "2026-11-26T00:00:00Z",
        }
        self._connectors.setdefault(user_id, {})[cid] = rec
        return dict(rec)

    def update_personal_connector(self, user_id: str, connector_id: str, **kwargs: Any) -> Optional[Dict[str, Any]]:
        rec = self._connectors.get(user_id, {}).get(connector_id)
        if not rec:
            return None
        for k, v in kwargs.items():
            if v is not None:
                rec[k] = v
        return dict(rec)

    def delete_personal_connector(self, user_id: str, connector_id: str) -> bool:
        return self._connectors.get(user_id, {}).pop(connector_id, None) is not None


def test_oauth_state_hmac_prevents_csrf():
    state = server._generate_mcp_oauth_state("user-1", "pc-1")
    assert server._validate_mcp_oauth_state(state, "user-1", "pc-1") is True
    assert server._validate_mcp_oauth_state(state, "user-2", "pc-1") is False
    assert server._validate_mcp_oauth_state(state, "user-1", "pc-2") is False
    assert server._validate_mcp_oauth_state("mcp_tampered_nonce", "user-1", "pc-1") is False


def test_marketplace_e2e_lifecycle_and_user_isolation(monkeypatch):
    store = _InMemoryConnectorChatStore()
    monkeypatch.setattr(server, "chat_store", store)
    monkeypatch.setattr(server, "provider_integration_store", None)

    current_user_box = {"user": {"user_id": "user-a", "email": "a@orbit.test"}}
    server.app.dependency_overrides[server.require_user] = lambda: current_user_box["user"]

    # Mock remote MCP + OAuth server
    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url.endswith("/.well-known/oauth-protected-resource"):
            return httpx.Response(200, json={"resource": "https://mcp.notion.com/mcp", "authorization_servers": ["https://mcp.notion.com"]})
        if url.endswith("/.well-known/oauth-authorization-server"):
            return httpx.Response(
                200,
                json={
                    "authorization_endpoint": "https://mcp.notion.com/authorize",
                    "token_endpoint": "https://mcp.notion.com/token",
                    "registration_endpoint": "https://mcp.notion.com/register",
                    "revocation_endpoint": "https://mcp.notion.com/revoke",
                    "scopes_supported": ["default"],
                },
            )
        if url == "https://mcp.notion.com/register":
            return httpx.Response(201, json={"client_id": "dyn-client-a", "client_secret": "dyn-secret-a"})
        if url == "https://mcp.notion.com/token":
            return httpx.Response(200, json={"access_token": "tok-access-1", "refresh_token": "tok-refresh-1", "expires_in": 3600})
        if url == "https://mcp.notion.com/revoke":
            return httpx.Response(200)
        if url == "https://mcp.notion.com/mcp":
            auth = request.headers.get("Authorization", "")
            if auth != "Bearer tok-access-1":
                return httpx.Response(401, headers={"WWW-Authenticate": 'Bearer resource_metadata="https://mcp.notion.com/.well-known/oauth-protected-resource"'})
            body = json.loads(request.content.decode("utf-8"))
            method = body.get("method")
            if method == "initialize":
                return httpx.Response(200, headers={"Mcp-Session-Id": "s-notion-1"}, json={"jsonrpc": "2.0", "id": body["id"], "result": {"protocolVersion": "2025-03-26", "capabilities": {}, "serverInfo": {"name": "Notion", "version": "1.0"}}})
            if method == "notifications/initialized":
                return httpx.Response(202)
            if method == "tools/list":
                return httpx.Response(200, json={"jsonrpc": "2.0", "id": body["id"], "result": {"tools": [{"name": "search_pages", "description": "Search Notion pages", "inputSchema": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}]}})
            if method == "tools/call":
                return httpx.Response(200, json={"jsonrpc": "2.0", "id": body["id"], "result": {"content": [{"type": "text", "text": "Found 2 Notion pages"}]}})
        return httpx.Response(404)

    transport = httpx.MockTransport(handler)
    real_async_client = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: real_async_client(transport=transport, **{k: v for k, v in kw.items() if k != "transport"}))
    mcp_client._DYN_REG_CACHE.clear()

    try:
        client = TestClient(server.app)

        # 1. Catalogue lists connectors with verification_status
        res_cat = client.get("/api/marketplace")
        assert res_cat.status_code == 200
        cat_data = res_cat.json()
        assert cat_data["stats"]["total_connectors"] >= 950
        by_id = {c["id"]: c for c in cat_data["connectors"]}
        assert "microsoft-365" not in by_id

        # 3. Toggle Notion -> needs_oauth=True, creates personal connector
        res_tog = client.post("/api/marketplace/connectors/notion/toggle", json={"enabled": True})
        assert res_tog.status_code == 200
        tog_data = res_tog.json()
        assert tog_data["is_connected"] is False
        assert tog_data["needs_oauth"] is True
        pid = tog_data["personal_connector_id"]
        assert pid

        # 4. Start OAuth -> returns auth_url with PKCE + HMAC state
        res_oauth = client.post(f"/api/settings/personal-connectors/{pid}/oauth/start")
        assert res_oauth.status_code == 200
        oauth_data = res_oauth.json()
        assert oauth_data["needs_oauth"] is True
        parsed_auth = urlparse(oauth_data["auth_url"])
        qs = parse_qs(parsed_auth.query)
        state = qs["state"][0]
        assert qs["code_challenge_method"][0] == "S256"

        # 5. Complete OAuth callback -> discovers tools & marks connected
        res_cb = client.get(f"/settings/personal-connectors/oauth/callback?code=auth-code-1&state={state}")
        assert res_cb.status_code == 200
        assert "Connected to MCP Server" in res_cb.text

        # 6. Diagnostics endpoint reflects connected status + 1 discovered tool
        res_diag = client.get("/api/marketplace/connectors/notion/diagnostics")
        assert res_diag.status_code == 200
        diag = res_diag.json()
        assert diag["is_connected"] is True
        assert diag["tools_count"] == 1
        assert diag["tools"][0]["name"] == "search_pages"

        # 7. Per-user isolation: User B cannot access or disconnect User A's connector
        current_user_box["user"] = {"user_id": "user-b", "email": "b@orbit.test"}
        res_b_disc = client.post(f"/api/settings/personal-connectors/{pid}/disconnect")
        assert res_b_disc.status_code == 404

        # 8. Switch back to User A and disconnect via marketplace toggle -> signs out, revokes token & clears headers
        current_user_box["user"] = {"user_id": "user-a", "email": "a@orbit.test"}
        res_a_disc = client.post("/api/marketplace/connectors/notion/toggle", json={"enabled": False})
        assert res_a_disc.status_code == 200
        assert res_a_disc.json()["is_connected"] is False
        pc_after = store.get_personal_connector("user-a", pid)
        assert pc_after is not None
        assert pc_after["is_connected"] is False
        assert pc_after["headers"] == []
        assert pc_after["tools"] == []
        assert "oauth_tokens" not in (pc_after.get("advanced") or {})
    finally:
        server.app.dependency_overrides.pop(server.require_user, None)
        mcp_client._SHARED_MCP_HTTP_CLIENT = None


@pytest.mark.asyncio
async def test_agent_executes_mcp_tool_with_auto_refresh(monkeypatch):
    store = _InMemoryConnectorChatStore()
    calls = []

    async def fake_call_mcp(url, tool_name, args, custom_headers=None, timeout=25.0):
        calls.append((url, tool_name, dict(custom_headers or {})))
        if (custom_headers or {}).get("Authorization") == "Bearer expired-tok":
            raise RuntimeError("MCP tool call failed with HTTP 401 (Unauthorized)")
        return f"ok:{tool_name}:{args.get('q')}"

    async def fake_refresh(client, token_endpoint, refresh_token, client_id, client_secret=None):
        return {"access_token": "fresh-tok", "refresh_token": "fresh-ref"}

    monkeypatch.setattr(mcp_client, "call_mcp_tool", fake_call_mcp)
    monkeypatch.setattr(mcp_client, "refresh_mcp_oauth_token", fake_refresh)

    rec = store.create_personal_connector(
        "user-1",
        name="Notion",
        mcp_url="https://mcp.notion.com/mcp",
        headers=[{"key": "Authorization", "value": "Bearer expired-tok"}],
        advanced={
            "oauth_tokens": {
                "refresh_token": "ref-1",
                "token_endpoint": "https://mcp.notion.com/token",
                "client_id": "cid-1",
            }
        },
        is_connected=True,
    )

    out = await _execute_mcp_tool_with_auto_refresh(
        rec,
        "search_pages",
        {"q": "roadmap"},
        {"Authorization": "Bearer expired-tok"},
        target_user_id="user-1",
        chat_store=store,
    )
    assert out == "ok:search_pages:roadmap"
    assert len(calls) == 2
    assert calls[1][2]["Authorization"] == "Bearer fresh-tok"
