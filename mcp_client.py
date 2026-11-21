"""Async HTTP / Streamable-MCP & HTTP+SSE client helpers for personal user MCP connectors."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import ipaddress
import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urljoin, urlparse, urlunparse

import httpx

MCP_PROTOCOL_VERSION = "2025-03-26"
MCP_PROTOCOL_FALLBACK = "2024-11-05"
READ_ONLY_TOOL_PREFIXES = (
    "get_",
    "list_",
    "read_",
    "search_",
    "query_",
    "fetch_",
    "find_",
    "lookup_",
    "describe_",
    "inspect_",
    "explain_",
    "view_",
    "show_",
)

DESTRUCTIVE_TOOL_TOKENS = {
    "create",
    "add",
    "new",
    "insert",
    "update",
    "modify",
    "patch",
    "edit",
    "delete",
    "remove",
    "drop",
    "destroy",
    "purge",
    "clear",
    "write",
    "send",
    "post",
    "publish",
    "trigger",
    "execute",
    "run",
    "mutate",
    "set",
    "move",
    "restore",
    "cancel",
    "stop",
    "spawn",
    "duplicate",
}

READ_ONLY_TOOL_TOKENS = {
    "get",
    "list",
    "read",
    "search",
    "query",
    "fetch",
    "find",
    "lookup",
    "describe",
    "inspect",
    "explain",
    "view",
    "show",
    "check",
}


def is_mcp_tool_read_only(name: str, annotations: Optional[Dict[str, Any]] = None) -> bool:
    """Determine whether an MCP tool is read-only (safe to execute without mutation)."""
    if isinstance(annotations, dict) and "readOnlyHint" in annotations:
        return bool(annotations["readOnlyHint"])

    clean = (name or "").lower().strip()
    if clean.startswith(READ_ONLY_TOOL_PREFIXES):
        return True

    tokens = [t for t in re.split(r"[-_]", clean) if t]
    if any(t in DESTRUCTIVE_TOOL_TOKENS for t in tokens):
        return False
    return any(t in READ_ONLY_TOOL_TOKENS for t in tokens)

UNSUPPORTED_SCHEMA_KEYS = {
    "$schema",
    "$id",
    "$comment",
    "$anchor",
    "$dynamicRef",
    "$dynamicAnchor",
    "examples",
    "default",
    "readOnly",
    "writeOnly",
    "deprecated",
}


def validate_mcp_url(raw_url: str) -> str:
    """Validate and normalize a user-supplied HTTP(S) MCP server URL."""
    trimmed = (raw_url or "").strip()
    if not trimmed:
        raise ValueError("MCP server URL is required.")
    parsed = urlparse(trimmed)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc or not parsed.hostname:
        raise ValueError("Enter a valid http:// or https:// MCP server URL.")

    hostname = parsed.hostname.lower()
    if hostname in {"metadata.google.internal", "metadata.goog"}:
        raise ValueError("Access to cloud metadata hosts is blocked.")
    try:
        ip = ipaddress.ip_address(hostname)
        if ip.is_link_local or ip.is_multicast:
            raise ValueError("Link-local and multicast addresses are not allowed.")
    except ValueError as err:
        if "not allowed" in str(err) or "blocked" in str(err):
            raise

    return trimmed


def _base_mcp_headers(custom_headers: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    headers: Dict[str, str] = {
        "Accept": "application/json, text/event-stream",
        "Content-Type": "application/json",
        "User-Agent": "OrbitAgentStudio-MCP/1.0",
    }
    if custom_headers:
        for k, v in custom_headers.items():
            key = str(k).strip()
            if key and key.lower() not in {"host", "content-length"}:
                headers[key] = str(v)
    return headers


def parse_mcp_text(text: str, content_type: str = "") -> Optional[Dict[str, Any]]:
    """Extract a JSON-RPC payload from buffered JSON or SSE response text."""
    stripped_body = (text or "").strip()
    if not stripped_body:
        return None
    ctype = (content_type or "").lower()
    if "application/json" in ctype or stripped_body.startswith("{"):
        try:
            parsed = json.loads(stripped_body)
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            pass

    last_obj: Optional[Dict[str, Any]] = None
    for line in stripped_body.splitlines():
        stripped = line.strip()
        if not stripped.startswith("data:"):
            continue
        raw_data = stripped[5:].strip()
        if not raw_data or raw_data == "[DONE]":
            continue
        try:
            obj = json.loads(raw_data)
            if isinstance(obj, dict):
                if "result" in obj or "error" in obj:
                    return obj
                last_obj = obj
        except json.JSONDecodeError:
            continue
    return last_obj


def parse_mcp_payload(response: httpx.Response) -> Optional[Dict[str, Any]]:
    """Extract a JSON-RPC payload from an httpx.Response."""
    return parse_mcp_text(response.text, response.headers.get("content-type", ""))


def sanitize_tool_schema(schema: Any) -> Dict[str, Any]:
    """Normalize an MCP JSON Schema for OpenAI/Groq/Gemini-compatible tool binding."""

    def _clean(node: Any) -> Any:
        if not isinstance(node, dict):
            return node
        cleaned: Dict[str, Any] = {}
        for k, v in node.items():
            if k in UNSUPPORTED_SCHEMA_KEYS:
                continue
            if k == "properties" and isinstance(v, dict):
                cleaned["properties"] = {str(pk): _clean(pv) for pk, pv in v.items()}
            elif k == "items":
                if isinstance(v, dict):
                    cleaned["items"] = _clean(v)
                elif isinstance(v, list):
                    cleaned["items"] = [_clean(item) for item in v]
            elif k in {"anyOf", "oneOf", "allOf"} and isinstance(v, list):
                cleaned[k] = [_clean(item) for item in v]
            elif k == "required" and isinstance(v, list):
                cleaned["required"] = [str(r) for r in v if r is not None]
            elif k == "enum" and isinstance(v, list) and len(v) > 0:
                cleaned["enum"] = v
            else:
                cleaned[k] = v

        if cleaned.get("type") == "object" and "properties" not in cleaned:
            cleaned["properties"] = {}
        return cleaned

    normalized = _clean(schema) if isinstance(schema, dict) else {}
    if not isinstance(normalized, dict):
        normalized = {}
    if normalized.get("type") != "object":
        normalized["type"] = "object"
    if not isinstance(normalized.get("properties"), dict):
        normalized["properties"] = {}
    return normalized


def normalize_mcp_tool(raw_tool: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize an MCP tools/list tool definition cleanly for UI & runtime."""
    name = str(raw_tool.get("name", "")).strip()
    description = str(raw_tool.get("description", "") or "").strip()
    input_schema = raw_tool.get("inputSchema")
    if not isinstance(input_schema, dict):
        input_schema = {"type": "object", "properties": {}}
    annotations = raw_tool.get("annotations")
    read_only = is_mcp_tool_read_only(name, annotations if isinstance(annotations, dict) else None)
    return {
        "name": name,
        "description": description,
        "inputSchema": input_schema,
        "readOnlyHint": read_only,
    }


def _protected_resource_url(mcp_url: str) -> str:
    parsed = urlparse(mcp_url)
    return urlunparse((parsed.scheme, parsed.netloc, "/.well-known/oauth-protected-resource", "", "", ""))


def _parse_www_authenticate(header: str) -> Dict[str, str]:
    """Parse WWW-Authenticate header parameters according to RFC 9728 / RFC 6750."""
    out: Dict[str, str] = {}
    if not header:
        return out
    parts = header.strip().split(None, 1)
    if parts:
        out["scheme"] = parts[0]
        if len(parts) > 1:
            matches = re.findall(r'([a-zA-Z0-9_-]+)\s*=\s*(?:"([^"]*)"|([^,\s]+))', parts[1])
            for k, v1, v2 in matches:
                out[k.lower()] = v1 if v1 != "" else v2
    return out


_DYN_REG_CACHE: Dict[Tuple[str, str], Tuple[str, Optional[str]]] = {}


def generate_pkce_pair() -> Tuple[str, str]:
    """Generate PKCE verifier and S256 challenge pair."""
    verifier = base64.urlsafe_b64encode(os.urandom(32)).decode("utf-8").rstrip("=")
    challenge = base64.urlsafe_b64encode(
        hashlib.sha256(verifier.encode("utf-8")).digest()
    ).decode("utf-8").rstrip("=")
    return verifier, challenge


async def discover_mcp_oauth_metadata(
    client: httpx.AsyncClient,
    mcp_url: str,
    www_auth_header: str = "",
) -> Optional[Dict[str, Any]]:
    """Discover OAuth 2.0 / 2.1 authorization endpoints for an MCP server (RFC 9728 & RFC 8414)."""
    parsed_auth = _parse_www_authenticate(www_auth_header)
    meta_url = parsed_auth.get("resource_metadata")
    parsed_url = urlparse(mcp_url)
    origin = f"{parsed_url.scheme}://{parsed_url.netloc}"

    if not meta_url:
        meta_url = f"{origin}/.well-known/oauth-protected-resource"

    resource_meta: Optional[Dict[str, Any]] = None
    try:
        r_meta = await client.get(meta_url, timeout=5.0)
        if r_meta.status_code == 200:
            resource_meta = r_meta.json()
    except Exception:
        pass

    auth_servers: List[str] = []
    resource_name = ""
    scopes: List[str] = []
    if resource_meta and isinstance(resource_meta, dict):
        auth_servers = [str(s) for s in resource_meta.get("authorization_servers", []) if s]
        resource_name = str(resource_meta.get("resource_name") or "")
        scopes = [str(sc) for sc in resource_meta.get("scopes_supported", []) if sc]

    if not auth_servers:
        auth_servers = [origin]

    for auth_server in auth_servers[:2]:
        base_as = auth_server.rstrip("/")
        for as_path in ["/.well-known/oauth-authorization-server", "/.well-known/openid-configuration"]:
            try:
                r_as = await client.get(f"{base_as}{as_path}", timeout=5.0)
                if r_as.status_code == 200:
                    as_meta = r_as.json()
                    if isinstance(as_meta, dict) and as_meta.get("authorization_endpoint"):
                        if not resource_name:
                            resource_name = as_meta.get("client_name") or (
                                "Notion MCP" if "notion" in mcp_url.lower() else "MCP Service"
                            )
                        return {
                            "requires_oauth": True,
                            "resource": (resource_meta.get("resource") if resource_meta else mcp_url) or mcp_url,
                            "resource_name": resource_name,
                            "authorization_server": auth_server,
                            "authorization_endpoint": as_meta["authorization_endpoint"],
                            "token_endpoint": as_meta.get("token_endpoint") or f"{base_as}/token",
                            "registration_endpoint": as_meta.get("registration_endpoint"),
                            "scopes": scopes or as_meta.get("scopes_supported") or ["default"],
                            "code_challenge_methods_supported": as_meta.get(
                                "code_challenge_methods_supported", ["S256"]
                            ),
                        }
            except Exception:
                continue

    if "notion.com" in mcp_url.lower():
        return {
            "requires_oauth": True,
            "resource": mcp_url,
            "resource_name": "Notion MCP (Beta)",
            "authorization_server": "https://mcp.notion.com",
            "authorization_endpoint": "https://mcp.notion.com/authorize",
            "token_endpoint": "https://mcp.notion.com/token",
            "registration_endpoint": "https://mcp.notion.com/register",
            "scopes": ["default"],
            "code_challenge_methods_supported": ["S256"],
        }

    return None


async def register_mcp_oauth_client(
    client: httpx.AsyncClient,
    registration_endpoint: str,
    redirect_uri: str,
    client_name: str = "Orbit Agent",
) -> Tuple[str, Optional[str]]:
    """Dynamically register an OAuth client (RFC 7591) for an MCP authorization server."""
    cache_key = (registration_endpoint, redirect_uri)
    if cache_key in _DYN_REG_CACHE:
        return _DYN_REG_CACHE[cache_key]

    body = {
        "client_name": client_name,
        "redirect_uris": [redirect_uri],
        "grant_types": ["authorization_code", "refresh_token"],
        "response_types": ["code"],
    }
    r = await client.post(registration_endpoint, json=body, timeout=8.0)
    if r.status_code not in (200, 201):
        raise RuntimeError(f"Client registration failed with HTTP {r.status_code}: {r.text[:200]}")
    data = r.json()
    client_id = data.get("client_id")
    client_secret = data.get("client_secret")
    if not client_id:
        raise RuntimeError("No client_id returned in registration response.")
    _DYN_REG_CACHE[cache_key] = (str(client_id), str(client_secret) if client_secret else None)
    return _DYN_REG_CACHE[cache_key]


async def exchange_mcp_oauth_code(
    client: httpx.AsyncClient,
    token_endpoint: str,
    code: str,
    redirect_uri: str,
    code_verifier: str,
    client_id: str,
    client_secret: Optional[str] = None,
) -> Dict[str, Any]:
    """Exchange OAuth authorization code for Bearer access token (RFC 6749 & RFC 7636)."""
    base_data = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": redirect_uri,
        "code_verifier": code_verifier,
    }
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/x-www-form-urlencoded",
    }

    # RFC 6749 Section 2.3.1: client_secret_basic sends HTTP Basic auth header
    # and MUST NOT repeat client_id or client_secret in the form body.
    if client_secret:
        auth = (client_id, client_secret)
        r = await client.post(
            token_endpoint, data=base_data, headers=headers, auth=auth, timeout=14.0
        )
    else:
        # Public client (PKCE only): client_id in form body
        public_data = {**base_data, "client_id": client_id}
        r = await client.post(
            token_endpoint, data=public_data, headers=headers, timeout=14.0
        )

    # Fallback to client_secret_post (credentials in form body without Basic auth)
    if r.status_code not in (200, 201) and client_secret:
        post_data = {**base_data, "client_id": client_id, "client_secret": client_secret}
        r_post = await client.post(
            token_endpoint, data=post_data, headers=headers, timeout=14.0
        )
        if r_post.status_code in (200, 201):
            r = r_post

    if r.status_code not in (200, 201):
        raise RuntimeError(f"Token exchange failed with HTTP {r.status_code}: {r.text[:300]}")
    payload = r.json()
    if not isinstance(payload, dict) or "access_token" not in payload:
        raise RuntimeError("MCP token response missing access_token.")
    return payload


async def refresh_mcp_oauth_token(
    client: httpx.AsyncClient,
    token_endpoint: str,
    refresh_token: str,
    client_id: str,
    client_secret: Optional[str] = None,
) -> Dict[str, Any]:
    """Refresh an expired OAuth access token using a refresh token (RFC 6749 Section 6)."""
    base_data = {
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
    }
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/x-www-form-urlencoded",
    }

    if client_secret:
        auth = (client_id, client_secret)
        r = await client.post(
            token_endpoint, data=base_data, headers=headers, auth=auth, timeout=14.0
        )
    else:
        public_data = {**base_data, "client_id": client_id}
        r = await client.post(
            token_endpoint, data=public_data, headers=headers, timeout=14.0
        )

    if r.status_code not in (200, 201) and client_secret:
        post_data = {**base_data, "client_id": client_id, "client_secret": client_secret}
        r_post = await client.post(
            token_endpoint, data=post_data, headers=headers, timeout=14.0
        )
        if r_post.status_code in (200, 201):
            r = r_post

    if r.status_code not in (200, 201):
        raise RuntimeError(f"Token refresh failed with HTTP {r.status_code}: {r.text[:300]}")
    payload = r.json()
    if not isinstance(payload, dict) or "access_token" not in payload:
        raise RuntimeError("MCP token refresh response missing access_token.")
    return payload


def _extract_rpc_error(payload: Optional[Dict[str, Any]]) -> Optional[str]:
    if not payload or "error" not in payload or not payload["error"]:
        return None
    err = payload["error"]
    if isinstance(err, dict):
        return str(err.get("message") or err.get("code") or "MCP server returned JSON-RPC error.")
    return str(err)


async def _post_mcp_jsonrpc(
    client: httpx.AsyncClient,
    mcp_url: str,
    headers: Dict[str, str],
    body: Dict[str, Any],
    *,
    rpc_id: Optional[Any] = None,
) -> Tuple[int, str, httpx.Headers, Optional[Dict[str, Any]]]:
    """POST JSON-RPC via Streamable HTTP without hanging on open text/event-stream responses."""
    async with client.stream("POST", mcp_url, headers=headers, json=body) as resp:
        status = resp.status_code
        reason = resp.reason_phrase or f"HTTP {status}"
        ctype = (resp.headers.get("content-type") or "").lower()

        if status >= 400 or "text/event-stream" not in ctype:
            raw_bytes = await resp.aread()
            text = raw_bytes.decode("utf-8", errors="replace")
            return status, reason, resp.headers, parse_mcp_text(text, ctype)

        last_obj: Optional[Dict[str, Any]] = None
        async for line in resp.aiter_lines():
            stripped = line.strip()
            if not stripped.startswith("data:"):
                continue
            raw_data = stripped[5:].strip()
            if not raw_data or raw_data == "[DONE]":
                continue
            try:
                obj = json.loads(raw_data)
            except json.JSONDecodeError:
                continue
            if not isinstance(obj, dict):
                continue
            last_obj = obj
            if rpc_id is not None and obj.get("id") != rpc_id:
                continue
            if "result" in obj or "error" in obj:
                return status, reason, resp.headers, obj
        return status, reason, resp.headers, last_obj


async def _probe_legacy_sse_endpoint(
    client: httpx.AsyncClient,
    url: str,
    headers: Dict[str, str],
) -> Tuple[int, str, httpx.Headers, Optional[str]]:
    """Safely inspect an SSE endpoint without blocking forever on open event streams."""
    sse_headers = dict(headers)
    sse_headers["Accept"] = "text/event-stream"
    async with client.stream("GET", url, headers=sse_headers) as resp:
        status = resp.status_code
        reason = resp.reason_phrase or f"HTTP {status}"
        ctype = (resp.headers.get("content-type") or "").lower()
        if status >= 400 or "text/event-stream" not in ctype:
            await resp.aread()
            return status, reason, resp.headers, None

        current_event = ""
        try:
            async with asyncio.timeout(3.0):
                async for line in resp.aiter_lines():
                    stripped = line.strip()
                    if not stripped:
                        current_event = ""
                        continue
                    if stripped.startswith("event:"):
                        current_event = stripped[6:].strip()
                    elif stripped.startswith("data:") and current_event == "endpoint":
                        rel = stripped[5:].strip()
                        if rel:
                            return status, reason, resp.headers, urljoin(url, rel)
        except TimeoutError:
            pass
        return status, reason, resp.headers, None


async def _init_streamable_session(
    client: httpx.AsyncClient,
    mcp_url: str,
    headers: Dict[str, str],
) -> Tuple[int, str, httpx.Headers, Optional[Dict[str, Any]], Optional[str]]:
    init_body = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": MCP_PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": {"name": "orbit-agent-studio", "version": "1.0.0"},
        },
    }
    status, reason, resp_headers, payload = await _post_mcp_jsonrpc(
        client, mcp_url, headers, init_body, rpc_id=1
    )
    session_id = resp_headers.get("mcp-session-id") or resp_headers.get("Mcp-Session-Id")
    return status, reason, resp_headers, payload, session_id


async def _fetch_all_tools_streamable(
    client: httpx.AsyncClient,
    mcp_url: str,
    rpc_headers: Dict[str, str],
) -> Tuple[int, List[Dict[str, Any]], Optional[str]]:
    tools: List[Dict[str, Any]] = []
    cursor: Optional[str] = None
    last_status = 200

    for page in range(5):
        params: Dict[str, Any] = {"cursor": cursor} if cursor else {}
        status, reason, _, payload = await _post_mcp_jsonrpc(
            client,
            mcp_url,
            rpc_headers,
            {"jsonrpc": "2.0", "id": 10 + page, "method": "tools/list", "params": params},
            rpc_id=10 + page,
        )
        last_status = status
        if status >= 400:
            return status, tools, f"tools/list failed with HTTP {status} ({reason})."
        rpc_err = _extract_rpc_error(payload)
        if rpc_err:
            return status, tools, rpc_err
        result = payload.get("result") if isinstance(payload, dict) else None
        if not isinstance(result, dict):
            break
        raw_tools = result.get("tools", [])
        if isinstance(raw_tools, list):
            for item in raw_tools:
                if isinstance(item, dict) and item.get("name"):
                    tools.append(normalize_mcp_tool(item))
        next_cursor = result.get("nextCursor")
        if not next_cursor:
            break
        cursor = str(next_cursor)

    return last_status, tools, None


async def _legacy_sse_exchange(
    client: httpx.AsyncClient,
    sse_url: str,
    headers: Dict[str, str],
    rpc_calls: List[Tuple[int, str, Dict[str, Any]]],
) -> Tuple[int, Dict[int, Dict[str, Any]], Optional[str]]:
    """Run initialize + subsequent RPCs against an HTTP+SSE (2024-11-05) MCP server."""
    sse_headers = dict(headers)
    sse_headers["Accept"] = "text/event-stream"
    results: Dict[int, Dict[str, Any]] = {}

    async with client.stream("GET", sse_url, headers=sse_headers) as sse_resp:
        status = sse_resp.status_code
        reason = sse_resp.reason_phrase or f"HTTP {status}"
        if status >= 400:
            await sse_resp.aread()
            return status, results, f"SSE endpoint returned HTTP {status} ({reason})."

        post_url: Optional[str] = None
        current_event = ""
        line_iter = sse_resp.aiter_lines().__aiter__()

        async def _next_event_message() -> Optional[Dict[str, Any]]:
            nonlocal current_event, post_url
            async for line in line_iter:
                stripped = line.strip()
                if not stripped:
                    current_event = ""
                    continue
                if stripped.startswith("event:"):
                    current_event = stripped[6:].strip()
                elif stripped.startswith("data:"):
                    raw = stripped[5:].strip()
                    if current_event == "endpoint" and raw and not post_url:
                        post_url = urljoin(sse_url, raw)
                        return {"__endpoint__": post_url}
                    try:
                        obj = json.loads(raw)
                        if isinstance(obj, dict):
                            return obj
                    except json.JSONDecodeError:
                        continue
            return None

        async with asyncio.timeout(10.0):
            while post_url is None:
                msg = await _next_event_message()
                if msg is None:
                    return 502, results, "SSE stream closed before advertising endpoint."

        assert post_url is not None

        async def _send_and_wait(rpc_id: int, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
            body = {"jsonrpc": "2.0", "id": rpc_id, "method": method, "params": params}
            p_status, p_reason, _, p_payload = await _post_mcp_jsonrpc(
                client, post_url, headers, body, rpc_id=rpc_id
            )
            if p_status >= 400:
                raise RuntimeError(f"{method} failed with HTTP {p_status} ({p_reason}).")
            if p_payload and ("result" in p_payload or "error" in p_payload):
                return p_payload
            async with asyncio.timeout(10.0):
                while True:
                    msg = await _next_event_message()
                    if msg is None:
                        raise RuntimeError("SSE stream closed before response arrived.")
                    if msg.get("id") == rpc_id and ("result" in msg or "error" in msg):
                        return msg

        init_payload = await _send_and_wait(
            1,
            "initialize",
            {
                "protocolVersion": MCP_PROTOCOL_FALLBACK,
                "capabilities": {},
                "clientInfo": {"name": "orbit-agent-studio", "version": "1.0.0"},
            },
        )
        results[1] = init_payload
        rpc_err = _extract_rpc_error(init_payload)
        if rpc_err:
            return status, results, rpc_err

        try:
            await client.post(
                post_url,
                headers=headers,
                json={"jsonrpc": "2.0", "method": "notifications/initialized"},
            )
        except Exception:
            pass

        for rpc_id, method, params in rpc_calls:
            payload = await _send_and_wait(rpc_id, method, params)
            results[rpc_id] = payload
            call_err = _extract_rpc_error(payload)
            if call_err:
                return status, results, call_err

        return 200, results, None


async def probe_mcp_server(
    mcp_url: str,
    custom_headers: Optional[Dict[str, str]] = None,
    timeout: float = 10.0,
) -> Dict[str, Any]:
    """Probe an MCP endpoint (Streamable HTTP + HTTP+SSE fallback) and discover tools/auth."""
    url = validate_mcp_url(mcp_url)
    headers = _base_mcp_headers(custom_headers)

    result: Dict[str, Any] = {
        "url": url,
        "reachable": False,
        "status_code": None,
        "connect_status": "error",
        "connect_detail": "Unreachable",
        "signin_settings_status": "Skipped",
        "signin_provider_status": "Skipped",
        "detected_auth": "none",
        "can_continue_manually": True,
        "server_info": None,
        "tools": [],
        "oauth_metadata": None,
        "error": None,
    }

    try:
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            status, reason, resp_headers, payload, session_id = await _init_streamable_session(
                client, url, headers
            )

            legacy_endpoint: Optional[str] = None
            if status == 405:
                status, reason, resp_headers, legacy_endpoint = await _probe_legacy_sse_endpoint(
                    client, url, headers
                )

            result["status_code"] = status

            if status in (401, 403):
                result["reachable"] = True
                result["connect_status"] = "done"
                result["connect_detail"] = "Done"
                www_auth = resp_headers.get("www-authenticate", "")
                result["detected_auth"] = "now"
                result["signin_settings_status"] = "Detected"

                oauth_meta = await discover_mcp_oauth_metadata(client, url, www_auth)
                if oauth_meta:
                    result["oauth_metadata"] = oauth_meta
                    res_name = oauth_meta.get("resource_name") or (
                        "Notion" if "notion" in url.lower() else ""
                    )
                    result["signin_provider_status"] = (
                        f"OAuth bearer ({res_name})" if res_name else "OAuth bearer"
                    )
                else:
                    result["signin_provider_status"] = (
                        "OAuth bearer" if "bearer" in www_auth.lower() else "Auth required"
                    )
                return result

            if 200 <= status < 300:
                result["reachable"] = True
                result["connect_status"] = "done"
                result["connect_detail"] = "Done"
                result["detected_auth"] = "none"
                result["signin_settings_status"] = "Skipped"
                result["signin_provider_status"] = "Skipped"

                if legacy_endpoint:
                    try:
                        _, sse_res, _ = await _legacy_sse_exchange(
                            client, url, headers, [(2, "tools/list", {})]
                        )
                        init_r = sse_res.get(1, {}).get("result")
                        if isinstance(init_r, dict):
                            result["server_info"] = init_r.get("serverInfo")
                        tools_r = sse_res.get(2, {}).get("result")
                        if isinstance(tools_r, dict) and isinstance(tools_r.get("tools"), list):
                            result["tools"] = [
                                normalize_mcp_tool(t)
                                for t in tools_r["tools"]
                                if isinstance(t, dict) and t.get("name")
                            ]
                    except Exception:
                        pass
                elif payload and isinstance(payload.get("result"), dict):
                    result["server_info"] = payload["result"].get("serverInfo")
                    try:
                        rpc_headers = dict(headers)
                        if session_id:
                            rpc_headers["Mcp-Session-Id"] = session_id
                        await client.post(
                            url,
                            headers=rpc_headers,
                            json={"jsonrpc": "2.0", "method": "notifications/initialized"},
                        )
                        _, tools, _ = await _fetch_all_tools_streamable(client, url, rpc_headers)
                        result["tools"] = tools
                    except Exception:
                        pass

                try:
                    oauth_resp = await client.get(_protected_resource_url(url), timeout=3.0)
                    if oauth_resp.status_code == 200 and "authorization_servers" in oauth_resp.text:
                        result["signin_settings_status"] = "Available"
                        result["signin_provider_status"] = "OAuth optional"
                        result["detected_auth"] = "later"
                except Exception:
                    pass

                return result

            result["connect_detail"] = reason
            result["error"] = f"Server returned HTTP {status} ({reason})."
            return result

    except httpx.TimeoutException:
        result["connect_detail"] = "Timed out"
        result["error"] = "Timed out while connecting to the MCP server."
        return result
    except httpx.RequestError as err:
        result["connect_detail"] = "Connection failed"
        result["error"] = f"Could not connect: {err}"
        return result


async def connect_and_list_mcp_tools(
    mcp_url: str,
    custom_headers: Optional[Dict[str, str]] = None,
    timeout: float = 12.0,
) -> Dict[str, Any]:
    """Connect to a personal MCP server (Streamable HTTP or HTTP+SSE) and list tools."""
    url = validate_mcp_url(mcp_url)
    headers = _base_mcp_headers(custom_headers)

    try:
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            status, reason, _, init_payload, session_id = await _init_streamable_session(
                client, url, headers
            )

            if status == 405:
                sse_status, sse_res, sse_err = await _legacy_sse_exchange(
                    client, url, headers, [(2, "tools/list", {})]
                )
                if sse_status in (401, 403):
                    return {
                        "ok": False,
                        "status_code": sse_status,
                        "tools": [],
                        "error": "Authentication rejected by MCP server. Configure credentials or request headers.",
                    }
                if sse_err:
                    return {"ok": False, "status_code": sse_status, "tools": [], "error": sse_err}
                raw_tools = (
                    sse_res.get(2, {}).get("result", {}).get("tools", [])
                    if isinstance(sse_res.get(2, {}).get("result"), dict)
                    else []
                )
                tools = [
                    normalize_mcp_tool(t)
                    for t in raw_tools
                    if isinstance(t, dict) and t.get("name")
                ]
                return {"ok": True, "status_code": sse_status, "tools": tools, "error": None}

            if status in (401, 403):
                return {
                    "ok": False,
                    "status_code": status,
                    "tools": [],
                    "error": "Authentication rejected by MCP server. Configure credentials or request headers.",
                }
            if status >= 400:
                return {
                    "ok": False,
                    "status_code": status,
                    "tools": [],
                    "error": f"Unable to connect ({status} {reason}).",
                }

            init_err = _extract_rpc_error(init_payload)
            if init_err:
                return {"ok": False, "status_code": status, "tools": [], "error": init_err}

            rpc_headers = dict(headers)
            if session_id:
                rpc_headers["Mcp-Session-Id"] = session_id

            if init_payload and isinstance(init_payload.get("result"), dict):
                try:
                    await client.post(
                        url,
                        headers=rpc_headers,
                        json={"jsonrpc": "2.0", "method": "notifications/initialized"},
                    )
                except Exception:
                    pass

            tools_status, tools, tools_err = await _fetch_all_tools_streamable(
                client, url, rpc_headers
            )
            if tools_err:
                return {
                    "ok": False,
                    "status_code": tools_status,
                    "tools": [],
                    "error": tools_err,
                }

            return {
                "ok": True,
                "status_code": tools_status if tools_status < 400 else status,
                "tools": tools,
                "error": None,
            }
    except Exception as err:
        return {
            "ok": False,
            "status_code": None,
            "tools": [],
            "error": str(err),
        }


def _format_mcp_call_payload(payload: Optional[Dict[str, Any]], raw_fallback: str = "") -> str:
    if not payload:
        return raw_fallback.strip()
    rpc_err = _extract_rpc_error(payload)
    if rpc_err:
        return f"MCP tool error: {rpc_err}"

    result = payload.get("result")
    if isinstance(result, dict):
        content_list = result.get("content")
        if isinstance(content_list, list):
            parts: List[str] = []
            for item in content_list:
                if isinstance(item, dict) and item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
                elif isinstance(item, dict):
                    parts.append(json.dumps(item, ensure_ascii=False))
                else:
                    parts.append(str(item))
            text_out = "\n".join(p for p in parts if p).strip()
            if result.get("isError"):
                return f"MCP tool error: {text_out or 'Tool reported an error.'}"
            if text_out:
                return text_out
        if result.get("structuredContent") is not None:
            return json.dumps(result["structuredContent"], ensure_ascii=False)
        return json.dumps(result, ensure_ascii=False)
    return json.dumps(payload, ensure_ascii=False)


async def call_mcp_tool(
    mcp_url: str,
    tool_name: str,
    arguments: Dict[str, Any],
    custom_headers: Optional[Dict[str, str]] = None,
    timeout: float = 25.0,
) -> str:
    """Invoke a remote MCP tool via Streamable HTTP JSON-RPC (with HTTP+SSE fallback)."""
    url = validate_mcp_url(mcp_url)
    headers = _base_mcp_headers(custom_headers)

    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
        status, reason, _, init_payload, session_id = await _init_streamable_session(
            client, url, headers
        )

        if status == 405:
            _, sse_res, sse_err = await _legacy_sse_exchange(
                client,
                url,
                headers,
                [(3, "tools/call", {"name": tool_name, "arguments": arguments})],
            )
            if sse_err:
                return f"MCP tool error: {sse_err}"
            return _format_mcp_call_payload(sse_res.get(3))

        if status >= 400:
            raise RuntimeError(f"MCP initialize failed with HTTP {status} ({reason})")
        init_err = _extract_rpc_error(init_payload)
        if init_err:
            raise RuntimeError(f"MCP initialize error: {init_err}")

        rpc_headers = dict(headers)
        if session_id:
            rpc_headers["Mcp-Session-Id"] = session_id

        try:
            await client.post(
                url,
                headers=rpc_headers,
                json={"jsonrpc": "2.0", "method": "notifications/initialized"},
            )
        except Exception:
            pass

        c_status, c_reason, _, call_payload = await _post_mcp_jsonrpc(
            client,
            url,
            rpc_headers,
            {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": {"name": tool_name, "arguments": arguments},
            },
            rpc_id=3,
        )
        if c_status >= 400:
            raise RuntimeError(f"MCP tool call failed with HTTP {c_status} ({c_reason})")
        return _format_mcp_call_payload(call_payload)
