"""Async HTTP / Streamable-MCP & HTTP+SSE client helpers for personal user MCP connectors."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import ipaddress
import json
import os
import re
import time
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urljoin, urlparse, urlunparse

import httpx


async def close_shared_mcp_http_client() -> None:
    """No-op shutdown hook; langchain-mcp-adapters manages per-session HTTP clients."""
    return None


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


MAX_MCP_RESPONSE_BYTES = 4 * 1024 * 1024  # 4 MB response size guard

_SSRF_BLOCKED_NETWORKS = (
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("169.254.0.0/16"),
    ipaddress.ip_network("100.64.0.0/10"),
    ipaddress.ip_network("0.0.0.0/8"),
    ipaddress.ip_network("::1/128"),
    ipaddress.ip_network("fc00::/7"),
    ipaddress.ip_network("fe80::/10"),
)

_CLOUD_METADATA_HOSTS = frozenset({
    "metadata.google.internal",
    "metadata.goog",
    "169.254.169.254",
    "fd00:ec2::254",
})

_LOCALHOST_HOSTNAMES = frozenset({
    "localhost",
    "localhost.localdomain",
    "ip6-localhost",
    "ip6-loopback",
})


def _check_ip_allowed(ip: ipaddress._BaseAddress, *, allow_private: bool = False) -> None:
    if ip.is_link_local or ip.is_multicast:
        raise ValueError("Link-local and multicast addresses are not allowed.")
    if not allow_private:
        for net in _SSRF_BLOCKED_NETWORKS:
            if ip in net:
                raise ValueError(
                    f"Connections to private, loopback, link-local, or reserved IP addresses are not allowed ({ip})."
                )
        if ip.is_private or ip.is_loopback or ip.is_reserved or ip.is_unspecified:
            raise ValueError(
                f"Connections to private, loopback, link-local, or reserved IP addresses are not allowed ({ip})."
            )


def validate_mcp_url(
    raw_url: str,
    *,
    allow_private: bool = False,
    require_https: bool = False,
) -> str:
    """Validate and normalize a user-supplied HTTP(S) MCP server URL with SSRF protection."""
    trimmed = (raw_url or "").strip()
    if not trimmed:
        raise ValueError("MCP server URL is required.")
    parsed = urlparse(trimmed)
    allowed_schemes = {"https"} if require_https else {"http", "https"}
    if parsed.scheme not in allowed_schemes or not parsed.netloc or not parsed.hostname:
        if require_https:
            raise ValueError("MCP server URL must use https://.")
        raise ValueError("Enter a valid http:// or https:// MCP server URL.")

    hostname = parsed.hostname.lower().rstrip(".")
    if hostname in _CLOUD_METADATA_HOSTS:
        raise ValueError("Access to cloud metadata hosts is blocked.")
    if not allow_private and hostname in _LOCALHOST_HOSTNAMES:
        raise ValueError("Connections to loopback/localhost addresses are not allowed.")

    ip_obj: Optional[ipaddress._BaseAddress] = None
    try:
        ip_obj = ipaddress.ip_address(hostname)
    except ValueError:
        ip_obj = None

    if ip_obj is not None:
        _check_ip_allowed(ip_obj, allow_private=allow_private)

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


_DYN_REG_CACHE: Dict[Tuple[str, str, str], Tuple[str, Optional[str]]] = {}


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
                            "revocation_endpoint": as_meta.get("revocation_endpoint"),
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
    user_id: str = "",
) -> Tuple[str, Optional[str]]:
    """Dynamically register an OAuth client (RFC 7591) for an MCP authorization server."""
    cache_key = (user_id, registration_endpoint, redirect_uri)
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


async def revoke_mcp_oauth_token(
    client: httpx.AsyncClient,
    revocation_endpoint: str,
    token: str,
    client_id: Optional[str] = None,
    client_secret: Optional[str] = None,
    token_type_hint: str = "access_token",
) -> bool:
    """Revoke an OAuth token via RFC 7009 revocation endpoint."""
    if not revocation_endpoint or not token:
        return False
    data: Dict[str, str] = {"token": token, "token_type_hint": token_type_hint}
    headers = {"Accept": "application/json", "Content-Type": "application/x-www-form-urlencoded"}
    try:
        if client_id and client_secret:
            r = await client.post(
                revocation_endpoint, data=data, headers=headers, auth=(client_id, client_secret), timeout=8.0
            )
        else:
            if client_id:
                data["client_id"] = client_id
            r = await client.post(revocation_endpoint, data=data, headers=headers, timeout=8.0)
        return r.status_code in (200, 204)
    except Exception:
        return False



from datetime import timedelta
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools


def _unwrap_mcp_exception(exc: BaseException) -> Tuple[Optional[int], str, Optional[httpx.Response]]:
    """Extract HTTP status code, message, and response from nested ExceptionGroups."""
    stack: List[BaseException] = [exc]
    while stack:
        cur = stack.pop(0)
        if isinstance(cur, BaseExceptionGroup):
            stack.extend(cur.exceptions)
            continue
        if isinstance(cur, httpx.HTTPStatusError):
            resp = cur.response
            return resp.status_code, str(cur), resp
        if isinstance(cur, httpx.TimeoutException):
            return None, "Timed out while connecting to the MCP server.", None
        if isinstance(cur, httpx.RequestError):
            return None, f"Could not connect: {cur}", None
        cause = getattr(cur, "__cause__", None) or getattr(cur, "__context__", None)
        if cause is not None:
            stack.append(cause)
    return None, str(exc), None


def _build_mcp_connection(
    url: str,
    headers: Dict[str, str],
    transport: str,
    timeout: float,
) -> Dict[str, Any]:
    if transport == "sse":
        return {
            "transport": "sse",
            "url": url,
            "headers": headers,
            "timeout": float(timeout),
            "sse_read_timeout": max(float(timeout), 30.0),
        }
    return {
        "transport": "streamable_http",
        "url": url,
        "headers": headers,
        "timeout": timedelta(seconds=float(timeout)),
        "sse_read_timeout": timedelta(seconds=max(float(timeout), 30.0)),
    }


def _langchain_tool_to_dict(tool: Any) -> Dict[str, Any]:
    """Convert a LangChain BaseTool from langchain-mcp-adapters into Orbit's tool dict."""
    raw_schema = getattr(tool, "args_schema", None)
    if not isinstance(raw_schema, dict):
        raw_schema = {}
    meta = getattr(tool, "metadata", None)
    annotations = meta.get("annotations") if isinstance(meta, dict) else None
    name = str(getattr(tool, "name", "")).strip()
    read_only = is_mcp_tool_read_only(name, annotations if isinstance(annotations, dict) else None)
    return {
        "name": name,
        "description": str(getattr(tool, "description", "") or "").strip(),
        "inputSchema": sanitize_tool_schema(raw_schema),
        "readOnlyHint": read_only,
    }


def _format_call_tool_result(res: Any) -> str:
    """Format an MCP CallToolResult returned by ClientSession.call_tool."""
    content_list = getattr(res, "content", None)
    is_error = bool(getattr(res, "isError", False))
    if isinstance(content_list, list) and content_list:
        parts: List[str] = []
        for item in content_list:
            text = getattr(item, "text", None)
            if text is not None:
                parts.append(str(text))
            elif isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            elif hasattr(item, "model_dump"):
                parts.append(json.dumps(item.model_dump(), ensure_ascii=False))
            else:
                parts.append(str(item))
        text_out = "\n".join(p for p in parts if p).strip()
        if is_error:
            return f"MCP tool error: {text_out or 'Tool reported an error.'}"
        if text_out:
            return text_out
    structured = getattr(res, "structuredContent", None)
    if structured is not None:
        return json.dumps(structured, ensure_ascii=False)
    if is_error:
        return "MCP tool error: Tool reported an error."
    return ""


async def probe_mcp_server(
    mcp_url: str,
    custom_headers: Optional[Dict[str, str]] = None,
    timeout: float = 10.0,
    *,
    allow_private: bool = False,
) -> Dict[str, Any]:
    """Probe an MCP endpoint using langchain-mcp-adapters and discover tools/auth."""
    url = validate_mcp_url(mcp_url, allow_private=allow_private)
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
        "handshake_ok": False,
        "transport": None,
        "server_info": None,
        "tools": [],
        "oauth_metadata": None,
        "error": None,
    }

    last_exc: Optional[BaseException] = None
    for transport in ("streamable_http", "sse"):
        try:
            client = MultiServerMCPClient({"server": _build_mcp_connection(url, headers, transport, timeout)})
            async with client.session("server", auto_initialize=False) as session:
                init_res = await session.initialize()
                lc_tools = await load_mcp_tools(session)
                server_info = None
                if getattr(init_res, "serverInfo", None) is not None:
                    si = init_res.serverInfo
                    server_info = si.model_dump() if hasattr(si, "model_dump") else dict(si)

                result["reachable"] = True
                result["status_code"] = 200
                result["connect_status"] = "done"
                result["connect_detail"] = "Done"
                result["handshake_ok"] = True
                result["transport"] = transport
                result["server_info"] = server_info
                result["tools"] = [_langchain_tool_to_dict(t) for t in lc_tools if getattr(t, "name", None)]

                try:
                    async with httpx.AsyncClient(timeout=3.0, follow_redirects=True) as http_client:
                        oauth_resp = await http_client.get(_protected_resource_url(url), timeout=3.0)
                        if oauth_resp.status_code == 200 and "authorization_servers" in oauth_resp.text:
                            result["signin_settings_status"] = "Available"
                            result["signin_provider_status"] = "OAuth optional"
                            result["detected_auth"] = "later"
                except Exception:
                    pass
                return result
        except BaseException as exc:
            last_exc = exc
            status, msg, resp = _unwrap_mcp_exception(exc)
            if status == 405 and transport == "streamable_http":
                continue
            if status in (401, 403):
                result["status_code"] = status
                result["reachable"] = True
                result["connect_status"] = "done"
                result["connect_detail"] = "Done"
                www_auth = resp.headers.get("www-authenticate", "") if resp is not None else ""
                result["detected_auth"] = "now"
                result["signin_settings_status"] = "Detected"
                try:
                    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as http_client:
                        oauth_meta = await discover_mcp_oauth_metadata(http_client, url, www_auth)
                except Exception:
                    oauth_meta = None
                if oauth_meta:
                    result["oauth_metadata"] = oauth_meta
                    res_name = oauth_meta.get("resource_name") or ("Notion" if "notion" in url.lower() else "")
                    result["signin_provider_status"] = f"OAuth bearer ({res_name})" if res_name else "OAuth bearer"
                else:
                    result["signin_provider_status"] = "OAuth bearer" if "bearer" in www_auth.lower() else "Auth required"
                return result
            if status is not None:
                result["status_code"] = status
                reason = resp.reason_phrase if resp is not None else str(status)
                result["connect_detail"] = reason
                result["error"] = f"Server returned HTTP {status} ({reason})."
                return result
            if "Timed out" in msg:
                result["connect_detail"] = "Timed out"
                result["error"] = msg
                return result
            if transport == "streamable_http":
                continue
            result["connect_detail"] = "Connection failed"
            result["error"] = msg
            return result

    if last_exc is not None:
        _, msg, _ = _unwrap_mcp_exception(last_exc)
        result["connect_detail"] = "Connection failed"
        result["error"] = msg
    return result


async def connect_and_list_mcp_tools(
    mcp_url: str,
    custom_headers: Optional[Dict[str, str]] = None,
    timeout: float = 12.0,
    *,
    allow_private: bool = False,
) -> Dict[str, Any]:
    """Connect to a remote MCP server via langchain-mcp-adapters and list tools."""
    url = validate_mcp_url(mcp_url, allow_private=allow_private)
    headers = _base_mcp_headers(custom_headers)

    for transport in ("streamable_http", "sse"):
        try:
            client = MultiServerMCPClient({"server": _build_mcp_connection(url, headers, transport, timeout)})
            lc_tools = await client.get_tools()
            tools = [_langchain_tool_to_dict(t) for t in lc_tools if getattr(t, "name", None)]
            return {"ok": True, "status_code": 200, "tools": tools, "error": None}
        except BaseException as exc:
            status, msg, resp = _unwrap_mcp_exception(exc)
            if status == 405 and transport == "streamable_http":
                continue
            if status in (401, 403):
                return {
                    "ok": False,
                    "status_code": status,
                    "tools": [],
                    "error": "Authentication rejected by MCP server. Configure credentials or request headers.",
                }
            if status is not None:
                reason = resp.reason_phrase if resp is not None else str(status)
                return {
                    "ok": False,
                    "status_code": status,
                    "tools": [],
                    "error": f"Unable to connect ({status} {reason}).",
                }
            if transport == "streamable_http":
                continue
            return {"ok": False, "status_code": None, "tools": [], "error": msg}

    return {"ok": False, "status_code": None, "tools": [], "error": "Unable to connect to MCP server."}


async def call_mcp_tool(
    mcp_url: str,
    tool_name: str,
    arguments: Dict[str, Any],
    custom_headers: Optional[Dict[str, str]] = None,
    timeout: float = 25.0,
    *,
    allow_private: bool = False,
) -> str:
    """Invoke a remote MCP tool via langchain-mcp-adapters."""
    url = validate_mcp_url(mcp_url, allow_private=allow_private)
    headers = _base_mcp_headers(custom_headers)

    last_exc: Optional[BaseException] = None
    for transport in ("streamable_http", "sse"):
        try:
            client = MultiServerMCPClient({"server": _build_mcp_connection(url, headers, transport, timeout)})
            async with client.session("server") as session:
                res = await session.call_tool(tool_name, arguments)
                return _format_call_tool_result(res)
        except BaseException as exc:
            last_exc = exc
            status, msg, resp = _unwrap_mcp_exception(exc)
            if status == 405 and transport == "streamable_http":
                continue
            if status is not None:
                reason = resp.reason_phrase if resp is not None else str(status)
                raise RuntimeError(f"MCP tool call failed with HTTP {status} ({reason})") from exc
            if transport == "streamable_http":
                continue
            raise RuntimeError(msg) from exc

    raise RuntimeError(str(last_exc) if last_exc else "MCP tool call failed.")
