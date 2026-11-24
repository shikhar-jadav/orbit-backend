"""Remote GitHub MCP Server integration using per-user Orbit GitHub OAuth tokens."""

from __future__ import annotations

import hashlib
import logging
import os
import time
from typing import Any, Dict, List, Optional

from credential_vault import CredentialReconnectRequiredError

logger = logging.getLogger("orbit.github_mcp")

DEFAULT_GITHUB_MCP_URL = "https://api.githubcopilot.com/mcp/"
_TOOLS_CACHE_TTL_SECONDS = 300.0
_GITHUB_TOOLS_CACHE: Dict[str, tuple[List[Dict[str, Any]], float]] = {}
_PROVIDER_INTEGRATION_STORE: Optional[Any] = None


def configure_provider_integration_store(store: Any) -> None:
    """Bind the shared ProviderIntegrationStore instance at startup."""
    global _PROVIDER_INTEGRATION_STORE
    _PROVIDER_INTEGRATION_STORE = store


def get_github_mcp_url() -> str:
    return os.getenv("GITHUB_MCP_URL", DEFAULT_GITHUB_MCP_URL).strip() or DEFAULT_GITHUB_MCP_URL


def resolve_user_github_token(user_id: Optional[str]) -> Optional[str]:
    """Return the decrypted GitHub OAuth access token for user_id, or None."""
    if not user_id or _PROVIDER_INTEGRATION_STORE is None:
        return None
    try:
        account = _PROVIDER_INTEGRATION_STORE.get_account(user_id, "github")
        if not account:
            return None
        token = str(account.get("access_token") or "").strip()
        return token or None
    except CredentialReconnectRequiredError as exc:
        logger.warning("GitHub credentials for user %s require reconnection: %s", user_id, exc)
        return None
    except Exception as exc:
        logger.warning("Could not resolve GitHub credentials for user %s: %s", user_id, exc)
        return None


def clear_github_tools_cache(user_id: Optional[str] = None) -> None:
    if user_id is None:
        _GITHUB_TOOLS_CACHE.clear()
        return
    prefix = f"{user_id}:"
    for key in [k for k in _GITHUB_TOOLS_CACHE if k.startswith(prefix)]:
        _GITHUB_TOOLS_CACHE.pop(key, None)


async def load_github_mcp_connector(user_id: Optional[str]) -> Optional[Dict[str, Any]]:
    """Discover tools from the remote GitHub MCP Server for the connected user."""
    access_token = resolve_user_github_token(user_id)
    if not access_token:
        return None

    mcp_url = get_github_mcp_url()
    token_hash = hashlib.sha256(access_token.encode("utf-8")).hexdigest()[:16]
    cache_key = f"{user_id}:{mcp_url}:{token_hash}"
    now = time.monotonic()

    cached = _GITHUB_TOOLS_CACHE.get(cache_key)
    if cached and cached[1] > now:
        tools = cached[0]
    else:
        import mcp_client

        headers = {"Authorization": f"Bearer {access_token}"}
        tools = await mcp_client.connect_and_list_mcp_tools(
            mcp_url, custom_headers=headers
        )
        _GITHUB_TOOLS_CACHE[cache_key] = (tools, now + _TOOLS_CACHE_TTL_SECONDS)

    return {
        "id": "marketplace_github_mcp",
        "name": "GitHub",
        "mcp_url": mcp_url,
        "is_connected": True,
        "is_cohosted": False,
        "headers": [{"key": "Authorization", "value": f"Bearer {access_token}"}],
        "tools": tools,
        "tool_permissions": {},
    }
