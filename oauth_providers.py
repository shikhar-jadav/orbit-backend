"""Provider-aware OAuth 2.0 / 2.1 configuration and credential resolution for Orbit MCP connectors.

Critical rules:
- Never fabricate client_id or client_secret.
- Never use one provider's OAuth app credentials for another provider.
- Never assume every provider supports RFC 7591 dynamic client registration.
- If credentials are missing, return a clear setup_required diagnostic.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class ProviderOAuthSpec:
    provider_key: str
    display_name: str
    supports_dynamic_registration: bool
    requires_static_app: bool
    client_id_env_vars: Tuple[str, ...] = ()
    client_secret_env_vars: Tuple[str, ...] = ()
    default_scopes: Tuple[str, ...] = ("default",)
    extra_auth_params: Dict[str, str] = field(default_factory=dict)
    setup_docs_url: Optional[str] = None
    setup_instructions: str = ""


PROVIDER_OAUTH_SPECS: Dict[str, ProviderOAuthSpec] = {
    "google": ProviderOAuthSpec(
        provider_key="google",
        display_name="Google",
        supports_dynamic_registration=False,
        requires_static_app=True,
        client_id_env_vars=("ORBIT_GOOGLE_OAUTH_CLIENT_ID", "GOOGLE_OAUTH_CLIENT_ID", "GOOGLE_CLIENT_ID"),
        client_secret_env_vars=("ORBIT_GOOGLE_OAUTH_CLIENT_SECRET", "GOOGLE_OAUTH_CLIENT_SECRET", "GOOGLE_CLIENT_SECRET"),
        default_scopes=("openid", "email", "profile"),
        extra_auth_params={"access_type": "offline", "prompt": "consent"},
        setup_docs_url="https://console.cloud.google.com/apis/credentials",
        setup_instructions=(
            "Configure GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET (or ORBIT_GOOGLE_OAUTH_CLIENT_ID / "
            "ORBIT_GOOGLE_OAUTH_CLIENT_SECRET) in the Orbit backend environment."
        ),
    ),
    "github": ProviderOAuthSpec(
        provider_key="github",
        display_name="GitHub",
        supports_dynamic_registration=False,
        requires_static_app=True,
        client_id_env_vars=("ORBIT_GITHUB_OAUTH_CLIENT_ID", "GITHUB_OAUTH_CLIENT_ID", "GITHUB_CLIENT_ID"),
        client_secret_env_vars=("ORBIT_GITHUB_OAUTH_CLIENT_SECRET", "GITHUB_OAUTH_CLIENT_SECRET", "GITHUB_CLIENT_SECRET"),
        default_scopes=("repo", "read:user"),
        setup_docs_url="https://github.com/settings/developers",
        setup_instructions=(
            "Configure GITHUB_CLIENT_ID and GITHUB_CLIENT_SECRET (or ORBIT_GITHUB_OAUTH_CLIENT_ID / "
            "ORBIT_GITHUB_OAUTH_CLIENT_SECRET) in the Orbit backend environment."
        ),
    ),
    "slack": ProviderOAuthSpec(
        provider_key="slack",
        display_name="Slack",
        supports_dynamic_registration=False,
        requires_static_app=True,
        client_id_env_vars=("ORBIT_SLACK_OAUTH_CLIENT_ID", "SLACK_CLIENT_ID"),
        client_secret_env_vars=("ORBIT_SLACK_OAUTH_CLIENT_SECRET", "SLACK_CLIENT_SECRET"),
        default_scopes=("channels:read", "chat:write", "users:read"),
        setup_docs_url="https://api.slack.com/apps",
        setup_instructions=(
            "Create a Slack App with Orbit's OAuth callback URL and set "
            "ORBIT_SLACK_OAUTH_CLIENT_ID and ORBIT_SLACK_OAUTH_CLIENT_SECRET."
        ),
    ),
    "figma": ProviderOAuthSpec(
        provider_key="figma",
        display_name="Figma",
        supports_dynamic_registration=False,
        requires_static_app=True,
        client_id_env_vars=("ORBIT_FIGMA_OAUTH_CLIENT_ID", "FIGMA_CLIENT_ID"),
        client_secret_env_vars=("ORBIT_FIGMA_OAUTH_CLIENT_SECRET", "FIGMA_CLIENT_SECRET"),
        default_scopes=("file_read",),
        setup_docs_url="https://www.figma.com/developers/apps",
        setup_instructions=(
            "Register a Figma OAuth App and set ORBIT_FIGMA_OAUTH_CLIENT_ID and "
            "ORBIT_FIGMA_OAUTH_CLIENT_SECRET."
        ),
    ),
    "atlassian": ProviderOAuthSpec(
        provider_key="atlassian",
        display_name="Atlassian",
        supports_dynamic_registration=False,
        requires_static_app=True,
        client_id_env_vars=("ORBIT_ATLASSIAN_OAUTH_CLIENT_ID", "ATLASSIAN_CLIENT_ID"),
        client_secret_env_vars=("ORBIT_ATLASSIAN_OAUTH_CLIENT_SECRET", "ATLASSIAN_CLIENT_SECRET"),
        default_scopes=("read:jira-work", "read:confluence-content.all", "offline_access"),
        setup_docs_url="https://developer.atlassian.com/console/myapps/",
        setup_instructions=(
            "Create an Atlassian OAuth 2.0 (3LO) app and set "
            "ORBIT_ATLASSIAN_OAUTH_CLIENT_ID and ORBIT_ATLASSIAN_OAUTH_CLIENT_SECRET."
        ),
    ),
    "hubspot": ProviderOAuthSpec(
        provider_key="hubspot",
        display_name="HubSpot",
        supports_dynamic_registration=False,
        requires_static_app=True,
        client_id_env_vars=("ORBIT_HUBSPOT_OAUTH_CLIENT_ID", "HUBSPOT_CLIENT_ID"),
        client_secret_env_vars=("ORBIT_HUBSPOT_OAUTH_CLIENT_SECRET", "HUBSPOT_CLIENT_SECRET"),
        default_scopes=("crm.objects.contacts.read",),
        setup_docs_url="https://developers.hubspot.com/",
        setup_instructions=(
            "Create a HubSpot public/OAuth app and set "
            "ORBIT_HUBSPOT_OAUTH_CLIENT_ID and ORBIT_HUBSPOT_OAUTH_CLIENT_SECRET."
        ),
    ),
    "notion": ProviderOAuthSpec(
        provider_key="notion",
        display_name="Notion",
        supports_dynamic_registration=True,
        requires_static_app=False,
        client_id_env_vars=("ORBIT_NOTION_OAUTH_CLIENT_ID",),
        client_secret_env_vars=("ORBIT_NOTION_OAUTH_CLIENT_SECRET",),
        default_scopes=("default",),
    ),
    "linear": ProviderOAuthSpec(
        provider_key="linear",
        display_name="Linear",
        supports_dynamic_registration=True,
        requires_static_app=False,
        client_id_env_vars=("ORBIT_LINEAR_OAUTH_CLIENT_ID",),
        client_secret_env_vars=("ORBIT_LINEAR_OAUTH_CLIENT_SECRET",),
        default_scopes=("read", "write"),
    ),
    "asana": ProviderOAuthSpec(
        provider_key="asana",
        display_name="Asana",
        supports_dynamic_registration=True,
        requires_static_app=False,
        client_id_env_vars=("ORBIT_ASANA_OAUTH_CLIENT_ID",),
        client_secret_env_vars=("ORBIT_ASANA_OAUTH_CLIENT_SECRET",),
        default_scopes=("default",),
    ),
    "box": ProviderOAuthSpec(
        provider_key="box",
        display_name="Box",
        supports_dynamic_registration=True,
        requires_static_app=False,
        client_id_env_vars=("ORBIT_BOX_OAUTH_CLIENT_ID",),
        client_secret_env_vars=("ORBIT_BOX_OAUTH_CLIENT_SECRET",),
        default_scopes=("root_readwrite",),
    ),
    "miro": ProviderOAuthSpec(
        provider_key="miro",
        display_name="Miro",
        supports_dynamic_registration=True,
        requires_static_app=False,
        client_id_env_vars=("ORBIT_MIRO_OAUTH_CLIENT_ID",),
        client_secret_env_vars=("ORBIT_MIRO_OAUTH_CLIENT_SECRET",),
        default_scopes=("boards:read",),
    ),
}


def get_provider_oauth_spec(provider_key: Optional[str]) -> Optional[ProviderOAuthSpec]:
    if not provider_key:
        return None
    return PROVIDER_OAUTH_SPECS.get(provider_key.lower().strip())


def get_env_oauth_credentials(provider_key: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    """Read Orbit-managed OAuth client_id and client_secret for a provider from env vars."""
    spec = get_provider_oauth_spec(provider_key)
    if not spec:
        return None, None
    client_id: Optional[str] = None
    client_secret: Optional[str] = None
    for env_name in spec.client_id_env_vars:
        val = (os.getenv(env_name) or "").strip()
        if val:
            client_id = val
            break
    for env_name in spec.client_secret_env_vars:
        val = (os.getenv(env_name) or "").strip()
        if val:
            client_secret = val
            break
    return client_id, client_secret


def is_provider_oauth_configured(provider_key: Optional[str]) -> bool:
    """Return True if the provider does not need static env credentials OR has them configured."""
    spec = get_provider_oauth_spec(provider_key)
    if not spec or not spec.requires_static_app:
        return True
    cid, csec = get_env_oauth_credentials(provider_key)
    return bool(cid and csec)


def resolve_oauth_client_credentials(
    provider_key: Optional[str],
    connector_advanced: Optional[Dict[str, Any]],
    oauth_meta: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Resolve OAuth client credentials without ever fabricating values.

    Resolution order:
    1. Connector-level user/admin supplied `oauth_client_id` / `oauth_client_secret`.
    2. Orbit-managed environment credentials for `provider_key`.
    3. Dynamic client registration eligibility (if `registration_endpoint` exists and
       the provider does not disallow dynamic registration).
    4. Clear `setup_required` diagnostic.
    """
    adv = connector_advanced or {}
    cid = (adv.get("oauth_client_id") or "").strip() or None
    csec = (adv.get("oauth_client_secret") or "").strip() or None
    if cid:
        return {
            "source": "connector",
            "client_id": cid,
            "client_secret": csec,
            "can_dynamic_register": False,
            "setup_required": False,
            "setup_instructions": None,
        }

    env_cid, env_csec = get_env_oauth_credentials(provider_key)
    if env_cid:
        return {
            "source": "orbit_env",
            "client_id": env_cid,
            "client_secret": env_csec,
            "can_dynamic_register": False,
            "setup_required": False,
            "setup_instructions": None,
        }

    spec = get_provider_oauth_spec(provider_key)
    has_reg_endpoint = bool((oauth_meta or {}).get("registration_endpoint"))
    if has_reg_endpoint and (not spec or spec.supports_dynamic_registration):
        return {
            "source": "dynamic_registration",
            "client_id": None,
            "client_secret": None,
            "can_dynamic_register": True,
            "setup_required": False,
            "setup_instructions": None,
        }

    instructions = (
        spec.setup_instructions
        if spec and spec.setup_instructions
        else "Configure an OAuth Client ID (and Client Secret if required) in Advanced settings or supply a Bearer / API token."
    )
    return {
        "source": "missing",
        "client_id": None,
        "client_secret": None,
        "can_dynamic_register": False,
        "setup_required": True,
        "setup_instructions": instructions,
    }


def list_provider_setup_requirements() -> List[Dict[str, Any]]:
    """Return status of all static-app OAuth providers for diagnostics/docs."""
    out: List[Dict[str, Any]] = []
    for key, spec in PROVIDER_OAUTH_SPECS.items():
        if not spec.requires_static_app:
            continue
        cid, csec = get_env_oauth_credentials(key)
        out.append({
            "provider": key,
            "display_name": spec.display_name,
            "configured": bool(cid and csec),
            "env_vars": list(spec.client_id_env_vars[:1] + spec.client_secret_env_vars[:1]),
            "setup_docs_url": spec.setup_docs_url,
            "setup_instructions": spec.setup_instructions,
        })
    return out
