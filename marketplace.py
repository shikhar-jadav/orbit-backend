import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import connector_registry
import oauth_providers
from credential_vault import CredentialUnrecoverableError, CredentialVault

BASE_DIR = Path(__file__).resolve().parent
SKILLS_DIR = BASE_DIR / ".agents" / "skills"
SKILLS_LOCK_PATH = BASE_DIR / "skills-lock.json"

# ------------------------------------------------------------------------------
# Connectors Catalog
# ------------------------------------------------------------------------------
GMAIL_MCP_CONNECTOR: Dict[str, Any] = {
    "id": "gmail-mcp",
    "name": "Gmail",
    "category": "Email & Productivity",
    "description": (
        "Connect Gmail to allow Orbit agents to search mailboxes, read email threads, "
        "compose drafts, send messages, and manage labels with per-user OAuth security."
    ),
    "icon": "mail",
    "badge": "Google",
    "authType": "OAuth 2.0",
    "oauthProvider": "gmail",
    "website": "https://github.com/klodr/gmail-mcp",
    "capabilities": [
        "Read Emails & Threads",
        "Compose & Send Emails",
        "Create & Update Drafts",
        "Manage Labels & Filters",
        "Search Mailbox",
        "Sandboxed Attachments",
    ],
    "configFields": [],
}

GITHUB_MCP_CONNECTOR: Dict[str, Any] = {
    "id": "github-mcp",
    "name": "GitHub",
    "category": "Developer & Code",
    "description": (
        "Connect GitHub to browse repositories and code, manage issues and pull requests, "
        "monitor GitHub Actions workflows, analyze security alerts, and power GitHub event triggers."
    ),
    "icon": "gitBranch",
    "badge": "GitHub",
    "authType": "OAuth 2.0",
    "oauthProvider": "github",
    "website": "https://github.com/github/github-mcp-server",
    "capabilities": [
        "Browse Repositories & Code",
        "Manage Issues & Pull Requests",
        "Monitor GitHub Actions & CI/CD",
        "Code & Security Analysis",
        "GitHub Event Triggers & Webhooks",
    ],
    "configFields": [],
}

def _enrich_builtin_connectors() -> None:
    google_ok = oauth_providers.is_provider_oauth_configured("google")
    google_spec = oauth_providers.get_provider_oauth_spec("google")
    GMAIL_MCP_CONNECTOR.update({
        "provider": "google",
        "source": "orbit_adapter",
        "source_type": "builtin",
        "implementation_type": "orbit_adapter",
        "transport": "in_process",
        "authentication": "oauth2",
        "oauth_provider_key": "google",
        "supports_dynamic_registration": False,
        "verification_status": "auth_required" if google_ok else "setup_required",
        "support_status": "active" if google_ok else "setup_required",
        "verification_notes": (
            "Co-hosted Orbit Gmail MCP adapter with per-user Google OAuth."
            if google_ok
            else (google_spec.setup_instructions if google_spec else "Requires Google OAuth credentials.")
        ),
        "setup_instructions": None if google_ok else (google_spec.setup_instructions if google_spec else None),
    })

    github_ok = oauth_providers.is_provider_oauth_configured("github")
    github_spec = oauth_providers.get_provider_oauth_spec("github")
    GITHUB_MCP_CONNECTOR.update({
        "provider": "github",
        "source": "official",
        "source_type": "builtin",
        "implementation_type": "mcp_server",
        "transport": "streamable_http",
        "authentication": "oauth2",
        "oauth_provider_key": "github",
        "supports_dynamic_registration": False,
        "verification_status": "auth_required" if github_ok else "setup_required",
        "support_status": "active" if github_ok else "setup_required",
        "verification_notes": (
            "Official GitHub MCP integration with per-user OAuth."
            if github_ok
            else (github_spec.setup_instructions if github_spec else "Requires GitHub OAuth credentials.")
        ),
        "setup_instructions": None if github_ok else (github_spec.setup_instructions if github_spec else None),
    })


AVAILABLE_CONNECTORS: List[Dict[str, Any]] = [GMAIL_MCP_CONNECTOR, GITHUB_MCP_CONNECTOR]

MARKETPLACE_MCPS_PATH = BASE_DIR / "data" / "marketplace_mcps.json"


def _load_marketplace_mcps() -> None:
    _enrich_builtin_connectors()
    manifests = connector_registry.load_marketplace_manifests()
    existing_ids = {c["id"] for c in AVAILABLE_CONNECTORS}
    for manifest in manifests:
        if manifest.id in existing_ids:
            continue
        existing_ids.add(manifest.id)
        AVAILABLE_CONNECTORS.append(connector_registry.manifest_to_connector_dict(manifest))


def reload_marketplace_connectors() -> int:
    """Reload built-in and registry connectors (e.g., after verification)."""
    AVAILABLE_CONNECTORS.clear()
    AVAILABLE_CONNECTORS.extend([GMAIL_MCP_CONNECTOR, GITHUB_MCP_CONNECTOR])
    _load_marketplace_mcps()
    return len(AVAILABLE_CONNECTORS)


_load_marketplace_mcps()


def register_connector(connector: Dict[str, Any]) -> None:
    """Register or update a connector definition."""
    existing_idx = next(
        (i for i, c in enumerate(AVAILABLE_CONNECTORS) if c["id"] == connector["id"]),
        None,
    )
    if existing_idx is not None:
        AVAILABLE_CONNECTORS[existing_idx] = connector
    else:
        AVAILABLE_CONNECTORS.append(connector)


def clear_all_connectors() -> None:
    """Clear all registered connectors."""
    AVAILABLE_CONNECTORS.clear()


# ------------------------------------------------------------------------------
# Skills Catalog (Empty - Clean Marketplace)
# ------------------------------------------------------------------------------
CUSTOM_SKILLS: List[Dict[str, Any]] = []


def register_custom_skill(skill: Dict[str, Any]) -> None:
    """Register or update a custom skill definition."""
    existing_idx = next(
        (i for i, s in enumerate(CUSTOM_SKILLS) if s["id"] == skill["id"]),
        None,
    )
    if existing_idx is not None:
        CUSTOM_SKILLS[existing_idx] = skill
    else:
        CUSTOM_SKILLS.append(skill)


def clear_all_skills() -> None:
    """Clear all registered skills."""
    CUSTOM_SKILLS.clear()


# ------------------------------------------------------------------------------
# Skills Lockfile Helpers
# ------------------------------------------------------------------------------
def get_installed_skill_ids(
    chat_store: Any = None, user_id: Optional[str] = None
) -> set[str]:
    if chat_store is not None and user_id is not None:
        raw = chat_store.get_setting("marketplace_installed_skills", user_id=user_id)
        if not raw:
            return set()
        try:
            items = json.loads(raw)
            return set(items) if isinstance(items, list) else set()
        except Exception:
            return set()
    if not SKILLS_LOCK_PATH.is_file():
        return set()
    try:
        data = json.loads(SKILLS_LOCK_PATH.read_text(encoding="utf-8"))
        return set(data.get("skills", {}).keys())
    except Exception:
        return set()


def toggle_skill_installed(
    skill_id: str,
    install: bool,
    chat_store: Any = None,
    user_id: Optional[str] = None,
) -> bool:
    if chat_store is not None and user_id is not None:
        installed = get_installed_skill_ids(chat_store=chat_store, user_id=user_id)
        if install:
            installed.add(skill_id)
        else:
            installed.discard(skill_id)
        chat_store.set_setting(
            "marketplace_installed_skills",
            json.dumps(sorted(installed)),
            user_id=user_id,
        )
        return True

    try:
        if SKILLS_LOCK_PATH.is_file():
            data = json.loads(SKILLS_LOCK_PATH.read_text(encoding="utf-8"))
        else:
            data = {"version": 1, "skills": {}}
    except Exception:
        data = {"version": 1, "skills": {}}

    skills_dict = data.setdefault("skills", {})
    if install:
        skills_dict[skill_id] = {
            "source": "Orbit/Marketplace",
            "sourceType": "custom",
            "installed_at": "now",
        }
    else:
        skills_dict.pop(skill_id, None)

    SKILLS_LOCK_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return True


def list_available_skills(
    force_refresh: bool = False,
    chat_store: Any = None,
    user_id: Optional[str] = None,
    *,
    installed_ids: Optional[set[str]] = None,
) -> List[Dict[str, Any]]:
    """List available skills in the marketplace (starts empty)."""
    if installed_ids is None:
        installed_ids = get_installed_skill_ids(chat_store=chat_store, user_id=user_id)
    return [{**s, "is_installed": s["id"] in installed_ids} for s in CUSTOM_SKILLS]


def get_skill_readme_content(skill_id: str) -> Optional[str]:
    """Retrieve readme content for a skill."""
    custom = next((s for s in CUSTOM_SKILLS if s["id"] == skill_id), None)
    if custom and custom.get("readme"):
        return custom["readme"]
    skill_file = SKILLS_DIR / skill_id / "SKILL.md"
    if skill_file.is_file():
        try:
            return skill_file.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return None
    return None


# ------------------------------------------------------------------------------
# Connectors Status & Persistence
# ------------------------------------------------------------------------------
def _mask_secret(value: str) -> str:
    return "••••••" if value else ""


def list_connectors_with_status(
    chat_store: Any,
    user_id: Optional[str] = None,
    provider_integration_store: Optional[Any] = None,
    *,
    connector_settings: Optional[Dict[str, str]] = None,
    public_accounts_list: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    if connector_settings is None:
        connector_settings = chat_store.get_settings_by_prefix("connector_", user_id=user_id)
    accounts_by_provider: Dict[str, Dict[str, Any]] = {}
    if user_id and provider_integration_store:
        if public_accounts_list is None and hasattr(provider_integration_store, "public_accounts"):
            try:
                public_accounts_list = provider_integration_store.public_accounts(user_id)
            except Exception:
                public_accounts_list = None
        if isinstance(public_accounts_list, list):
            for acc in public_accounts_list:
                prov = str(acc.get("provider") or "")
                if prov and prov not in accounts_by_provider:
                    accounts_by_provider[prov] = acc
    personal_by_marketplace_id: dict[str, dict[str, Any]] = {}
    if chat_store is not None and user_id:
        try:
            for item in chat_store.list_personal_connectors_for_user(user_id):
                mid = (item.get("advanced") or {}).get("marketplace_id")
                if mid:
                    personal_by_marketplace_id[str(mid)] = item
        except Exception:
            pass
    results = []
    vault = CredentialVault()

    for conn in AVAILABLE_CONNECTORS:
        conn_id = conn["id"]
        enabled_key = f"connector_{conn_id}_enabled"
        config_key = f"connector_{conn_id}_config"

        is_enabled = connector_settings.get(enabled_key) == "1"
        raw_config = connector_settings.get(config_key, "{}")
        try:
            config_dict = json.loads(raw_config) if raw_config else {}
        except Exception:
            config_dict = {}

        reconnect_required = False
        reconnect_message: Optional[str] = None
        integration_ctx = f"marketplace_connector:{conn_id}"
        uid = str(user_id or "global")
        for field in conn.get("configFields", []):
            if field.get("type") == "password":
                key = field.get("key")
                if key and config_dict.get(key):
                    try:
                        config_dict[key] = vault.decrypt_text(
                            str(config_dict[key]), user_id=uid, integration=integration_ctx
                        )
                    except CredentialUnrecoverableError:
                        config_dict[key] = ""
                        reconnect_required = True
                        reconnect_message = (
                            f"{conn['name']} credentials use an outdated encryption key and need to be re-entered."
                        )

        # Check required fields
        required_keys = [f["key"] for f in conn.get("configFields", []) if f.get("required")]
        all_required_set = all(bool(config_dict.get(k)) for k in required_keys)
        is_configured = (not required_keys) or all_required_set

        # Check OAuth provider connection
        oauth_provider = conn.get("oauthProvider")
        connected_account = None
        oauth_reconnect_required = False
        if oauth_provider and user_id and provider_integration_store:
            try:
                acc = accounts_by_provider.get(oauth_provider)
                if acc is None and public_accounts_list is None:
                    acc = provider_integration_store.get_account(
                        user_id, oauth_provider, allow_unrecoverable=True
                    )
                if acc:
                    oauth_reconnect_required = bool(acc.get("reconnect_required"))
                    connected_account = {
                        "account_name": acc.get("account_name"),
                        "email": acc.get("provider_account_id"),
                        "reconnect_required": oauth_reconnect_required,
                    }
                    if oauth_reconnect_required:
                        reconnect_required = True
                        reconnect_message = acc.get("reconnect_message") or (
                            f"Your {oauth_provider.title()} account needs to be reconnected."
                        )
            except Exception:
                pass

        personal_conn: Optional[Dict[str, Any]] = None
        if conn.get("mcpUrl") and user_id:
            personal_conn = personal_by_marketplace_id.get(conn_id)
            is_connected_remote = bool(personal_conn and personal_conn.get("is_connected"))
            is_enabled = is_connected_remote
            is_authenticated = is_connected_remote
        else:
            is_authenticated = (
                (bool(connected_account) and not oauth_reconnect_required)
                if oauth_provider
                else True
            )

        ver_status = str(conn.get("verification_status") or "auth_required")
        if is_enabled and is_configured and is_authenticated and not reconnect_required:
            status = "active"
        elif ver_status in ("unsupported", "deprecated", "package_available"):
            status = ver_status
        elif is_enabled and (not is_authenticated or reconnect_required):
            status = "needs_auth"
        elif is_enabled and not is_configured:
            status = "needs_config"
        elif ver_status == "setup_required":
            status = "setup_required"
        elif ver_status == "unverified":
            status = "unverified"
        else:
            status = "available"

        # Mask password fields
        masked_config: dict[str, Any] = {}
        for f in conn.get("configFields", []):
            val = config_dict.get(f["key"])
            if val is not None:
                if f.get("type") == "password":
                    masked_config[f["key"]] = _mask_secret(str(val))
                else:
                    masked_config[f["key"]] = val

        discovered_tools = (personal_conn.get("tools") or []) if personal_conn else []
        results.append({
            **conn,
            "type": "connector",
            "is_enabled": is_enabled,
            "is_configured": is_configured,
            "is_authenticated": is_authenticated,
            "connected_account": connected_account,
            "status": status,
            "reconnect_required": reconnect_required,
            "reconnect_message": reconnect_message,
            "config": masked_config,
            "tool_count": len(discovered_tools) if personal_conn else None,
            "tools": discovered_tools if personal_conn else [],
            "last_checked_at": personal_conn.get("last_checked_at") if personal_conn else None,
            "last_status_code": personal_conn.get("last_status_code") if personal_conn else None,
            "personal_connector_id": personal_conn.get("id") if personal_conn else None,
        })

    return results


def save_connector_configuration(
    chat_store: Any,
    connector_id: str,
    enabled: bool,
    config: Dict[str, Any],
    user_id: Optional[str] = None,
) -> Dict[str, Any]:
    target = next((c for c in AVAILABLE_CONNECTORS if c["id"] == connector_id), None)
    if not target:
        raise KeyError(f"Unknown connector {connector_id}")

    config_key = f"connector_{connector_id}_config"
    enabled_key = f"connector_{connector_id}_enabled"

    existing_raw = chat_store.get_setting(config_key, "{}", user_id=user_id)
    try:
        existing_config = json.loads(existing_raw) if existing_raw else {}
    except Exception:
        existing_config = {}

    vault = CredentialVault()
    uid = str(user_id or "global")
    integration_ctx = f"marketplace_connector:{connector_id}"
    merged_config = dict(existing_config)
    for field in target.get("configFields", []):
        if field.get("type") == "password":
            key = field.get("key")
            if key and existing_config.get(key):
                try:
                    merged_config[key] = vault.decrypt_text(
                        str(existing_config[key]), user_id=uid, integration=integration_ctx
                    )
                except CredentialUnrecoverableError:
                    merged_config[key] = ""

    for k, v in config.items():
        if isinstance(v, str) and "••••" in v:
            continue
        if v is not None:
            merged_config[k] = v

    for field in target.get("configFields", []):
        if field.get("type") == "password":
            key = field.get("key")
            if key and merged_config.get(key):
                merged_config[key] = vault.encrypt_text(
                    str(merged_config[key]), user_id=uid, integration=integration_ctx
                )

    chat_store.set_setting(config_key, json.dumps(merged_config, ensure_ascii=False), user_id=user_id)
    chat_store.set_setting(enabled_key, "1" if enabled else "0", user_id=user_id)

    return {
        "connector_id": connector_id,
        "enabled": enabled,
        "is_configured": all(
            bool(merged_config.get(f["key"]))
            for f in target.get("configFields", [])
            if f.get("required")
        ),
    }


def toggle_connector_enabled(
    chat_store: Any,
    connector_id: str,
    enabled: bool,
    user_id: Optional[str] = None,
) -> bool:
    target = next((c for c in AVAILABLE_CONNECTORS if c["id"] == connector_id), None)
    if not target:
        raise KeyError(f"Unknown connector {connector_id}")
    enabled_key = f"connector_{connector_id}_enabled"
    chat_store.set_setting(enabled_key, "1" if enabled else "0", user_id=user_id)
    return True
