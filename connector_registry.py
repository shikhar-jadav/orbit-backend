"""Orbit Connector Registry — structured, versioned manifest for all marketplace connectors.

Loads `data/marketplace_mcps.json` (the 963-entry discovery catalogue) and optionally
merges read-only probe results from `data/marketplace_verification.json`.

Preserves existing connector IDs so user connections and settings are never lost.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import oauth_providers

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MARKETPLACE_MCPS_PATH = DATA_DIR / "marketplace_mcps.json"
VERIFICATION_REPORT_PATH = DATA_DIR / "marketplace_verification.json"

# Explicit catalogue verification states
STATUS_VERIFIED = "verified"
STATUS_AUTH_REQUIRED = "auth_required"
STATUS_SETUP_REQUIRED = "setup_required"
STATUS_PACKAGE_AVAILABLE = "package_available"
STATUS_ADAPTER_REQUIRED = "adapter_required"
STATUS_UNVERIFIED = "unverified"
STATUS_UNSUPPORTED = "unsupported"
STATUS_DEPRECATED = "deprecated"

ALL_VERIFICATION_STATUSES = frozenset({
    STATUS_VERIFIED,
    STATUS_AUTH_REQUIRED,
    STATUS_SETUP_REQUIRED,
    STATUS_PACKAGE_AVAILABLE,
    STATUS_ADAPTER_REQUIRED,
    STATUS_UNVERIFIED,
    STATUS_UNSUPPORTED,
    STATUS_DEPRECATED,
})

_CLAUDE_HOST_PATTERNS = (
    r"\.mcp\.claude\.com$",
    r"^academy\.claude\.com$",
)

_EXAMPLE_HOST_PATTERNS = (
    r"^example-server\.modelcontextprotocol\.io$",
)


@dataclass
class ConnectorManifest:
    id: str
    name: str
    provider: str
    description: str
    category: str
    icon: str
    source: str                     # "official" | "third_party" | "orbit_adapter"
    source_type: str                # "remote_http" | "package" | "adapter" | "builtin"
    implementation_type: str        # "mcp_server" | "orbit_adapter" | "none"
    server_url: Optional[str]
    package_name: Optional[str]
    repository_url: Optional[str]
    documentation_url: Optional[str]
    transport: str                  # "streamable_http" | "sse" | "stdio" | "in_process"
    authentication: str             # "oauth2" | "api_key" | "bearer" | "none"
    oauth_provider_key: Optional[str]
    supports_dynamic_registration: bool
    required_config: List[Dict[str, Any]] = field(default_factory=list)
    capabilities: List[str] = field(default_factory=list)
    verification_status: str = STATUS_AUTH_REQUIRED
    support_status: str = "active"
    last_verified_at: Optional[str] = None
    verification_notes: str = ""
    setup_instructions: Optional[str] = None
    badge: Optional[str] = None


def _is_claude_hosted(hostname: str) -> bool:
    h = (hostname or "").lower().strip()
    return any(re.search(pat, h) for pat in _CLAUDE_HOST_PATTERNS)


def _is_example_server(hostname: str) -> bool:
    h = (hostname or "").lower().strip()
    return any(re.search(pat, h) for pat in _EXAMPLE_HOST_PATTERNS)


def _infer_provider_key(url: str, conn_id: str, name: str) -> Optional[str]:
    u = (url or "").lower()
    cid = (conn_id or "").lower()
    n = (name or "").lower()
    if "googleapis.com" in u or cid in ("google-drive", "google-calendar", "bigquery", "google-compute-engine", "gmail-mcp"):
        return "google"
    if "mcp.slack.com" in u or cid == "slack":
        return "slack"
    if "mcp.figma.com" in u or cid == "figma":
        return "figma"
    if "mcp.atlassian.com" in u or cid == "atlassian":
        return "atlassian"
    if "mcp.hubspot.com" in u or cid == "hubspot":
        return "hubspot"
    if "mcp.notion.com" in u or cid == "notion":
        return "notion"
    if "mcp.linear.app" in u or cid == "linear":
        return "linear"
    if "mcp.asana.com" in u or cid == "asana":
        return "asana"
    if "mcp.box.com" in u or cid == "box":
        return "box"
    if "mcp.miro.com" in u or cid == "miro":
        return "miro"
    if "github" in cid or "github.com" in u:
        return "github"
    if "zapier" in u or cid == "zapier":
        return "zapier"
    if "dune.com" in u or cid == "dune":
        return "dune"
    if "supabase" in u or cid == "supabase":
        return "supabase"
    if "canva" in u or cid == "canva":
        return "canva"
    if "intercom" in u or cid == "intercom":
        return "intercom"
    if "monday.com" in u or cid == "monday":
        return "monday"
    if "workos" in u or cid == "workos":
        return "workos"
    return None


def _infer_category(name: str, description: str, provider_key: Optional[str]) -> str:
    text = f"{name} {description}".lower()
    if provider_key in ("google", "atlassian", "notion", "linear", "asana", "monday", "box", "miro"):
        return "Productivity"
    if provider_key in ("slack", "intercom"):
        return "Communication"
    if provider_key in ("github",):
        return "Developer & Code"
    if provider_key in ("hubspot",):
        return "CRM & Sales"
    if any(kw in text for kw in ("database", "sql", "postgres", "mysql", "supabase", "bigquery")):
        return "Data & Analytics"
    if any(kw in text for kw in ("cloud", "aws", "azure", "gcp", "kubernetes", "compute")):
        return "Cloud & Infrastructure"
    if any(kw in text for kw in ("payment", "stripe", "billing", "finance", "invoice")):
        return "Finance & Commerce"
    if any(kw in text for kw in ("design", "figma", "canva", "adobe", "image", "video")):
        return "Design & Creative"
    if any(kw in text for kw in ("security", "auth", "identity", "sso")):
        return "Security & Identity"
    return "MCP Directory"


def _load_verification_map() -> Dict[str, Dict[str, Any]]:
    if not VERIFICATION_REPORT_PATH.is_file():
        return {}
    try:
        data = json.loads(VERIFICATION_REPORT_PATH.read_text(encoding="utf-8"))
        results = data.get("results", []) if isinstance(data, dict) else data
        if not isinstance(results, list):
            return {}
        return {str(r["id"]): r for r in results if isinstance(r, dict) and r.get("id")}
    except Exception as exc:
        logger.warning("Could not load marketplace_verification.json: %s", exc)
        return {}


def classify_marketplace_entry(
    entry: Dict[str, Any],
    verification_info: Optional[Dict[str, Any]] = None,
) -> ConnectorManifest:
    """Build and classify a ConnectorManifest from a marketplace_mcps.json entry."""
    cid = str(entry.get("id") or "").strip()
    name = str(entry.get("name") or cid).strip()
    description = str(entry.get("description") or "").strip()
    mcp_url = str(entry.get("mcpUrl") or entry.get("server_url") or "").strip() or None
    website = str(entry.get("website") or entry.get("documentation_url") or mcp_url or "").strip() or None

    parsed_host = ""
    if mcp_url:
        try:
            parsed_host = (urlparse(mcp_url).hostname or "").lower()
        except Exception:
            parsed_host = ""

    provider_key = _infer_provider_key(mcp_url or "", cid, name)
    spec = oauth_providers.get_provider_oauth_spec(provider_key)
    supports_dyn_reg = bool(spec.supports_dynamic_registration) if spec else True

    setup_instructions: Optional[str] = None
    last_verified_at: Optional[str] = None
    transport = "streamable_http"
    authentication = "oauth2" if (spec or (mcp_url and "oauth" in description.lower())) else "bearer"

    if not mcp_url:
        verification_status = STATUS_UNVERIFIED
        support_status = "unknown"
        verification_notes = "No remote MCP endpoint URL is configured."
    elif _is_claude_hosted(parsed_host):
        verification_status = STATUS_UNSUPPORTED
        support_status = "unsupported"
        verification_notes = (
            f"Endpoint ({parsed_host}) is hosted specifically for Claude and cannot be connected independently by Orbit."
        )
    elif _is_example_server(parsed_host):
        verification_status = STATUS_UNSUPPORTED
        support_status = "unsupported"
        verification_notes = "Reference example server URL — not a production integration."
    elif spec and spec.requires_static_app and not oauth_providers.is_provider_oauth_configured(provider_key):
        verification_status = STATUS_SETUP_REQUIRED
        support_status = "setup_required"
        setup_instructions = spec.setup_instructions
        verification_notes = (
            f"Requires Orbit administrator OAuth app credentials for {spec.display_name}. "
            f"{spec.setup_instructions}"
        )
    else:
        verification_status = STATUS_AUTH_REQUIRED
        support_status = "active"
        verification_notes = "Remote MCP endpoint; user authentication or API credentials required to connect."

    # Merge live verification probe evidence if available (without overriding unsupported/setup_required)
    if verification_info:
        last_verified_at = verification_info.get("last_verified_at")
        if verification_info.get("transport"):
            transport = str(verification_info["transport"])
        if "dynamic_registration_supported" in verification_info:
            supports_dyn_reg = bool(verification_info["dynamic_registration_supported"])
        v_status = verification_info.get("verification_status")
        if verification_status not in (STATUS_UNSUPPORTED, STATUS_SETUP_REQUIRED):
            if v_status in ALL_VERIFICATION_STATUSES:
                verification_status = v_status
            if verification_info.get("error"):
                verification_notes = f"Last probe note: {verification_info['error']}"
            elif v_status == STATUS_VERIFIED:
                t_count = verification_info.get("tools_count", 0)
                verification_notes = f"Verified via live MCP handshake ({t_count} tools discovered without auth)."
            elif v_status == STATUS_AUTH_REQUIRED:
                auth_disc = verification_info.get("auth_discovered") or "OAuth / Bearer"
                verification_notes = f"Endpoint reachable; authentication required ({auth_disc})."

    return ConnectorManifest(
        id=cid,
        name=name,
        provider=provider_key or name,
        description=description,
        category=_infer_category(name, description, provider_key),
        icon="plug",
        source="official",
        source_type="remote_http" if mcp_url else "none",
        implementation_type="mcp_server" if mcp_url else "none",
        server_url=mcp_url,
        package_name=None,
        repository_url=None,
        documentation_url=website,
        transport=transport,
        authentication=authentication,
        oauth_provider_key=provider_key,
        supports_dynamic_registration=supports_dyn_reg,
        required_config=[],
        capabilities=["Remote MCP Tools"],
        verification_status=verification_status,
        support_status=support_status,
        last_verified_at=last_verified_at,
        verification_notes=verification_notes,
        setup_instructions=setup_instructions,
        badge=spec.display_name if spec else "MCP",
    )


def load_marketplace_manifests(force_reload: bool = False) -> List[ConnectorManifest]:
    """Load and classify all entries from data/marketplace_mcps.json."""
    if not MARKETPLACE_MCPS_PATH.is_file():
        return []
    try:
        entries = json.loads(MARKETPLACE_MCPS_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.error("Failed to read %s: %s", MARKETPLACE_MCPS_PATH, exc)
        return []
    if not isinstance(entries, list):
        return []

    ver_map = _load_verification_map()
    manifests: List[ConnectorManifest] = []
    seen: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        cid = str(entry.get("id") or "").strip()
        if not cid or cid in seen:
            continue
        seen.add(cid)
        manifests.append(classify_marketplace_entry(entry, ver_map.get(cid)))
    return manifests


def manifest_to_connector_dict(manifest: ConnectorManifest) -> Dict[str, Any]:
    """Convert a ConnectorManifest into Orbit's marketplace connector dict."""
    auth_display = {
        "oauth2": "OAuth 2.0",
        "api_key": "API Key",
        "bearer": "Remote MCP",
        "none": "None",
    }.get(manifest.authentication, "Remote MCP")

    return {
        "id": manifest.id,
        "name": manifest.name,
        "category": manifest.category,
        "description": manifest.description,
        "icon": manifest.icon,
        "badge": manifest.badge or "MCP",
        "authType": auth_display,
        "mcpUrl": manifest.server_url,
        "website": manifest.documentation_url or manifest.server_url,
        "configFields": manifest.required_config,
        "capabilities": manifest.capabilities,
        # Extended manifest fields
        "provider": manifest.provider,
        "source": manifest.source,
        "source_type": manifest.source_type,
        "implementation_type": manifest.implementation_type,
        "server_url": manifest.server_url,
        "documentation_url": manifest.documentation_url,
        "repository_url": manifest.repository_url,
        "transport": manifest.transport,
        "authentication": manifest.authentication,
        "oauth_provider_key": manifest.oauth_provider_key,
        "supports_dynamic_registration": manifest.supports_dynamic_registration,
        "verification_status": manifest.verification_status,
        "support_status": manifest.support_status,
        "last_verified_at": manifest.last_verified_at,
        "verification_notes": manifest.verification_notes,
        "setup_instructions": manifest.setup_instructions,
    }


def get_registry_summary(manifests: Optional[List[ConnectorManifest]] = None) -> Dict[str, int]:
    items = manifests if manifests is not None else load_marketplace_manifests()
    counts: Dict[str, int] = {s: 0 for s in ALL_VERIFICATION_STATUSES}
    for m in items:
        counts[m.verification_status] = counts.get(m.verification_status, 0) + 1
    counts["total"] = len(items)
    return counts
