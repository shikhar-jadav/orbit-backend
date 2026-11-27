"""Tests for connector_registry, oauth_providers, and marketplace catalogue integrity."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import connector_registry  # noqa: E402
import oauth_providers  # noqa: E402


def test_loads_all_950_marketplace_entries():
    manifests = connector_registry.load_marketplace_manifests()
    assert len(manifests) == 950
    ids = [m.id for m in manifests]
    assert len(ids) == len(set(ids)), "Duplicate connector IDs found"
    removed_ids = {
        "microsoft-365",
        "pubmed",
        "clinical-trials",
        "biorxiv",
        "claude-academy",
        "economic-index",
        "npi-registry",
        "icd-10-codes",
        "cms-coverage",
        "military-and-veteran-family-assistant",
        "pdf-viewer",
        "threejs-3d-viewer",
        "play-sheet-music",
    }
    assert not (set(ids) & removed_ids)


def test_claude_and_example_endpoints_marked_unsupported():
    claude_manifest = connector_registry.classify_marketplace_entry({
        "id": "test-claude-endpoint",
        "name": "Test Claude Endpoint",
        "mcpUrl": "https://microsoft365.mcp.claude.com/mcp",
    })
    assert claude_manifest.verification_status == connector_registry.STATUS_UNSUPPORTED
    assert claude_manifest.support_status == "unsupported"

    example_manifest = connector_registry.classify_marketplace_entry({
        "id": "test-example-endpoint",
        "name": "Test Example Endpoint",
        "mcpUrl": "https://example-server.modelcontextprotocol.io/pdf/mcp",
    })
    assert example_manifest.verification_status == connector_registry.STATUS_UNSUPPORTED


def test_google_connectors_require_setup_when_env_missing(monkeypatch):
    for var in (
        "ORBIT_GOOGLE_OAUTH_CLIENT_ID",
        "ORBIT_GOOGLE_OAUTH_CLIENT_SECRET",
        "GOOGLE_OAUTH_CLIENT_ID",
        "GOOGLE_OAUTH_CLIENT_SECRET",
        "GOOGLE_CLIENT_ID",
        "GOOGLE_CLIENT_SECRET",
    ):
        monkeypatch.delenv(var, raising=False)

    manifests = connector_registry.load_marketplace_manifests()
    by_id = {m.id: m for m in manifests}
    assert by_id["google-drive"].verification_status == connector_registry.STATUS_SETUP_REQUIRED
    assert by_id["google-drive"].setup_instructions


def test_google_connectors_become_auth_required_when_env_configured(monkeypatch):
    monkeypatch.setenv("GOOGLE_OAUTH_CLIENT_ID", "test-google-client-id")
    monkeypatch.setenv("GOOGLE_OAUTH_CLIENT_SECRET", "test-google-client-secret")

    m = connector_registry.classify_marketplace_entry({
        "id": "google-drive",
        "name": "Google Drive",
        "description": "Search, read, and upload files",
        "mcpUrl": "https://drivemcp.googleapis.com/mcp/v1",
    })
    assert m.verification_status == connector_registry.STATUS_AUTH_REQUIRED
    assert m.oauth_provider_key == "google"


def test_oauth_credential_resolution_never_fabricates(monkeypatch):
    for var in ("ORBIT_SLACK_OAUTH_CLIENT_ID", "ORBIT_SLACK_OAUTH_CLIENT_SECRET", "SLACK_CLIENT_ID", "SLACK_CLIENT_SECRET"):
        monkeypatch.delenv(var, raising=False)

    res = oauth_providers.resolve_oauth_client_credentials(
        "slack",
        connector_advanced={},
        oauth_meta={"registration_endpoint": "https://mcp.slack.com/register"},
    )
    # Slack requires a static app; dynamic registration must NOT be used for Slack
    assert res["client_id"] is None
    assert res["setup_required"] is True
    assert "Slack" in (res["setup_instructions"] or "")

    # Notion supports dynamic registration
    res_notion = oauth_providers.resolve_oauth_client_credentials(
        "notion",
        connector_advanced={},
        oauth_meta={"registration_endpoint": "https://mcp.notion.com/register"},
    )
    assert res_notion["can_dynamic_register"] is True
    assert res_notion["setup_required"] is False
