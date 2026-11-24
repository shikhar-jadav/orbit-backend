import json

import pytest

import marketplace
from chat_store import PostgresChatStore
from credential_vault import CredentialVault
from policy_engine import evaluate_policy, redact_sensitive_data
from policy_engine import parse_action_approval


def test_custom_connector_credentials_are_encrypted_and_masked(postgres_database, monkeypatch):
    key = CredentialVault().configured
    assert key
    store = PostgresChatStore(postgres_database)
    connector = store.create_personal_connector(
        "user-vault-1",
        name="Private MCP",
        mcp_url="https://mcp.example.test",
        headers=[{"key": "Authorization", "value": "Bearer private-header-token"}],
        advanced={"oauth_client_id": "public-client-id", "oauth_client_secret": "private-oauth-secret"},
    )

    with postgres_database.connection() as connection:
        row = connection.execute(
            "SELECT headers_json, advanced_json FROM user_personal_connectors WHERE id = %s",
            (connector["id"],),
        ).fetchone()

    stored = row["headers_json"] + row["advanced_json"]
    assert "private-header-token" not in stored
    assert "private-oauth-secret" not in stored
    assert "enc:v1:" in stored

    public = store.get_personal_connector("user-vault-1", connector["id"])
    assert public["headers"][0]["has_value"] is True
    assert "private-header-token" not in json.dumps(public)
    assert "private-oauth-secret" not in json.dumps(public)

    private = store.get_personal_connector(
        "user-vault-1", connector["id"], include_secrets=True
    )
    assert private["headers"][0]["value"] == "Bearer private-header-token"
    assert private["advanced"]["oauth_client_secret"] == "private-oauth-secret"


def test_legacy_custom_connector_credentials_are_migrated(postgres_database):
    store = PostgresChatStore(postgres_database)
    connector = store.create_personal_connector(
        "user-vault-2",
        name="Legacy MCP",
        mcp_url="https://legacy.example.test",
    )
    with postgres_database.connection() as connection:
        connection.execute(
            """
            UPDATE user_personal_connectors
            SET headers_json = %s, advanced_json = %s
            WHERE id = %s AND user_id = %s
            """,
            (
                json.dumps([{"key": "Authorization", "value": "legacy-header-secret"}]),
                json.dumps({"oauth_client_secret": "legacy-oauth-secret"}),
                connector["id"],
                "user-vault-2",
            ),
        )

    result = store.migrate_legacy_connector_credentials()
    assert result["personal_connectors_migrated"] == 1

    with postgres_database.connection() as connection:
        row = connection.execute(
            "SELECT headers_json, advanced_json FROM user_personal_connectors WHERE id = %s",
            (connector["id"],),
        ).fetchone()
    stored = row["headers_json"] + row["advanced_json"]
    assert "legacy-header-secret" not in stored
    assert "legacy-oauth-secret" not in stored

    recovered = store.get_personal_connector(
        "user-vault-2", connector["id"], include_secrets=True
    )
    assert recovered["headers"][0]["value"] == "legacy-header-secret"
    assert recovered["advanced"]["oauth_client_secret"] == "legacy-oauth-secret"


def test_marketplace_password_fields_are_encrypted(postgres_database):
    store = PostgresChatStore(postgres_database)
    connector = {
        "id": "vault-test-connector",
        "name": "Vault test connector",
        "configFields": [
            {"key": "api_token", "type": "password", "required": True},
            {"key": "workspace", "type": "text", "required": False},
        ],
    }
    marketplace.register_connector(connector)
    try:
        marketplace.save_connector_configuration(
            store,
            "vault-test-connector",
            True,
            {"api_token": "marketplace-private-token", "workspace": "demo"},
            user_id="user-vault-3",
        )
        raw = store.get_setting(
            "connector_vault-test-connector_config", user_id="user-vault-3"
        )
        assert "marketplace-private-token" not in raw
        assert "enc:v1:" in raw

        status = marketplace.list_connectors_with_status(
            store, user_id="user-vault-3"
        )
        item = next(x for x in status if x["id"] == "vault-test-connector")
        assert item["is_configured"] is True
        assert "marketplace-private-token" not in json.dumps(item)
    finally:
        marketplace.AVAILABLE_CONNECTORS[:] = [
            item for item in marketplace.AVAILABLE_CONNECTORS
            if item["id"] != "vault-test-connector"
        ]


def test_credential_vault_fails_closed_without_a_key(monkeypatch):
    monkeypatch.delenv("ORBIT_CREDENTIAL_ENCRYPTION_KEY", raising=False)
    monkeypatch.delenv("INTEGRATION_TOKEN_ENCRYPTION_KEY", raising=False)
    vault = CredentialVault()
    with pytest.raises(RuntimeError, match="(?i)encryption[_ ]key"):
        vault.encrypt_text("new-secret")
    with pytest.raises(RuntimeError, match="(?i)encryption[_ ]key"):
        vault.decrypt_text("legacy-plaintext-secret")


def test_critical_actions_cannot_be_preapproved_or_allowlisted():
    decision = evaluate_policy(
        "delete_customer",
        {"customer_id": "cus_123"},
        connector_info={
            "id": "billing",
            "name": "Billing",
            "raw_name": "delete_customer",
            "tool_permissions": {"delete_customer": "allow"},
        },
        approved_tools={"delete_customer"},
    )
    assert decision.risk_level == "critical"
    assert decision.requires_approval is True
    assert decision.options[0]["title"] == "Approve once (Recommended)"


def test_memory_deletion_requires_fresh_approval():
    decision = evaluate_policy(
        "manage_memory",
        {"action": "delete", "key": "old-rule"},
        approved_tools={"manage_memory"},
    )
    assert decision.risk_level == "critical"
    assert decision.requires_approval is True


def test_sensitive_values_are_redacted_from_approval_payloads():
    safe = redact_sensitive_data(
        {
            "to": "customer@example.test",
            "api_token": "private-token",
            "note": "Forward with Bearer private-bearer only when explicitly requested",
            "headers": [{"key": "X-Secret", "value": "header-secret"}],
        }
    )
    rendered = json.dumps(safe)
    assert "private-token" not in rendered
    assert "private-bearer" not in rendered
    assert "header-secret" not in rendered
    assert "customer@example.test" in rendered


def test_approval_parser_fails_closed_and_never_remembers_critical_actions():
    assert parse_action_approval({"action": "answer", "selected_titles": []}, "high") == (False, False)
    assert parse_action_approval({"action": "decline"}, "high") == (False, False)
    assert parse_action_approval(
        {"action": "approve", "always_allow": True}, "critical"
    ) == (True, False)
    assert parse_action_approval(
        {"action": "approve", "always_allow": True}, "high"
    ) == (True, True)


def test_approval_update_is_owner_scoped_and_only_pending_can_resolve(postgres_database):
    store = PostgresChatStore(postgres_database)
    store.create_approval(
        approval_id="appr-owner-scope",
        user_id="owner-1",
        thread_id=None,
        tool_name="send_email",
        action_summary="Send email",
        risk_level="high",
    )
    assert store.update_approval(
        "appr-owner-scope", status="approved", user_id="owner-2"
    ) is None
    assert store.get_approval("appr-owner-scope")["status"] == "pending"

    resolved = store.update_approval(
        "appr-owner-scope",
        status="approved",
        decision={"action": "approve"},
        user_id="owner-1",
    )
    assert resolved["status"] == "approved"
    assert store.update_approval(
        "appr-owner-scope", status="rejected", user_id="owner-1"
    ) is None
