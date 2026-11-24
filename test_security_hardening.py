import json
from uuid import uuid4

import pytest

import marketplace
from chat_store import PostgresChatStore
from credential_vault import CredentialVault
from policy_engine import evaluate_policy, redact_sensitive_data
from policy_engine import parse_action_approval





def _make_vault_test_user(store, prefix: str) -> str:
    email = f"{prefix}-{uuid4().hex}@example.test"
    user, _ = store.create_user(
        email=email,
        display_name="Vault Test User",
        password_hash="test-password-hash",
    )
    return user["user_id"]

def test_custom_connector_credentials_are_encrypted_and_masked(postgres_database, monkeypatch):
    key = CredentialVault().configured
    assert key
    store = PostgresChatStore(postgres_database)
    user_id = _make_vault_test_user(store, "user-vault-1")
    connector = store.create_personal_connector(
        user_id,
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

    public = store.get_personal_connector(user_id, connector["id"])
    assert public["headers"][0]["has_value"] is True
    assert "private-header-token" not in json.dumps(public)
    assert "private-oauth-secret" not in json.dumps(public)

    private = store.get_personal_connector(
        user_id, connector["id"], include_secrets=True
    )
    assert private["headers"][0]["value"] == "Bearer private-header-token"
    assert private["advanced"]["oauth_client_secret"] == "private-oauth-secret"


def test_legacy_custom_connector_credentials_are_migrated(postgres_database):
    store = PostgresChatStore(postgres_database)
    user_id = _make_vault_test_user(store, "user-vault-2")
    connector = store.create_personal_connector(
        user_id,
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
                user_id,
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
        user_id, connector["id"], include_secrets=True
    )
    assert recovered["headers"][0]["value"] == "legacy-header-secret"
    assert recovered["advanced"]["oauth_client_secret"] == "legacy-oauth-secret"


def test_marketplace_password_fields_are_encrypted(postgres_database):
    store = PostgresChatStore(postgres_database)
    user_id = _make_vault_test_user(store, "user-vault-3")
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
            user_id=user_id,
        )
        raw = store.get_setting(
            "connector_vault-test-connector_config", user_id=user_id
        )
        assert "marketplace-private-token" not in raw
        assert "enc:v1:" in raw

        status = marketplace.list_connectors_with_status(
            store, user_id=user_id
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
    monkeypatch.delenv("INTEGRATION_ENCRYPTION_KEY", raising=False)
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


def test_approval_parser_fails_closed_and_separates_once_session_and_persistent():
    assert parse_action_approval({"action": "answer", "selected_titles": []}, "high") == (False, False, False)
    assert parse_action_approval({"action": "decline"}, "high") == (False, False, False)
    assert parse_action_approval(
        {"action": "approve", "selected_titles": ["Allow once (Recommended)"]}, "high"
    ) == (True, False, False)
    assert parse_action_approval(
        {"action": "approve", "selected_titles": ["Approve for this session"]}, "high"
    ) == (True, False, True)
    assert parse_action_approval(
        {"action": "approve", "selected_titles": ["Always allow this tool"]}, "high"
    ) == (True, True, False)
    assert parse_action_approval(
        {"action": "approve", "always_allow": True}, "critical"
    ) == (True, False, False)


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

def test_pending_approvals_are_interrupted_after_server_restart(postgres_database):
    store = PostgresChatStore(postgres_database)
    approval_id = "appr-interrupted-after-restart"
    store.create_approval(
        approval_id=approval_id,
        user_id="restart-user",
        thread_id=None,
        tool_name="send_email",
        action_summary="Send an email",
        risk_level="high",
    )

    assert store.interrupt_pending_approvals() == 1
    approval = store.get_approval(approval_id)
    assert approval["status"] == "interrupted"
    assert "restarted" in approval["decision"]["reason"].lower()
    assert store.update_approval(
        approval_id, status="approved", user_id="restart-user"
    ) is None

def test_github_webhook_callback_tokens_are_hashed_and_not_listed(postgres_database):
    from credential_vault import generate_encryption_key
    from proactive_integrations import ProviderIntegrationStore

    key = generate_encryption_key()
    store = ProviderIntegrationStore(postgres_database, encryption_key=key)
    callback_token = "private-github-callback-token"
    signing_secret = "private-github-signing-secret"

    store.create_github_hook(
        token=callback_token,
        user_id="github-hook-owner",
        repository="owner/repository",
        routine_id="routine-1",
        hook_id=42,
        secret=signing_secret,
    )

    with postgres_database.connection() as connection:
        stored = connection.execute(
            "SELECT token, secret FROM proactive_github_hooks WHERE hook_id = 42"
        ).fetchone()
    assert stored["token"] != callback_token
    assert callback_token not in stored["token"]
    assert signing_secret not in stored["secret"]

    hook = store.get_github_hook(callback_token)
    assert hook["secret"] == signing_secret
    assert hook["hook_id"] == 42

    listed = store.list_github_hooks_for_user("github-hook-owner")
    assert listed and "token" not in listed[0] and "secret" not in listed[0]


def test_legacy_github_callback_tokens_are_hashed(postgres_database):
    from credential_vault import generate_encryption_key
    from proactive_integrations import ProviderIntegrationStore

    store = ProviderIntegrationStore(
        postgres_database, encryption_key=generate_encryption_key()
    )
    store.create_github_hook(
        token="temporary-callback-token",
        user_id="legacy-github-owner",
        repository="owner/repository",
        routine_id="routine-legacy",
        hook_id=43,
        secret="legacy-signing-secret",
    )
    with postgres_database.connection() as connection:
        row = connection.execute(
            "SELECT id FROM proactive_github_hooks WHERE hook_id = 43"
        ).fetchone()
        connection.execute(
            "UPDATE proactive_github_hooks SET token = %s WHERE id = %s",
            ("legacy-raw-callback-token", row["id"]),
        )

    assert store.migrate_github_hook_tokens() == 1
    assert store.get_github_hook("legacy-raw-callback-token")["hook_id"] == 43
