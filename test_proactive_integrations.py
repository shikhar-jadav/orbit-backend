import hashlib
import hmac
import base64
import time

from cryptography.fernet import Fernet

from proactive_integrations import (
    ProviderIntegrationStore,
    normalize_github_event,
    normalize_gmail_message,
    normalize_slack_event,
    verify_github_signature,
    verify_slack_signature,
)


def test_provider_account_secrets_are_not_returned_by_public_listing(postgres_database):
    key = Fernet.generate_key().decode("ascii")
    store = ProviderIntegrationStore(postgres_database, encryption_key=key)
    store.save_account(
        user_id="user-1",
        provider="github",
        provider_account_id="42",
        account_name="octocat",
        access_token="gho_private",
        scopes=["admin:repo_hook"],
    )

    account = store.get_account("user-1", "github")
    public = store.public_accounts("user-1")

    assert account["access_token"] == "gho_private"
    assert public[0]["account_name"] == "octocat"
    assert "access_token" not in public[0]
    assert "refresh_token" not in public[0]


def test_provider_account_storage_requires_a_key(postgres_database):
    store = ProviderIntegrationStore(postgres_database)

    try:
        store.save_account(
            user_id="user-1",
            provider="github",
            provider_account_id="42",
            account_name="octocat",
            access_token="gho_private",
        )
    except RuntimeError as exc:
        assert "INTEGRATION_TOKEN_ENCRYPTION_KEY" in str(exc)
    else:
        raise AssertionError("Provider token was accepted without encryption configured.")


def test_provider_tokens_and_webhook_secrets_are_encrypted_at_rest(postgres_database):
    key = Fernet.generate_key().decode("ascii")
    store = ProviderIntegrationStore(postgres_database, encryption_key=key)
    access_token = "provider-access-sentinel"
    webhook_secret = "github-webhook-sentinel"

    store.save_account(
        user_id="user-1",
        provider="github",
        provider_account_id="42",
        account_name="octocat",
        access_token=access_token,
    )
    store.create_github_hook(
        token="hook-token",
        user_id="user-1",
        repository="octocat/repo",
        routine_id="routine-1",
        hook_id=1,
        secret=webhook_secret,
    )

    with postgres_database.connection() as connection:
        stored_access_token = connection.execute(
            "SELECT access_token FROM proactive_provider_accounts"
        ).fetchone()["access_token"]
        stored_webhook_secret = connection.execute(
            "SELECT secret FROM proactive_github_hooks"
        ).fetchone()["secret"]
    assert access_token not in stored_access_token
    assert webhook_secret not in stored_webhook_secret
    assert store.get_account("user-1", "github")["access_token"] == access_token
    assert store.get_github_hook("hook-token")["secret"] == webhook_secret


def test_provider_store_rejects_an_invalid_encryption_key(postgres_database):
    try:
        ProviderIntegrationStore(postgres_database, encryption_key="invalid")
    except (ValueError, TypeError):
        pass
    else:
        raise AssertionError("Invalid Fernet key was accepted.")


def test_github_signature_verifies_raw_body():
    body = b'{"action":"opened"}'
    secret = "hook-secret"
    signature = "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()

    assert verify_github_signature(secret, body, signature)
    assert not verify_github_signature(secret, body + b" ", signature)
    assert not verify_github_signature(secret, body, "sha1=invalid")


def test_slack_signature_checks_timestamp_and_body():
    body = b'{"type":"event_callback"}'
    secret = "slack-signing-secret"
    timestamp = str(int(time.time()))
    base = f"v0:{timestamp}:".encode() + body
    signature = "v0=" + hmac.new(secret.encode(), base, hashlib.sha256).hexdigest()

    assert verify_slack_signature(secret, body, timestamp, signature)
    assert not verify_slack_signature(secret, body + b" ", timestamp, signature)
    assert not verify_slack_signature(secret, body, str(int(time.time()) - 600), signature)


def test_github_event_normalization_preserves_target_and_labels():
    event = normalize_github_event(
        {
            "action": "opened",
            "repository": {"full_name": "octo/repo"},
            "issue": {"number": 7, "labels": [{"name": "bug"}]},
        },
        "issues",
        "delivery-1",
        "user-1",
    )

    assert event["source"] == "github"
    assert event["event_type"] == "issues.opened"
    assert event["tenant_id"] == "user-1"
    assert event["payload"]["repository"] == "octo/repo"
    assert event["payload"]["labels"] == ["bug"]
    assert event["metadata"]["provider_event_id"] == "delivery-1"


def test_slack_event_normalization_maps_channel_and_mention():
    event = normalize_slack_event(
        {
            "type": "app_mention",
            "channel": "C123",
            "text": "<@U123> help with this",
            "user": "U456",
            "ts": "123.456",
        },
        "T123",
        "user-1",
    )

    assert event["source"] == "slack"
    assert event["event_type"] == "app_mention"
    assert event["payload"]["channel_id"] == "C123"
    assert event["payload"]["text"] == "<@U123> help with this"
    assert event["tenant_id"] == "user-1"


def test_gmail_watch_cursor_is_monotonic_and_expiration_can_be_renewed(postgres_database):
    store = ProviderIntegrationStore(
        postgres_database,
        encryption_key=Fernet.generate_key().decode("ascii"),
    )
    store.save_gmail_watch_state(
        user_id="user-1",
        email_address="User@Example.com",
        history_id="100",
        expiration="2025-01-01T00:00:00+00:00",
    )

    store.save_gmail_watch_state(
        user_id="user-1",
        email_address="user@example.com",
        history_id="99",
        expiration="2025-01-02T00:00:00+00:00",
    )
    assert store.get_gmail_watch_state("user-1", "USER@example.com")["history_id"] == "100"

    assert store.update_gmail_watch_expiration(
        "user-1", "user@example.com", "2025-01-03T00:00:00+00:00"
    )
    assert (
        store.get_gmail_watch_state("user-1", "user@example.com")["expiration"]
        == "2025-01-03T00:00:00+00:00"
    )


def test_gmail_message_normalization_extracts_headers_and_body():
    body = base64.urlsafe_b64encode(b"Hello Orbit").decode("ascii").rstrip("=")
    event = normalize_gmail_message(
        {
            "id": "message-7",
            "threadId": "thread-3",
            "internalDate": "1735689600000",
            "snippet": "Hello",
            "labelIds": ["INBOX"],
            "payload": {
                "headers": [
                    {"name": "From", "value": "sender@example.com"},
                    {"name": "Subject", "value": "A test"},
                ],
                "parts": [{"mimeType": "text/plain", "body": {"data": body}}],
            },
        },
        "user@example.com",
        "user-1",
    )

    assert event["source"] == "gmail"
    assert event["event_type"] == "email.received"
    assert event["tenant_id"] == "user-1"
    assert event["payload"]["from"] == "sender@example.com"
    assert event["payload"]["subject"] == "A test"
    assert event["payload"]["body"] == "Hello Orbit"
    assert event["metadata"]["provider_event_id"] == "message-7"
