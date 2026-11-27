"""Regression tests for the AES-256-GCM integration encryption system."""

from __future__ import annotations

import json
import time
from contextlib import contextmanager
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from credential_vault import (
    CredentialReconnectRequiredError,
    CredentialUnrecoverableError,
    CredentialVault,
    CredentialVaultNotConfiguredError,
    generate_encryption_key,
)


class _InMemoryCursor:
    def __init__(self, rows: list[dict[str, Any]] | None = None, rowcount: int = 0):
        self._rows = rows or []
        self.rowcount = rowcount

    def fetchone(self):
        return self._rows[0] if self._rows else None

    def fetchall(self):
        return list(self._rows)


class _InMemoryConnection:
    def __init__(self, db: "_InMemoryDatabase"):
        self.db = db

    def execute(self, query: str, params: tuple[Any, ...] | list[Any] = ()):
        q = " ".join(query.split())
        params_list = list(params)

        if q.startswith("INSERT INTO proactive_provider_accounts"):
            (
                row_id, user_id, provider, provider_account_id, account_name,
                access_token, refresh_token, scopes, metadata,
                enc_ver, enc_kv, cred_status, cred_err, created_at, updated_at
            ) = params_list
            key = (user_id, provider, provider_account_id)
            existing = self.db.provider_accounts.get(key)
            self.db.provider_accounts[key] = {
                "id": existing["id"] if existing else row_id,
                "user_id": user_id,
                "provider": provider,
                "provider_account_id": provider_account_id,
                "account_name": account_name,
                "access_token": access_token,
                "refresh_token": refresh_token,
                "scopes": scopes,
                "metadata": metadata,
                "encryption_version": enc_ver,
                "encryption_key_version": enc_kv,
                "credential_status": "active",
                "credential_error": None,
                "created_at": existing["created_at"] if existing else created_at,
                "updated_at": updated_at,
            }
            return _InMemoryCursor(rowcount=1)

        if q.startswith("SELECT * FROM proactive_provider_accounts WHERE user_id = %s AND provider = %s"):
            user_id, provider = params_list[0], params_list[1]
            acct_id = params_list[2] if len(params_list) > 2 else None
            matches = [
                dict(r) for r in self.db.provider_accounts.values()
                if r["user_id"] == user_id and r["provider"] == provider
                and (acct_id is None or r["provider_account_id"] == acct_id)
            ]
            matches.sort(key=lambda r: r["updated_at"], reverse=True)
            return _InMemoryCursor(matches[:1])

        if q.startswith("SELECT * FROM proactive_provider_accounts WHERE provider = %s"):
            provider = params_list[0]
            return _InMemoryCursor([dict(r) for r in self.db.provider_accounts.values() if r["provider"] == provider])

        if q == "SELECT * FROM proactive_provider_accounts":
            return _InMemoryCursor([dict(r) for r in self.db.provider_accounts.values()])

        if q.startswith("SELECT id, user_id, provider, provider_account_id, access_token, refresh_token, credential_status FROM proactive_provider_accounts"):
            return _InMemoryCursor([dict(r) for r in self.db.provider_accounts.values()])

        if q.startswith("SELECT id, user_id, provider, access_token, refresh_token, credential_status FROM proactive_provider_accounts"):
            return _InMemoryCursor([
                dict(r) for r in self.db.provider_accounts.values()
                if r.get("credential_status") in (None, "active")
            ])

        if q.startswith("SELECT provider, provider_account_id, account_name, scopes, updated_at"):
            user_id = params_list[0]
            matches = [dict(r) for r in self.db.provider_accounts.values() if r["user_id"] == user_id]
            matches.sort(key=lambda r: (r["provider"], r["account_name"]))
            return _InMemoryCursor(matches)

        if q.startswith("UPDATE proactive_provider_accounts SET credential_status = 'reconnect_required'") and "WHERE id = %s" in q:
            reason, row_id = params_list
            for r in self.db.provider_accounts.values():
                if r["id"] == row_id:
                    r["credential_status"] = "reconnect_required"
                    r["encryption_version"] = "legacy-unrecoverable"
                    r["encryption_key_version"] = None
                    r["credential_error"] = reason
                    return _InMemoryCursor(rowcount=1)
            return _InMemoryCursor(rowcount=0)

        if q.startswith("UPDATE proactive_provider_accounts SET credential_status = 'reconnect_required'"):
            enc_ver, reason, user_id, provider, acct_id = params_list
            r = self.db.provider_accounts.get((user_id, provider, acct_id))
            if r:
                r["credential_status"] = "reconnect_required"
                r["encryption_version"] = enc_ver
                r["encryption_key_version"] = None
                r["credential_error"] = reason
                return _InMemoryCursor(rowcount=1)
            return _InMemoryCursor(rowcount=0)

        if q.startswith("UPDATE proactive_provider_accounts SET access_token = %s"):
            new_acc, new_ref, kv, row_id = params_list
            for r in self.db.provider_accounts.values():
                if r["id"] == row_id:
                    r["access_token"] = new_acc
                    r["refresh_token"] = new_ref
                    r["encryption_version"] = "v2"
                    r["encryption_key_version"] = kv
                    return _InMemoryCursor(rowcount=1)
            return _InMemoryCursor(rowcount=0)

        if q.startswith("DELETE FROM proactive_provider_accounts"):
            user_id, provider = params_list[0], params_list[1]
            acct_id = params_list[2] if len(params_list) > 2 else None
            to_del = [
                k for k, r in self.db.provider_accounts.items()
                if r["user_id"] == user_id and r["provider"] == provider
                and (acct_id is None or r["provider_account_id"] == acct_id)
            ]
            for k in to_del:
                del self.db.provider_accounts[k]
            return _InMemoryCursor(rowcount=len(to_del))

        if q.startswith("INSERT INTO proactive_github_hooks"):
            (hook_id, token_hash, user_id, repo, routine_id, gh_hook_id, secret, enc_ver, enc_kv, cred_status, created_at) = params_list
            self.db.github_hooks[hook_id] = {
                "id": hook_id, "token": token_hash, "user_id": user_id,
                "repository": repo, "routine_id": routine_id, "hook_id": gh_hook_id,
                "secret": secret, "encryption_version": enc_ver,
                "encryption_key_version": enc_kv, "credential_status": cred_status,
                "created_at": created_at,
            }
            return _InMemoryCursor(rowcount=1)

        if q.startswith("SELECT * FROM proactive_github_hooks WHERE token = %s OR token = %s"):
            token_hash, raw_token = params_list
            matches = [dict(r) for r in self.db.github_hooks.values() if r["token"] in (token_hash, raw_token)]
            return _InMemoryCursor(matches[:1])

        if q.startswith("SELECT id, user_id, secret, credential_status FROM proactive_github_hooks"):
            if "WHERE credential_status" in q:
                return _InMemoryCursor([
                    dict(r) for r in self.db.github_hooks.values()
                    if r.get("credential_status") in (None, "active")
                ])
            return _InMemoryCursor([dict(r) for r in self.db.github_hooks.values()])

        if q.startswith("UPDATE proactive_github_hooks SET secret = %s"):
            new_sec, kv, row_id = params_list
            if row_id in self.db.github_hooks:
                self.db.github_hooks[row_id]["secret"] = new_sec
                self.db.github_hooks[row_id]["encryption_version"] = "v2"
                self.db.github_hooks[row_id]["encryption_key_version"] = kv
                return _InMemoryCursor(rowcount=1)
            return _InMemoryCursor(rowcount=0)

        if q.startswith("UPDATE proactive_github_hooks SET credential_status = 'reconnect_required'"):
            row_id = params_list[0]
            if row_id in self.db.github_hooks:
                self.db.github_hooks[row_id]["credential_status"] = "reconnect_required"
                self.db.github_hooks[row_id]["encryption_version"] = "legacy-unrecoverable"
                self.db.github_hooks[row_id]["encryption_key_version"] = None
                return _InMemoryCursor(rowcount=1)
            return _InMemoryCursor(rowcount=0)

        raise NotImplementedError(f"Unhandled SQL in mock DB: {q}")


class _InMemoryDatabase:
    def __init__(self):
        self.provider_accounts: dict[tuple[str, str, str], dict[str, Any]] = {}
        self.github_hooks: dict[str, dict[str, Any]] = {}

    @contextmanager
    def connection(self):
        yield _InMemoryConnection(self)


def test_vault_aes256_gcm_roundtrip_and_isolation(monkeypatch):
    key = generate_encryption_key()
    monkeypatch.setenv("INTEGRATION_ENCRYPTION_KEY", key)
    monkeypatch.setenv("INTEGRATION_ENCRYPTION_KEY_VERSION", "k1")

    vault = CredentialVault()
    assert vault.configured is True
    assert vault.key_version == "k1"

    ct = vault.encrypt_text("ya29.secret-token", user_id="user-1", integration="gmail")
    assert ct.startswith("enc:v2:k1:")
    assert "ya29.secret-token" not in ct
    assert vault.decrypt_text(ct, user_id="user-1", integration="gmail") == "ya29.secret-token"

    # Multi-user isolation: ciphertext for user-1 cannot be decrypted for user-2
    with pytest.raises(CredentialUnrecoverableError):
        vault.decrypt_text(ct, user_id="user-2", integration="gmail")

    # Multi-integration isolation: ciphertext for gmail cannot be decrypted for github
    with pytest.raises(CredentialUnrecoverableError):
        vault.decrypt_text(ct, user_id="user-1", integration="github")


def test_vault_fails_closed_when_missing_or_invalid_key(monkeypatch):
    monkeypatch.delenv("INTEGRATION_ENCRYPTION_KEY", raising=False)
    vault = CredentialVault()
    assert vault.configured is False

    with pytest.raises(CredentialVaultNotConfiguredError):
        vault.encrypt_text("secret", user_id="u1", integration="gmail")

    with pytest.raises(CredentialVaultNotConfiguredError):
        vault.decrypt_text("enc:v2:k1:abc", user_id="u1", integration="gmail")

    with pytest.raises(CredentialVaultNotConfiguredError):
        CredentialVault(key="not-a-valid-32-byte-key")


def test_legacy_unrecoverable_ciphertext_cannot_be_decrypted_or_migrated():
    vault = CredentialVault(key=generate_encryption_key())

    for legacy in ("enc:v1:gAAAAABkLegacyCiphertext", "gAAAAABkRawFernetToken"):
        with pytest.raises(CredentialUnrecoverableError):
            vault.decrypt_text(legacy, user_id="u1", integration="gmail")
        info = vault.inspect_value(legacy)
        assert info["encrypted"] is True
        assert info["encryption_version"].startswith("legacy-")


def test_key_rotation_and_restart_persistence(monkeypatch):
    key_v1 = generate_encryption_key()
    key_v2 = generate_encryption_key()

    # Encrypt under k1
    vault_v1 = CredentialVault(key=key_v1, key_version="k1")
    ct_v1 = vault_v1.encrypt_text("gho_initial_token", user_id="u1", integration="github")
    assert ct_v1.startswith("enc:v2:k1:")

    # Simulate restart with k2 as active key and k1 in INTEGRATION_ENCRYPTION_PREVIOUS_KEYS
    monkeypatch.setenv("INTEGRATION_ENCRYPTION_KEY", key_v2)
    monkeypatch.setenv("INTEGRATION_ENCRYPTION_KEY_VERSION", "k2")
    monkeypatch.setenv("INTEGRATION_ENCRYPTION_PREVIOUS_KEYS", f"k1:{key_v1}")

    vault_v2 = CredentialVault()
    assert vault_v2.decrypt_text(ct_v1, user_id="u1", integration="github") == "gho_initial_token"
    assert vault_v2.needs_rotation(ct_v1) is True

    ct_v2 = vault_v2.reencrypt_text(ct_v1, user_id="u1", integration="github")
    assert ct_v2.startswith("enc:v2:k2:")
    assert vault_v2.needs_rotation(ct_v2) is False

    # After retiring k1, k2 still decrypts rotated credentials across restarts
    monkeypatch.delenv("INTEGRATION_ENCRYPTION_PREVIOUS_KEYS", raising=False)
    vault_v2_restarted = CredentialVault()
    assert vault_v2_restarted.decrypt_text(ct_v2, user_id="u1", integration="github") == "gho_initial_token"
    with pytest.raises(CredentialUnrecoverableError):
        vault_v2_restarted.decrypt_text(ct_v1, user_id="u1", integration="github")


def test_provider_store_preserves_and_flags_unrecoverable_accounts_and_reconnect_replaces_token():
    from proactive_integrations import ProviderIntegrationStore

    db = _InMemoryDatabase()
    new_key = generate_encryption_key()

    # Seed legacy unrecoverable records (encrypted with lost Fernet key)
    db.provider_accounts[("user-a", "gmail", "alice@gmail.com")] = {
        "id": "acct_legacy_gmail",
        "user_id": "user-a",
        "provider": "gmail",
        "provider_account_id": "alice@gmail.com",
        "account_name": "alice@gmail.com",
        "access_token": "gAAAAABkLostFernetAccessToken",
        "refresh_token": "gAAAAABkLostFernetRefreshToken",
        "scopes": json.dumps(["https://www.googleapis.com/auth/gmail.readonly"]),
        "metadata": json.dumps({"expires_at": 1700000000}),
        "encryption_version": None,
        "encryption_key_version": None,
        "credential_status": "active",
        "credential_error": None,
        "created_at": "2026-01-01T00:00:00+00:00",
        "updated_at": "2026-01-01T00:00:00+00:00",
    }
    db.github_hooks["ghhook_legacy"] = {
        "id": "ghhook_legacy",
        "token": "deadbeef" * 8,
        "user_id": "user-a",
        "repository": "shikhar-jadav/orbit-backend",
        "routine_id": "rtn_1",
        "hook_id": 99,
        "secret": "gAAAAABkLostFernetWebhookSecret",
        "encryption_version": None,
        "encryption_key_version": None,
        "credential_status": "active",
        "created_at": "2026-01-01T00:00:00+00:00",
    }

    store = ProviderIntegrationStore(db, encryption_key=new_key)

    # Reconciliation marks unrecoverable records without deleting them
    reconciled = store.reconcile_legacy_credentials()
    assert reconciled["accounts_marked_reconnect_required"] == 1
    assert reconciled["hooks_marked_reconnect_required"] == 1
    assert ("user-a", "gmail", "alice@gmail.com") in db.provider_accounts

    # Public account listing succeeds and indicates reconnect_required without exposing tokens
    public = store.public_accounts("user-a")
    assert len(public) == 1
    assert public[0]["reconnect_required"] is True
    assert public[0]["credential_status"] == "reconnect_required"
    assert "reconnected" in (public[0]["reconnect_message"] or "").lower()
    assert "access_token" not in public[0]

    # Live token access raises CredentialReconnectRequiredError
    with pytest.raises(CredentialReconnectRequiredError):
        store.get_account("user-a", "gmail")

    # Background list_accounts does not crash
    assert store.list_accounts("gmail") == []

    # Reconnecting via OAuth replaces the old unrecoverable token with enc:v2
    reconnected = store.save_account(
        user_id="user-a",
        provider="gmail",
        provider_account_id="alice@gmail.com",
        account_name="alice@gmail.com",
        access_token="ya29.fresh-token",
        refresh_token="1//fresh-refresh",
        scopes=["https://www.googleapis.com/auth/gmail.readonly"],
        metadata={"expires_at": int(time.time()) + 3600},
    )
    assert reconnected["access_token"] == "ya29.fresh-token"
    assert reconnected["reconnect_required"] is False
    assert reconnected["credential_status"] == "active"

    stored_row = db.provider_accounts[("user-a", "gmail", "alice@gmail.com")]
    assert stored_row["access_token"].startswith("enc:v2:k1:")
    assert stored_row["encryption_version"] == "v2"
    assert stored_row["encryption_key_version"] == "k1"


def test_provider_store_github_webhook_and_rotation():
    from proactive_integrations import ProviderIntegrationStore, verify_github_signature

    db = _InMemoryDatabase()
    key_v1 = generate_encryption_key()
    store_v1 = ProviderIntegrationStore(db, vault=CredentialVault(key=key_v1, key_version="k1"))

    store_v1.save_account(
        user_id="user-gh",
        provider="github",
        provider_account_id="12345",
        account_name="octocat",
        access_token="gho_valid_token",
    )
    store_v1.create_github_hook(
        token="raw-hook-token",
        user_id="user-gh",
        repository="octocat/hello-world",
        routine_id="rtn_gh_1",
        hook_id=77,
        secret="whsec_test_123",
    )

    hook = store_v1.get_github_hook("raw-hook-token")
    assert hook is not None
    assert hook["secret"] == "whsec_test_123"
    assert verify_github_signature("whsec_test_123", b"{}", "sha256=" + __import__("hmac").new(b"whsec_test_123", b"{}", "sha256").hexdigest())

    # Rotate to k2
    key_v2 = generate_encryption_key()
    store_v2 = ProviderIntegrationStore(
        db,
        vault=CredentialVault(key=key_v2, key_version="k2", previous_keys={"k1": key_v1}),
    )
    rotated = store_v2.rotate_credentials()
    assert rotated["accounts_rotated"] == 1
    assert rotated["hooks_rotated"] == 1

    assert db.provider_accounts[("user-gh", "github", "12345")]["access_token"].startswith("enc:v2:k2:")
    assert store_v2.get_account("user-gh", "github")["access_token"] == "gho_valid_token"
    assert store_v2.get_github_hook("raw-hook-token")["secret"] == "whsec_test_123"

