"""Provider account storage and webhook helpers for event-driven routines."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import re
import threading
from contextlib import contextmanager
from typing import Any, Optional
from uuid import uuid4

import httpx
from credential_vault import (
    CredentialReconnectRequiredError,
    CredentialUnrecoverableError,
    CredentialVault,
    CredentialVaultNotConfiguredError,
)
from database import PostgresDatabase


RECONNECT_MESSAGE_TEMPLATE = (
    "Your {provider} connection uses an outdated encryption key and needs to be reconnected."
)


class ProviderIntegrationStore:
    """PostgreSQL persistence for provider accounts and webhook bindings."""

    def __init__(
        self,
        database: PostgresDatabase,
        encryption_key: Optional[str] = None,
        vault: Optional[CredentialVault] = None,
    ):
        self.database = database
        self._lock = threading.Lock()
        self._vault = vault if vault is not None else CredentialVault(key=encryption_key)

    @property
    def token_storage_configured(self) -> bool:
        return self._vault.configured

    @property
    def vault(self) -> CredentialVault:
        return self._vault

    @contextmanager
    def _connection(self):
        with self.database.connection() as conn:
            yield conn

    def save_account(
        self,
        *,
        user_id: str,
        provider: str,
        provider_account_id: str,
        account_name: str,
        access_token: str,
        refresh_token: Optional[str] = None,
        scopes: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        from datetime import datetime, timezone

        if not access_token.strip():
            raise ValueError("Provider access token cannot be empty.")
        encrypted_access_token = self._encrypt(
            access_token, user_id=user_id, integration=provider
        )
        encrypted_refresh_token = (
            self._encrypt(refresh_token, user_id=user_id, integration=provider)
            if refresh_token
            else None
        )
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with self._lock, self._connection() as conn:
            conn.execute(
                """
                INSERT INTO proactive_provider_accounts (
                    id, user_id, provider, provider_account_id, account_name,
                    access_token, refresh_token, scopes, metadata,
                    encryption_version, encryption_key_version,
                    credential_status, credential_error,
                    created_at, updated_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT(user_id, provider, provider_account_id) DO UPDATE SET
                    account_name = excluded.account_name,
                    access_token = excluded.access_token,
                    refresh_token = excluded.refresh_token,
                    scopes = excluded.scopes,
                    metadata = excluded.metadata,
                    encryption_version = excluded.encryption_version,
                    encryption_key_version = excluded.encryption_key_version,
                    credential_status = 'active',
                    credential_error = NULL,
                    updated_at = excluded.updated_at
                """,
                (
                    f"acct_{uuid4().hex}",
                    user_id,
                    provider,
                    provider_account_id,
                    account_name,
                    encrypted_access_token,
                    encrypted_refresh_token,
                    json.dumps(scopes or []),
                    json.dumps(metadata or {}),
                    "v2",
                    self._vault.key_version,
                    "active",
                    None,
                    now,
                    now,
                ),
            )
        return self.get_account(user_id, provider, provider_account_id) or {}

    def _mark_account_reconnect_required(
        self,
        user_id: str,
        provider: str,
        provider_account_id: str,
        *,
        encryption_version: str = "legacy-unrecoverable",
        reason: Optional[str] = None,
    ) -> None:
        message = reason or RECONNECT_MESSAGE_TEMPLATE.format(provider=provider.title())
        with self._lock, self._connection() as conn:
            conn.execute(
                """
                UPDATE proactive_provider_accounts
                SET credential_status = 'reconnect_required',
                    encryption_version = %s,
                    encryption_key_version = NULL,
                    credential_error = %s
                WHERE user_id = %s AND provider = %s AND provider_account_id = %s
                """,
                (encryption_version, message, user_id, provider, provider_account_id),
            )

    def _unrecoverable_account_stub(self, row: dict[str, Any]) -> dict[str, Any]:
        account = dict(row)
        provider = str(account.get("provider") or "integration")
        account["access_token"] = None
        account["refresh_token"] = None
        account["scopes"] = json.loads(account.get("scopes") or "[]")
        account["metadata"] = json.loads(account.get("metadata") or "{}")
        account["credential_status"] = "reconnect_required"
        account["reconnect_required"] = True
        account["reconnect_message"] = (
            account.get("credential_error")
            or RECONNECT_MESSAGE_TEMPLATE.format(provider=provider.title())
        )
        return account

    def get_account(
        self,
        user_id: str,
        provider: str,
        provider_account_id: Optional[str] = None,
        *,
        allow_unrecoverable: bool = False,
    ) -> Optional[dict[str, Any]]:
        query = """
            SELECT * FROM proactive_provider_accounts
            WHERE user_id = %s AND provider = %s
        """
        params: list[Any] = [user_id, provider]
        if provider_account_id:
            query += " AND provider_account_id = %s"
            params.append(provider_account_id)
        query += " ORDER BY updated_at DESC LIMIT 1"
        with self._connection() as conn:
            row = conn.execute(query, params).fetchone()
        if not row:
            return None

        if row.get("credential_status") == "reconnect_required":
            if allow_unrecoverable:
                return self._unrecoverable_account_stub(row)
            raise CredentialReconnectRequiredError(
                row.get("credential_error")
                or RECONNECT_MESSAGE_TEMPLATE.format(provider=provider.title())
            )

        try:
            return self._decode_account(row)
        except CredentialUnrecoverableError as exc:
            inspected = self._vault.inspect_value(row.get("access_token"))
            enc_ver = inspected.get("encryption_version") or "legacy-unrecoverable"
            if enc_ver in {"legacy-v1", "legacy-fernet"}:
                enc_ver = "legacy-unrecoverable"
            reason = RECONNECT_MESSAGE_TEMPLATE.format(provider=provider.title())
            self._mark_account_reconnect_required(
                row["user_id"],
                row["provider"],
                row["provider_account_id"],
                encryption_version=enc_ver,
                reason=reason,
            )
            if allow_unrecoverable:
                stub = self._unrecoverable_account_stub(row)
                stub["encryption_version"] = enc_ver
                stub["reconnect_message"] = reason
                return stub
            raise CredentialReconnectRequiredError(reason) from exc

    def list_accounts(
        self,
        provider: Optional[str] = None,
        *,
        include_unrecoverable: bool = False,
    ) -> list[dict[str, Any]]:
        query = "SELECT * FROM proactive_provider_accounts"
        params: tuple[Any, ...] = ()
        if provider:
            query += " WHERE provider = %s"
            params = (provider,)
        with self._connection() as conn:
            rows = conn.execute(query, params).fetchall()

        accounts: list[dict[str, Any]] = []
        for row in rows:
            if row.get("credential_status") == "reconnect_required":
                if include_unrecoverable:
                    accounts.append(self._unrecoverable_account_stub(row))
                continue
            try:
                accounts.append(self._decode_account(row))
            except CredentialUnrecoverableError:
                inspected = self._vault.inspect_value(row.get("access_token"))
                enc_ver = inspected.get("encryption_version") or "legacy-unrecoverable"
                if enc_ver in {"legacy-v1", "legacy-fernet"}:
                    enc_ver = "legacy-unrecoverable"
                reason = RECONNECT_MESSAGE_TEMPLATE.format(
                    provider=str(row.get("provider") or "integration").title()
                )
                self._mark_account_reconnect_required(
                    row["user_id"],
                    row["provider"],
                    row["provider_account_id"],
                    encryption_version=enc_ver,
                    reason=reason,
                )
                if include_unrecoverable:
                    stub = self._unrecoverable_account_stub(row)
                    stub["encryption_version"] = enc_ver
                    stub["reconnect_message"] = reason
                    accounts.append(stub)
        return accounts

    def _decode_account(self, row: dict[str, Any]) -> dict[str, Any]:
        account = dict(row)
        user_id = str(account["user_id"])
        provider = str(account["provider"])
        account["access_token"] = self._decrypt(
            account["access_token"], user_id=user_id, integration=provider
        )
        if account.get("refresh_token"):
            account["refresh_token"] = self._decrypt(
                account["refresh_token"], user_id=user_id, integration=provider
            )
        account["scopes"] = json.loads(account.get("scopes") or "[]")
        account["metadata"] = json.loads(account.get("metadata") or "{}")
        account["credential_status"] = account.get("credential_status") or "active"
        account["reconnect_required"] = account["credential_status"] == "reconnect_required"
        return account

    def _encrypt(self, value: str, *, user_id: str, integration: str) -> str:
        if not self._vault.configured:
            raise CredentialVaultNotConfiguredError(
                "Set INTEGRATION_ENCRYPTION_KEY before connecting provider accounts."
            )
        return self._vault.encrypt_text(value, user_id=user_id, integration=integration)

    def _decrypt(self, value: str, *, user_id: str, integration: str) -> str:
        if not self._vault.configured:
            raise CredentialVaultNotConfiguredError(
                "Set INTEGRATION_ENCRYPTION_KEY to use connected provider accounts."
            )
        return self._vault.decrypt_text(value, user_id=user_id, integration=integration)

    def public_accounts(self, user_id: str) -> list[dict[str, Any]]:
        with self._connection() as conn:
            rows = conn.execute(
                """
                SELECT provider, provider_account_id, account_name, scopes, updated_at,
                       encryption_version, encryption_key_version,
                       credential_status, credential_error
                FROM proactive_provider_accounts WHERE user_id = %s
                ORDER BY provider, account_name
                """,
                (user_id,),
            ).fetchall()
        results: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            status = item.get("credential_status") or "active"
            reconnect_required = status == "reconnect_required"
            results.append(
                {
                    "provider": item["provider"],
                    "provider_account_id": item["provider_account_id"],
                    "account_name": item["account_name"],
                    "scopes": json.loads(item["scopes"] or "[]"),
                    "updated_at": item["updated_at"],
                    "encryption_version": item.get("encryption_version"),
                    "encryption_key_version": item.get("encryption_key_version"),
                    "credential_status": status,
                    "reconnect_required": reconnect_required,
                    "reconnect_message": (
                        item.get("credential_error")
                        or RECONNECT_MESSAGE_TEMPLATE.format(
                            provider=str(item["provider"]).title()
                        )
                    )
                    if reconnect_required
                    else None,
                }
            )
        return results

    def delete_account(
        self,
        user_id: str,
        provider: str,
        provider_account_id: Optional[str] = None,
    ) -> bool:
        query = "DELETE FROM proactive_provider_accounts WHERE user_id = %s AND provider = %s"
        params: list[Any] = [user_id, provider]
        if provider_account_id:
            query += " AND provider_account_id = %s"
            params.append(provider_account_id)
        with self._lock, self._connection() as conn:
            cur = conn.execute(query, params)
            return cur.rowcount > 0

    def get_gmail_watch_state(
        self, user_id: str, email_address: str
    ) -> Optional[dict[str, Any]]:
        with self._connection() as conn:
            row = conn.execute(
                """
                SELECT user_id, email_address, history_id, expiration, updated_at
                FROM proactive_gmail_watch_state
                WHERE user_id = %s AND email_address = %s
                """,
                (user_id, email_address.casefold()),
            ).fetchone()
        return dict(row) if row else None

    def save_gmail_watch_state(
        self,
        *,
        user_id: str,
        email_address: str,
        history_id: str,
        expiration: Optional[str],
    ) -> None:
        from datetime import datetime, timezone

        with self._lock, self._connection() as conn:
            conn.execute(
                """
                INSERT INTO proactive_gmail_watch_state
                    (user_id, email_address, history_id, expiration, updated_at)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT(user_id, email_address) DO UPDATE SET
                    history_id = CASE
                        WHEN length(excluded.history_id)
                             > length(proactive_gmail_watch_state.history_id)
                          OR (
                            length(excluded.history_id)
                                = length(proactive_gmail_watch_state.history_id)
                            AND excluded.history_id
                                > proactive_gmail_watch_state.history_id
                          )
                        THEN excluded.history_id
                        ELSE proactive_gmail_watch_state.history_id
                    END,
                    expiration = excluded.expiration,
                    updated_at = excluded.updated_at
                """,
                (
                    user_id,
                    email_address.casefold(),
                    history_id,
                    expiration,
                    datetime.now(timezone.utc).isoformat(timespec="seconds"),
                ),
            )

    def update_gmail_watch_expiration(
        self, user_id: str, email_address: str, expiration: str
    ) -> bool:
        from datetime import datetime, timezone

        with self._lock, self._connection() as conn:
            cur = conn.execute(
                """
                UPDATE proactive_gmail_watch_state
                SET expiration = %s, updated_at = %s
                WHERE user_id = %s AND email_address = %s
                """,
                (
                    expiration,
                    datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    user_id,
                    email_address.casefold(),
                ),
            )
            return cur.rowcount > 0

    def delete_gmail_watch_state(self, user_id: str, email_address: str) -> bool:
        with self._lock, self._connection() as conn:
            cur = conn.execute(
                "DELETE FROM proactive_gmail_watch_state WHERE user_id = %s AND email_address = %s",
                (user_id, email_address.casefold()),
            )
            return cur.rowcount > 0

    def create_github_hook(
        self,
        *,
        token: str,
        user_id: str,
        repository: str,
        routine_id: str,
        hook_id: Optional[int],
        secret: str,
    ) -> dict[str, Any]:
        from datetime import datetime, timezone

        hook = {
            "id": f"ghhook_{uuid4().hex}",
            "token": token,
            "user_id": user_id,
            "repository": repository.lower(),
            "routine_id": routine_id,
            "hook_id": hook_id,
            "secret": secret,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        encrypted_secret = self._encrypt(
            secret, user_id=user_id, integration="github_webhook"
        )
        token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
        with self._lock, self._connection() as conn:
            conn.execute(
                """
                INSERT INTO proactive_github_hooks
                    (id, token, user_id, repository, routine_id, hook_id, secret,
                     encryption_version, encryption_key_version, credential_status, created_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    hook["id"],
                    token_hash,
                    user_id,
                    hook["repository"],
                    routine_id,
                    hook_id,
                    encrypted_secret,
                    "v2",
                    self._vault.key_version,
                    "active",
                    hook["created_at"],
                ),
            )
        return hook

    def get_github_hook(self, token: str) -> Optional[dict[str, Any]]:
        token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
        with self._connection() as conn:
            row = conn.execute(
                """
                SELECT * FROM proactive_github_hooks
                WHERE token = %s OR token = %s
                """,
                (token_hash, token),
            ).fetchone()
        if not row:
            return None
        hook = dict(row)
        try:
            hook["secret"] = self._decrypt(
                hook["secret"],
                user_id=str(hook["user_id"]),
                integration="github_webhook",
            )
        except CredentialUnrecoverableError as exc:
            with self._lock, self._connection() as conn:
                conn.execute(
                    """
                    UPDATE proactive_github_hooks
                    SET credential_status = 'reconnect_required',
                        encryption_version = 'legacy-unrecoverable',
                        encryption_key_version = NULL
                    WHERE id = %s
                    """,
                    (hook["id"],),
                )
            raise CredentialReconnectRequiredError(
                "GitHub webhook secret uses an outdated encryption key and requires reconnection."
            ) from exc
        return hook

    def list_github_hooks_for_routine(self, routine_id: str) -> list[dict[str, Any]]:
        with self._connection() as conn:
            rows = conn.execute(
                "SELECT * FROM proactive_github_hooks WHERE routine_id = %s", (routine_id,)
            ).fetchall()
        return [
            {key: value for key, value in dict(row).items() if key not in {"token", "secret"}}
            for row in rows
        ]

    def list_github_hooks_for_user(self, user_id: str) -> list[dict[str, Any]]:
        with self._connection() as conn:
            rows = conn.execute(
                "SELECT * FROM proactive_github_hooks WHERE user_id = %s", (user_id,)
            ).fetchall()
        return [
            {key: value for key, value in dict(row).items() if key not in {"token", "secret"}}
            for row in rows
        ]

    def delete_github_hook_record(self, hook_id: str) -> bool:
        with self._lock, self._connection() as conn:
            cur = conn.execute("DELETE FROM proactive_github_hooks WHERE id = %s", (hook_id,))
            return cur.rowcount > 0

    def delete_github_hooks_for_routine(self, routine_id: str) -> list[dict[str, Any]]:
        with self._lock, self._connection() as conn:
            rows = conn.execute(
                "SELECT * FROM proactive_github_hooks WHERE routine_id = %s", (routine_id,)
            ).fetchall()
            conn.execute("DELETE FROM proactive_github_hooks WHERE routine_id = %s", (routine_id,))
        return [
            {key: value for key, value in dict(row).items() if key not in {"token", "secret"}}
            for row in rows
        ]

    def migrate_github_hook_tokens(self) -> int:
        """Hash legacy raw GitHub webhook callback tokens at rest."""
        migrated = 0
        with self._lock, self._connection() as conn:
            rows = conn.execute(
                "SELECT id, token FROM proactive_github_hooks"
            ).fetchall()
            for row in rows:
                token = str(row["token"] or "")
                if token and not re.fullmatch(r"[0-9a-f]{64}", token):
                    conn.execute(
                        "UPDATE proactive_github_hooks SET token = %s WHERE id = %s",
                        (hashlib.sha256(token.encode("utf-8")).hexdigest(), row["id"]),
                    )
                    migrated += 1
        return migrated

    def reconcile_legacy_credentials(self) -> dict[str, int]:
        """Detect undecryptable provider/hook credentials and mark reconnect_required.

        Never deletes rows or marks unrecoverable ciphertext as migrated.
        """
        accounts_marked = 0
        hooks_marked = 0
        with self._lock, self._connection() as conn:
            account_rows = conn.execute(
                """
                SELECT id, user_id, provider, provider_account_id, access_token, refresh_token,
                       credential_status
                FROM proactive_provider_accounts
                """
            ).fetchall()
            for row in account_rows:
                if row.get("credential_status") == "reconnect_required":
                    continue
                user_id = str(row["user_id"])
                provider = str(row["provider"])
                try:
                    if self._vault.configured:
                        self._vault.decrypt_text(
                            row["access_token"], user_id=user_id, integration=provider
                        )
                        if row.get("refresh_token"):
                            self._vault.decrypt_text(
                                row["refresh_token"], user_id=user_id, integration=provider
                            )
                        continue
                    elif not self._vault.is_legacy_ciphertext(row.get("access_token") or ""):
                        continue
                except CredentialUnrecoverableError:
                    pass

                if self._vault.configured or self._vault.is_legacy_ciphertext(
                    row.get("access_token") or ""
                ):
                    reason = RECONNECT_MESSAGE_TEMPLATE.format(provider=provider.title())
                    conn.execute(
                        """
                        UPDATE proactive_provider_accounts
                        SET credential_status = 'reconnect_required',
                            encryption_version = 'legacy-unrecoverable',
                            encryption_key_version = NULL,
                            credential_error = %s
                        WHERE id = %s
                        """,
                        (reason, row["id"]),
                    )
                    accounts_marked += 1

            hook_rows = conn.execute(
                """
                SELECT id, user_id, secret, credential_status
                FROM proactive_github_hooks
                """
            ).fetchall()
            for row in hook_rows:
                if row.get("credential_status") == "reconnect_required":
                    continue
                try:
                    if self._vault.configured:
                        self._vault.decrypt_text(
                            row["secret"],
                            user_id=str(row["user_id"]),
                            integration="github_webhook",
                        )
                        continue
                    elif not self._vault.is_legacy_ciphertext(row.get("secret") or ""):
                        continue
                except CredentialUnrecoverableError:
                    pass

                if self._vault.configured or self._vault.is_legacy_ciphertext(
                    row.get("secret") or ""
                ):
                    conn.execute(
                        """
                        UPDATE proactive_github_hooks
                        SET credential_status = 'reconnect_required',
                            encryption_version = 'legacy-unrecoverable',
                            encryption_key_version = NULL
                        WHERE id = %s
                        """,
                        (row["id"],),
                    )
                    hooks_marked += 1

        return {
            "accounts_marked_reconnect_required": accounts_marked,
            "hooks_marked_reconnect_required": hooks_marked,
        }

    def rotate_credentials(self) -> dict[str, int]:
        """Re-encrypt active credentials encrypted under a previous key version."""
        if not self._vault.configured:
            raise CredentialVaultNotConfiguredError(
                "Configure INTEGRATION_ENCRYPTION_KEY before rotating credentials."
            )
        accounts_rotated = 0
        hooks_rotated = 0
        with self._lock, self._connection() as conn:
            account_rows = conn.execute(
                """
                SELECT id, user_id, provider, access_token, refresh_token, credential_status
                FROM proactive_provider_accounts
                WHERE credential_status IS NULL OR credential_status = 'active'
                """
            ).fetchall()
            for row in account_rows:
                user_id = str(row["user_id"])
                provider = str(row["provider"])
                access_val = row.get("access_token") or ""
                refresh_val = row.get("refresh_token") or ""
                if not (
                    self._vault.needs_rotation(access_val)
                    or (refresh_val and self._vault.needs_rotation(refresh_val))
                ):
                    continue
                new_access = self._vault.reencrypt_text(
                    access_val, user_id=user_id, integration=provider
                )
                new_refresh = (
                    self._vault.reencrypt_text(
                        refresh_val, user_id=user_id, integration=provider
                    )
                    if refresh_val
                    else None
                )
                conn.execute(
                    """
                    UPDATE proactive_provider_accounts
                    SET access_token = %s,
                        refresh_token = %s,
                        encryption_version = 'v2',
                        encryption_key_version = %s
                    WHERE id = %s
                    """,
                    (new_access, new_refresh, self._vault.key_version, row["id"]),
                )
                accounts_rotated += 1

            hook_rows = conn.execute(
                """
                SELECT id, user_id, secret, credential_status
                FROM proactive_github_hooks
                WHERE credential_status IS NULL OR credential_status = 'active'
                """
            ).fetchall()
            for row in hook_rows:
                secret_val = row.get("secret") or ""
                if not self._vault.needs_rotation(secret_val):
                    continue
                new_secret = self._vault.reencrypt_text(
                    secret_val,
                    user_id=str(row["user_id"]),
                    integration="github_webhook",
                )
                conn.execute(
                    """
                    UPDATE proactive_github_hooks
                    SET secret = %s,
                        encryption_version = 'v2',
                        encryption_key_version = %s
                    WHERE id = %s
                    """,
                    (new_secret, self._vault.key_version, row["id"]),
                )
                hooks_rotated += 1

        return {
            "accounts_rotated": accounts_rotated,
            "hooks_rotated": hooks_rotated,
        }


def verify_github_signature(secret: str, body: bytes, signature: Optional[str]) -> bool:
    if not signature or not signature.startswith("sha256="):
        return False
    expected = "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, signature)


def verify_slack_signature(
    signing_secret: str, body: bytes, timestamp: Optional[str], signature: Optional[str]
) -> bool:
    if not timestamp or not signature or not signature.startswith("v0="):
        return False
    try:
        import time

        if abs(time.time() - int(timestamp)) > 300:
            return False
    except (TypeError, ValueError):
        return False
    base = b"v0:" + timestamp.encode() + b":" + body
    expected = "v0=" + hmac.new(signing_secret.encode(), base, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, signature)


def normalize_github_event(
    payload: dict[str, Any], provider_event: str, delivery_id: Optional[str], user_id: str
) -> dict[str, Any]:
    action = str(payload.get("action") or "received")
    repository = payload.get("repository") or {}
    issue = payload.get("issue") or payload.get("pull_request") or {}
    labels = [
        str(label.get("name", ""))
        for label in issue.get("labels", [])
        if isinstance(label, dict) and label.get("name")
    ]
    return {
        "event_id": delivery_id or f"github_{uuid4().hex}",
        "source": "github",
        "event_type": f"{provider_event}.{action}",
        "occurred_at": str(payload.get("updated_at") or payload.get("created_at") or ""),
        "tenant_id": user_id,
        "payload": {
            "repository": repository.get("full_name"),
            "action": action,
            "issue": issue,
            "labels": labels,
            "sender": payload.get("sender"),
            "comment": payload.get("comment"),
            "pull_request": payload.get("pull_request"),
        },
        "metadata": {"provider_event_id": delivery_id},
    }


def normalize_slack_event(
    event: dict[str, Any], team_id: str, user_id: str
) -> dict[str, Any]:
    event_type = str(event.get("type") or "message")
    if event_type == "message" and event.get("subtype"):
        kind = f"message.{event['subtype']}"
    elif event_type == "message":
        kind = "message.channels"
    else:
        kind = event_type
    payload = {
        "team_id": team_id,
        "channel_id": event.get("channel"),
        "text": event.get("text") or "",
        "user_id": event.get("user"),
        "ts": event.get("ts"),
        "thread_ts": event.get("thread_ts"),
        "reaction": event.get("reaction"),
        "item": event.get("item"),
    }
    return {
        "event_id": str(event.get("event_id") or event.get("ts") or uuid4().hex),
        "source": "slack",
        "event_type": kind,
        "occurred_at": str(event.get("event_ts") or event.get("ts") or ""),
        "tenant_id": user_id,
        "payload": payload,
        "metadata": {"provider_event_id": event.get("event_id") or event.get("ts")},
    }


def normalize_gmail_message(
    message: dict[str, Any], email_address: str, user_id: str
) -> dict[str, Any]:
    headers = {
        str(header.get("name", "")).casefold(): str(header.get("value", ""))
        for header in (message.get("payload") or {}).get("headers", [])
        if isinstance(header, dict)
    }

    def decode_body(part: dict[str, Any]) -> list[str]:
        chunks: list[str] = []
        body = part.get("body")
        if isinstance(body, dict) and isinstance(body.get("data"), str):
            try:
                decoded = base64.urlsafe_b64decode(body["data"] + "===")
                chunks.append(decoded.decode("utf-8", errors="replace"))
            except (ValueError, UnicodeEncodeError):
                pass
        for child in part.get("parts", []):
            if isinstance(child, dict):
                chunks.extend(decode_body(child))
        return chunks

    body = "\n".join(decode_body(message.get("payload") or {}))[:20_000]
    return {
        "event_id": f"gmail_{email_address}_{message.get('id', '')}",
        "source": "gmail",
        "event_type": "email.received",
        "occurred_at": str(message.get("internalDate") or ""),
        "tenant_id": user_id,
        "payload": {
            "email_address": email_address,
            "message_id": str(message.get("id") or ""),
            "thread_id": str(message.get("threadId") or ""),
            "from": headers.get("from", ""),
            "to": headers.get("to", ""),
            "subject": headers.get("subject", ""),
            "date": headers.get("date", ""),
            "snippet": str(message.get("snippet") or "")[:2_000],
            "body": body,
            "labels": message.get("labelIds") or [],
        },
        "metadata": {"provider_event_id": str(message.get("id") or "")},
    }


async def create_github_webhook(
    *,
    access_token: str,
    repository: str,
    callback_url: str,
    secret: str,
    provider_event: str,
) -> int:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository):
        raise ValueError("Repository must use the owner/repository format.")
    if provider_event not in {"issues", "pull_request", "issue_comment"}:
        raise ValueError("Unsupported GitHub event type.")
    async with httpx.AsyncClient(timeout=15.0) as client:
        response = await client.post(
            f"https://api.github.com/repos/{repository}/hooks",
            headers={
                "Authorization": f"Bearer {access_token}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
            },
            json={
                "name": "web",
                "active": True,
                "events": [provider_event],
                "config": {
                    "url": callback_url,
                    "content_type": "json",
                    "secret": secret,
                    "insecure_ssl": "0",
                },
            },
        )
    if response.status_code != 201:
        detail = "GitHub could not create the repository webhook."
        try:
            detail = response.json().get("message") or detail
        except (ValueError, AttributeError):
            pass
        raise RuntimeError(f"{detail} (HTTP {response.status_code})")
    hook_id = response.json().get("id")
    if not isinstance(hook_id, int):
        raise RuntimeError("GitHub created a webhook without returning its hook ID.")
    return hook_id


async def delete_github_webhook(
    *, access_token: str, repository: str, hook_id: int
) -> bool:
    async with httpx.AsyncClient(timeout=15.0) as client:
        response = await client.delete(
            f"https://api.github.com/repos/{repository}/hooks/{hook_id}",
            headers={
                "Authorization": f"Bearer {access_token}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
            },
        )
    if response.status_code == 404:
        return True
    if response.status_code != 204:
        raise RuntimeError(f"GitHub could not delete the webhook (HTTP {response.status_code}).")
    return True


async def resolve_slack_channel_id(access_token: str, channel: str) -> str:
    candidate = channel.strip().lstrip("#")
    if re.fullmatch(r"[CGD][A-Z0-9]+", candidate):
        return candidate
    cursor: Optional[str] = None
    async with httpx.AsyncClient(timeout=15.0) as client:
        for _ in range(20):
            response = await client.get(
                "https://slack.com/api/conversations.list",
                headers={"Authorization": f"Bearer {access_token}"},
                params={
                    "types": "public_channel,private_channel",
                    "limit": 200,
                    **({"cursor": cursor} if cursor else {}),
                },
            )
            response.raise_for_status()
            payload = response.json()
            if not payload.get("ok"):
                raise RuntimeError(payload.get("error") or "Slack could not list channels.")
            for item in payload.get("channels", []):
                if str(item.get("name", "")).casefold() == candidate.casefold():
                    return str(item["id"])
            cursor = str((payload.get("response_metadata") or {}).get("next_cursor") or "")
            if not cursor:
                break
    raise ValueError(f"Could not find Slack channel #{candidate} in the connected workspace.")
