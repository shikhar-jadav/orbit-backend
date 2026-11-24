"""Application-wide encryption helpers for user connector credentials.

Values are version-prefixed so encrypted records can be rotated and legacy plaintext
records can be migrated without confusing ciphertext with ordinary values.
"""
from __future__ import annotations

import os
import re
from typing import Any, Optional

from cryptography.fernet import Fernet, InvalidToken


ENCRYPTED_PREFIX = "enc:v1:"
_SECRET_KEY_RE = re.compile(
    r"(?:secret|token|password|passwd|api[_-]?key|access[_-]?key|"
    r"private[_-]?key|credential|authorization)",
    re.IGNORECASE,
)


class CredentialVault:
    """Fernet-based credential encryption backed by a stable server-side key."""

    def __init__(self, key: Optional[str] = None) -> None:
        configured = key or os.getenv("INTEGRATION_TOKEN_ENCRYPTION_KEY") or os.getenv(
            "ORBIT_CREDENTIAL_ENCRYPTION_KEY"
        )
        self._fernet = (
            Fernet(configured.encode("ascii"))
            if configured and configured.strip()
            else None
        )

    @property
    def configured(self) -> bool:
        return self._fernet is not None

    def encrypt_text(self, value: str) -> str:
        if not value:
            return value
        if value.startswith(ENCRYPTED_PREFIX):
            return value
        if self._fernet is None:
            raise RuntimeError(
                "Configure ORBIT_CREDENTIAL_ENCRYPTION_KEY (or the existing "
                "INTEGRATION_TOKEN_ENCRYPTION_KEY) before saving connector credentials."
            )
        return ENCRYPTED_PREFIX + self._fernet.encrypt(value.encode("utf-8")).decode("ascii")

    def decrypt_text(self, value: str) -> str:
        if not value:
            return value
        if not value.startswith(ENCRYPTED_PREFIX):
            # Existing plaintext records are migrated during application startup.
            # Never hand legacy plaintext credentials to an agent when encryption is
            # unavailable; fail closed until the server key is configured.
            if self._fernet is None:
                raise RuntimeError(
                    "Connector credentials are not protected because the credential "
                    "encryption key is missing. Configure the key and restart Orbit."
                )
            return value
        if self._fernet is None:
            raise RuntimeError(
                "Configure the credential encryption key to read saved connector credentials."
            )
        token = value[len(ENCRYPTED_PREFIX):]
        try:
            return self._fernet.decrypt(token.encode("ascii")).decode("utf-8")
        except (InvalidToken, UnicodeDecodeError) as exc:
            raise RuntimeError(
                "Credential decryption failed; verify the stable credential encryption key."
            ) from exc

    def encrypt_headers(self, headers: Any) -> list[dict[str, str]]:
        result: list[dict[str, str]] = []
        for item in headers if isinstance(headers, list) else []:
            if not isinstance(item, dict):
                continue
            key = str(item.get("key", "")).strip()
            if not key:
                continue
            result.append({
                "key": key,
                "value": self.encrypt_text(str(item.get("value", ""))),
            })
        return result

    def decrypt_headers(self, headers: Any) -> list[dict[str, str]]:
        result: list[dict[str, str]] = []
        for item in headers if isinstance(headers, list) else []:
            if not isinstance(item, dict):
                continue
            key = str(item.get("key", "")).strip()
            if not key:
                continue
            result.append({
                "key": key,
                "value": self.decrypt_text(str(item.get("value", ""))),
            })
        return result

    def encrypt_secret_fields(self, value: Any) -> Any:
        if isinstance(value, dict):
            result: dict[str, Any] = {}
            for key, item in value.items():
                if isinstance(item, str) and _SECRET_KEY_RE.search(str(key)):
                    result[str(key)] = self.encrypt_text(item)
                else:
                    result[str(key)] = self.encrypt_secret_fields(item)
            return result
        if isinstance(value, list):
            return [self.encrypt_secret_fields(item) for item in value]
        return value

    def decrypt_secret_fields(self, value: Any) -> Any:
        if isinstance(value, dict):
            result: dict[str, Any] = {}
            for key, item in value.items():
                if isinstance(item, str) and _SECRET_KEY_RE.search(str(key)):
                    result[str(key)] = self.decrypt_text(item)
                else:
                    result[str(key)] = self.decrypt_secret_fields(item)
            return result
        if isinstance(value, list):
            return [self.decrypt_secret_fields(item) for item in value]
        return value

    def mask_text(self, value: str) -> str:
        return "••••••" if value else ""
