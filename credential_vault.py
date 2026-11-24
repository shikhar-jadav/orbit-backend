"""Central authenticated encryption service for Orbit integration credentials.

Uses AES-256-GCM with per-(user_id, integration) HKDF-SHA256 subkeys and
associated authenticated data (AAD). Encrypted values carry a versioned
envelope (`enc:v2:<key_version>:<payload>`) so key rotation and unrecoverable
legacy ciphertext detection work cleanly without touching secrets.
"""
from __future__ import annotations

import base64
import binascii
import json
import os
import re
import secrets
from typing import Any, Optional

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF


ENCRYPTED_V2_PREFIX = "enc:v2:"
LEGACY_V1_PREFIX = "enc:v1:"
LEGACY_FERNET_PREFIX = "gAAAAA"
DEFAULT_KEY_VERSION = "k1"
_KEY_VERSION_RE = re.compile(r"^[A-Za-z0-9_-]{1,32}$")
_SECRET_KEY_RE = re.compile(
    r"(?:secret|token|password|passwd|api[_-]?key|access[_-]?key|"
    r"private[_-]?key|credential|authorization)",
    re.IGNORECASE,
)


class CredentialEncryptionError(RuntimeError):
    """Base error raised when credential encryption or decryption cannot proceed."""


class CredentialVaultNotConfiguredError(CredentialEncryptionError):
    """Raised when INTEGRATION_ENCRYPTION_KEY is missing or invalid."""


class CredentialUnrecoverableError(CredentialEncryptionError):
    """Raised when stored ciphertext cannot be decrypted with any configured key."""


class CredentialReconnectRequiredError(CredentialUnrecoverableError):
    """Raised when an integration account must be reconnected by the user."""


def generate_encryption_key() -> str:
    """Generate a cryptographically random 256-bit key encoded as URL-safe base64."""
    return base64.urlsafe_b64encode(secrets.token_bytes(32)).decode("ascii")


def _decode_master_key(raw_key: str) -> bytes:
    """Decode and strictly validate a 256-bit (32-byte) master key."""
    candidate = (raw_key or "").strip()
    if not candidate:
        raise CredentialVaultNotConfiguredError(
            "INTEGRATION_ENCRYPTION_KEY is not configured."
        )

    # Support 64-char hex
    if re.fullmatch(r"[0-9a-fA-F]{64}", candidate):
        return bytes.fromhex(candidate)

    # Support URL-safe or standard base64 (with or without trailing '=')
    padded = candidate + "=" * (-len(candidate) % 4)
    for decoder in (base64.urlsafe_b64decode, base64.b64decode):
        try:
            decoded = decoder(padded.encode("ascii"))
            if len(decoded) == 32:
                return decoded
        except (binascii.Error, ValueError, UnicodeEncodeError):
            continue

    raise CredentialVaultNotConfiguredError(
        "INTEGRATION_ENCRYPTION_KEY must decode to exactly 32 bytes "
        "(URL-safe base64 or 64-character hex)."
    )


def _normalize_key_version(version: Optional[str]) -> str:
    value = (version or DEFAULT_KEY_VERSION).strip() or DEFAULT_KEY_VERSION
    if not _KEY_VERSION_RE.match(value):
        raise CredentialVaultNotConfiguredError(
            "INTEGRATION_ENCRYPTION_KEY_VERSION must be 1-32 characters of [A-Za-z0-9_-]."
        )
    return value


def _parse_previous_keys(raw: Optional[str]) -> dict[str, bytes]:
    text = (raw or "").strip()
    if not text:
        return {}
    parsed: dict[str, bytes] = {}
    if text.startswith("{"):
        data = json.loads(text)
        if not isinstance(data, dict):
            raise CredentialVaultNotConfiguredError(
                "INTEGRATION_ENCRYPTION_PREVIOUS_KEYS JSON must be an object mapping version to key."
            )
        for ver, key_val in data.items():
            parsed[_normalize_key_version(str(ver))] = _decode_master_key(str(key_val))
        return parsed

    for entry in text.split(","):
        item = entry.strip()
        if not item:
            continue
        if ":" not in item:
            raise CredentialVaultNotConfiguredError(
                "INTEGRATION_ENCRYPTION_PREVIOUS_KEYS entries must have format '<version>:<key>'."
            )
        ver, key_val = item.split(":", 1)
        parsed[_normalize_key_version(ver)] = _decode_master_key(key_val)
    return parsed


def _derive_subkey(master_key: bytes, key_version: str, user_id: str, integration: str) -> bytes:
    info = f"orbit:credential:v2:{key_version}:{user_id}:{integration}".encode("utf-8")
    hkdf = HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=None,
        info=info,
    )
    return hkdf.derive(master_key)


def _build_aad(key_version: str, user_id: str, integration: str) -> bytes:
    return f"v2|kv={key_version}|user={user_id}|integration={integration}".encode("utf-8")


class CredentialVault:
    """AES-256-GCM credential encryption backed by INTEGRATION_ENCRYPTION_KEY."""

    def __init__(
        self,
        key: Optional[str] = None,
        *,
        key_version: Optional[str] = None,
        previous_keys: Optional[dict[str, str]] = None,
    ) -> None:
        raw_key = key if key is not None else os.getenv("INTEGRATION_ENCRYPTION_KEY")
        self._key_version = _normalize_key_version(
            key_version if key_version is not None else os.getenv("INTEGRATION_ENCRYPTION_KEY_VERSION")
        )
        self._keys: dict[str, bytes] = {}
        self._active_key: Optional[bytes] = None

        if raw_key and raw_key.strip():
            decoded = _decode_master_key(raw_key)
            self._active_key = decoded
            self._keys[self._key_version] = decoded

        if previous_keys:
            for ver, prev_key in previous_keys.items():
                norm_ver = _normalize_key_version(ver)
                if norm_ver not in self._keys:
                    self._keys[norm_ver] = _decode_master_key(prev_key)
        else:
            env_prev = os.getenv("INTEGRATION_ENCRYPTION_PREVIOUS_KEYS")
            if env_prev and env_prev.strip():
                for norm_ver, prev_bytes in _parse_previous_keys(env_prev).items():
                    if norm_ver not in self._keys:
                        self._keys[norm_ver] = prev_bytes

    @property
    def configured(self) -> bool:
        return self._active_key is not None

    @property
    def key_version(self) -> str:
        return self._key_version

    @staticmethod
    def is_encrypted_v2(value: str) -> bool:
        return isinstance(value, str) and value.startswith(ENCRYPTED_V2_PREFIX)

    @staticmethod
    def is_legacy_ciphertext(value: str) -> bool:
        if not isinstance(value, str) or not value:
            return False
        return value.startswith(LEGACY_V1_PREFIX) or value.startswith(LEGACY_FERNET_PREFIX)

    @staticmethod
    def parse_v2_envelope(value: str) -> tuple[str, bytes]:
        if not isinstance(value, str) or not value.startswith(ENCRYPTED_V2_PREFIX):
            raise CredentialUnrecoverableError("Not a v2 encrypted credential.")
        body = value[len(ENCRYPTED_V2_PREFIX):]
        if ":" not in body:
            raise CredentialUnrecoverableError("Malformed v2 encrypted credential envelope.")
        ver, encoded_payload = body.split(":", 1)
        if not _KEY_VERSION_RE.match(ver):
            raise CredentialUnrecoverableError("Invalid key version in encrypted credential.")
        try:
            padded = encoded_payload + "=" * (-len(encoded_payload) % 4)
            raw_payload = base64.urlsafe_b64decode(padded.encode("ascii"))
        except (binascii.Error, ValueError, UnicodeEncodeError) as exc:
            raise CredentialUnrecoverableError("Corrupted encrypted credential payload.") from exc
        if len(raw_payload) <= 12 + 16:
            raise CredentialUnrecoverableError("Truncated encrypted credential payload.")
        return ver, raw_payload

    def inspect_value(self, value: Optional[str]) -> dict[str, Any]:
        if not value:
            return {
                "encrypted": False,
                "encryption_version": None,
                "key_version": None,
            }
        if value.startswith(ENCRYPTED_V2_PREFIX):
            try:
                ver, _ = self.parse_v2_envelope(value)
                return {
                    "encrypted": True,
                    "encryption_version": "v2",
                    "key_version": ver,
                }
            except CredentialUnrecoverableError:
                return {
                    "encrypted": True,
                    "encryption_version": "v2-corrupted",
                    "key_version": None,
                }
        if value.startswith(LEGACY_V1_PREFIX):
            return {
                "encrypted": True,
                "encryption_version": "legacy-v1",
                "key_version": None,
            }
        if value.startswith(LEGACY_FERNET_PREFIX):
            return {
                "encrypted": True,
                "encryption_version": "legacy-fernet",
                "key_version": None,
            }
        return {
            "encrypted": False,
            "encryption_version": "plaintext",
            "key_version": None,
        }

    def encrypt_text(
        self,
        value: str,
        *,
        user_id: str = "global",
        integration: str = "default",
    ) -> str:
        if not value:
            return value
        if value.startswith(ENCRYPTED_V2_PREFIX):
            return value
        if self._active_key is None:
            raise CredentialVaultNotConfiguredError(
                "Configure INTEGRATION_ENCRYPTION_KEY before saving connector credentials."
            )
        subkey = _derive_subkey(self._active_key, self._key_version, user_id, integration)
        aad = _build_aad(self._key_version, user_id, integration)
        nonce = os.urandom(12)
        ciphertext = AESGCM(subkey).encrypt(nonce, value.encode("utf-8"), aad)
        encoded = base64.urlsafe_b64encode(nonce + ciphertext).decode("ascii")
        return f"{ENCRYPTED_V2_PREFIX}{self._key_version}:{encoded}"

    def decrypt_text(
        self,
        value: str,
        *,
        user_id: str = "global",
        integration: str = "default",
    ) -> str:
        if not value:
            return value
        if self._active_key is None and not self._keys:
            raise CredentialVaultNotConfiguredError(
                "Configure INTEGRATION_ENCRYPTION_KEY to read saved connector credentials."
            )
        if value.startswith(LEGACY_V1_PREFIX) or value.startswith(LEGACY_FERNET_PREFIX):
            raise CredentialUnrecoverableError(
                "Stored credential uses a retired encryption key and cannot be decrypted. "
                "Reconnect this integration to re-encrypt it."
            )
        if not value.startswith(ENCRYPTED_V2_PREFIX):
            # Legacy unencrypted value when a valid key is configured: callers
            # migrating legacy plaintext records can re-encrypt them.
            return value

        ver, raw_payload = self.parse_v2_envelope(value)
        master_key = self._keys.get(ver)
        if master_key is None:
            raise CredentialUnrecoverableError(
                f"Encryption key version '{ver}' is not available. Reconnect this integration."
            )

        nonce = raw_payload[:12]
        ciphertext = raw_payload[12:]
        subkey = _derive_subkey(master_key, ver, user_id, integration)
        aad = _build_aad(ver, user_id, integration)
        try:
            plaintext = AESGCM(subkey).decrypt(nonce, ciphertext, aad)
            return plaintext.decode("utf-8")
        except (InvalidTag, UnicodeDecodeError) as exc:
            raise CredentialUnrecoverableError(
                "Credential decryption failed; the credential may belong to a different "
                "key, user, or integration. Reconnect this integration."
            ) from exc

    def needs_rotation(self, value: Optional[str]) -> bool:
        if not value or not value.startswith(ENCRYPTED_V2_PREFIX):
            return False
        try:
            ver, _ = self.parse_v2_envelope(value)
            return ver != self._key_version
        except CredentialUnrecoverableError:
            return False

    def reencrypt_text(
        self,
        value: str,
        *,
        user_id: str = "global",
        integration: str = "default",
    ) -> str:
        if not value:
            return value
        plaintext = self.decrypt_text(value, user_id=user_id, integration=integration)
        if self._active_key is None:
            raise CredentialVaultNotConfiguredError(
                "Configure INTEGRATION_ENCRYPTION_KEY before rotating credentials."
            )
        subkey = _derive_subkey(self._active_key, self._key_version, user_id, integration)
        aad = _build_aad(self._key_version, user_id, integration)
        nonce = os.urandom(12)
        ciphertext = AESGCM(subkey).encrypt(nonce, plaintext.encode("utf-8"), aad)
        encoded = base64.urlsafe_b64encode(nonce + ciphertext).decode("ascii")
        return f"{ENCRYPTED_V2_PREFIX}{self._key_version}:{encoded}"

    def encrypt_headers(
        self,
        headers: Any,
        *,
        user_id: str = "global",
        integration: str = "default",
    ) -> list[dict[str, str]]:
        result: list[dict[str, str]] = []
        for item in headers if isinstance(headers, list) else []:
            if not isinstance(item, dict):
                continue
            key = str(item.get("key", "")).strip()
            if not key:
                continue
            result.append({
                "key": key,
                "value": self.encrypt_text(
                    str(item.get("value", "")),
                    user_id=user_id,
                    integration=integration,
                ),
            })
        return result

    def decrypt_headers(
        self,
        headers: Any,
        *,
        user_id: str = "global",
        integration: str = "default",
    ) -> list[dict[str, str]]:
        result: list[dict[str, str]] = []
        for item in headers if isinstance(headers, list) else []:
            if not isinstance(item, dict):
                continue
            key = str(item.get("key", "")).strip()
            if not key:
                continue
            result.append({
                "key": key,
                "value": self.decrypt_text(
                    str(item.get("value", "")),
                    user_id=user_id,
                    integration=integration,
                ),
            })
        return result

    def encrypt_secret_fields(
        self,
        value: Any,
        *,
        user_id: str = "global",
        integration: str = "default",
    ) -> Any:
        if isinstance(value, dict):
            result: dict[str, Any] = {}
            for key, item in value.items():
                if isinstance(item, str) and _SECRET_KEY_RE.search(str(key)):
                    result[str(key)] = self.encrypt_text(
                        item, user_id=user_id, integration=integration
                    )
                else:
                    result[str(key)] = self.encrypt_secret_fields(
                        item, user_id=user_id, integration=integration
                    )
            return result
        if isinstance(value, list):
            return [
                self.encrypt_secret_fields(item, user_id=user_id, integration=integration)
                for item in value
            ]
        return value

    def decrypt_secret_fields(
        self,
        value: Any,
        *,
        user_id: str = "global",
        integration: str = "default",
    ) -> Any:
        if isinstance(value, dict):
            result: dict[str, Any] = {}
            for key, item in value.items():
                if isinstance(item, str) and _SECRET_KEY_RE.search(str(key)):
                    result[str(key)] = self.decrypt_text(
                        item, user_id=user_id, integration=integration
                    )
                else:
                    result[str(key)] = self.decrypt_secret_fields(
                        item, user_id=user_id, integration=integration
                    )
            return result
        if isinstance(value, list):
            return [
                self.decrypt_secret_fields(item, user_id=user_id, integration=integration)
                for item in value
            ]
        return value

    def mask_text(self, value: str) -> str:
        return "••••••" if value else ""
