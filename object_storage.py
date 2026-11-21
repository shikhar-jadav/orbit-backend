"""Private S3-compatible storage for attachments and workspace files."""

from __future__ import annotations

import os
from pathlib import Path, PurePosixPath
from typing import Any


class S3ObjectStorage:
    def __init__(
        self,
        *,
        endpoint_url: str | None = None,
        bucket: str | None = None,
        access_key_id: str | None = None,
        secret_access_key: str | None = None,
        region: str | None = None,
        client: Any = None,
    ):
        self.bucket = (bucket or os.getenv("S3_BUCKET", "")).strip()
        endpoint = (endpoint_url or os.getenv("S3_ENDPOINT_URL", "")).strip()
        access_key = (access_key_id or os.getenv("S3_ACCESS_KEY_ID", "")).strip()
        secret_key = (secret_access_key or os.getenv("S3_SECRET_ACCESS_KEY", "")).strip()
        region_name = (region or os.getenv("S3_REGION", "auto")).strip()

        if not self.bucket:
            raise RuntimeError("S3_BUCKET must be configured for object storage.")
        if client is None and not all((endpoint, access_key, secret_key)):
            raise RuntimeError(
                "S3_ENDPOINT_URL, S3_ACCESS_KEY_ID, and S3_SECRET_ACCESS_KEY "
                "must be configured for object storage."
            )

        if client is None:
            import boto3

            client = boto3.client(
                "s3",
                endpoint_url=endpoint,
                aws_access_key_id=access_key,
                aws_secret_access_key=secret_key,
                region_name=region_name,
            )
        self.client = client

    def put_bytes(
        self,
        key: str,
        content: bytes,
        content_type: str = "application/octet-stream",
    ) -> None:
        safe_key = _normalize_key(key)
        self.client.put_object(
            Bucket=self.bucket,
            Key=safe_key,
            Body=content,
            ContentType=content_type,
        )

    def get_bytes(self, key: str) -> bytes:
        response = self.client.get_object(
            Bucket=self.bucket,
            Key=_normalize_key(key),
        )
        return response["Body"].read()

    def delete(self, key: str) -> None:
        self.client.delete_object(
            Bucket=self.bucket,
            Key=_normalize_key(key),
        )

    def delete_prefix(self, prefix: str) -> None:
        safe_prefix = _normalize_prefix(prefix)
        keys = self.list_keys(safe_prefix)
        for offset in range(0, len(keys), 1000):
            batch = keys[offset : offset + 1000]
            self.client.delete_objects(
                Bucket=self.bucket,
                Delete={"Objects": [{"Key": key} for key in batch], "Quiet": True},
            )

    def list_keys(self, prefix: str) -> list[str]:
        safe_prefix = _normalize_prefix(prefix)
        paginator = self.client.get_paginator("list_objects_v2")
        keys: list[str] = []
        for page in paginator.paginate(Bucket=self.bucket, Prefix=safe_prefix):
            keys.extend(item["Key"] for item in page.get("Contents", []))
        return keys

    def download_prefix(self, prefix: str, destination: Path) -> None:
        safe_prefix = _normalize_prefix(prefix)
        root = destination.resolve()
        root.mkdir(parents=True, exist_ok=True)
        for key in self.list_keys(safe_prefix):
            relative = key[len(safe_prefix) :]
            if not relative:
                continue
            target = (root / Path(*PurePosixPath(relative).parts)).resolve()
            if target != root and root not in target.parents:
                raise ValueError(f"Object key escapes the destination directory: {key}")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(self.get_bytes(key))

    def sync_directory(self, source: Path, prefix: str) -> None:
        safe_prefix = _normalize_prefix(prefix)
        root = source.resolve()
        root.mkdir(parents=True, exist_ok=True)
        local_keys: set[str] = set()
        for path in root.rglob("*"):
            if not path.is_file() or path.is_symlink():
                continue
            relative = path.relative_to(root).as_posix()
            key = _normalize_key(f"{safe_prefix}{relative}")
            local_keys.add(key)
            self.client.upload_file(
                str(path),
                self.bucket,
                key,
                ExtraArgs={"ContentType": _content_type(path)},
            )

        stale_keys = [
            key for key in self.list_keys(safe_prefix) if key not in local_keys
        ]
        for offset in range(0, len(stale_keys), 1000):
            batch = stale_keys[offset : offset + 1000]
            self.client.delete_objects(
                Bucket=self.bucket,
                Delete={"Objects": [{"Key": key} for key in batch], "Quiet": True},
            )


def _normalize_key(key: str) -> str:
    path = PurePosixPath(key)
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError("Object keys must be relative and cannot contain traversal segments.")
    return path.as_posix()


def _normalize_prefix(prefix: str) -> str:
    safe = _normalize_key(prefix.rstrip("/"))
    return safe + "/"


def _content_type(path: Path) -> str:
    import mimetypes

    return mimetypes.guess_type(path.name)[0] or "application/octet-stream"
