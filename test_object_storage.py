from __future__ import annotations

from io import BytesIO
from pathlib import Path

from object_storage import S3ObjectStorage
import workspace


class FakeS3Client:
    def __init__(self):
        self.objects: dict[tuple[str, str], bytes] = {}

    def put_object(self, *, Bucket, Key, Body, ContentType):
        self.objects[(Bucket, Key)] = bytes(Body)

    def get_object(self, *, Bucket, Key):
        return {"Body": BytesIO(self.objects[(Bucket, Key)])}

    def delete_object(self, *, Bucket, Key):
        self.objects.pop((Bucket, Key), None)

    def delete_objects(self, *, Bucket, Delete):
        for item in Delete["Objects"]:
            self.objects.pop((Bucket, item["Key"]), None)

    def upload_file(self, filename, bucket, key, ExtraArgs):
        self.objects[(bucket, key)] = Path(filename).read_bytes()

    def get_paginator(self, operation):
        assert operation == "list_objects_v2"
        return self

    def paginate(self, *, Bucket, Prefix):
        keys = [
            key
            for bucket, key in self.objects
            if bucket == Bucket and key.startswith(Prefix)
        ]
        return [{"Contents": [{"Key": key} for key in keys]}]


def test_s3_storage_round_trip_and_workspace_hydration(tmp_path, monkeypatch):
    client = FakeS3Client()
    storage = S3ObjectStorage(bucket="orbit-test", client=client)
    cache_root = tmp_path / "workspace-cache"
    monkeypatch.setattr(workspace, "WORKSPACE_ROOT", cache_root)
    workspace.configure_workspace_storage(storage)

    try:
        result = workspace.create_or_update_file(
            "reports/summary.md",
            "Persist this workspace file.",
            thread_id="thread-42",
        )
        assert result["status"] == "created"
        assert (
            client.objects[
                ("orbit-test", "workspaces/thread-42/reports/summary.md")
            ]
            == b"Persist this workspace file."
        )

        (cache_root / "thread-42" / "reports" / "summary.md").unlink()
        workspace.configure_workspace_storage(storage)
        restored = workspace.get_workspace_dir("thread-42") / "reports" / "summary.md"
        assert restored.read_text(encoding="utf-8") == "Persist this workspace file."

        storage.put_bytes("attachments/file-1", b"attachment", "text/plain")
        assert storage.get_bytes("attachments/file-1") == b"attachment"
    finally:
        workspace.configure_workspace_storage(None)
