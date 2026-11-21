import pytest

from test_database import IsolatedPostgresDatabase


class MemoryObjectStorage:
    def __init__(self):
        self.objects = {}

    def put_bytes(self, key, content, content_type="application/octet-stream"):
        self.objects[key] = bytes(content)

    def get_bytes(self, key):
        return self.objects[key]

    def delete(self, key):
        self.objects.pop(key, None)

    def delete_prefix(self, prefix):
        prefix = prefix.rstrip("/") + "/"
        for key in list(self.objects):
            if key.startswith(prefix):
                del self.objects[key]

    def download_prefix(self, prefix, destination):
        from pathlib import Path

        prefix = prefix.rstrip("/") + "/"
        root = Path(destination)
        for key, content in self.objects.items():
            if key.startswith(prefix):
                target = root / key[len(prefix) :]
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(content)

    def sync_directory(self, source, prefix):
        from pathlib import Path

        prefix = prefix.rstrip("/") + "/"
        root = Path(source)
        local = {}
        for path in root.rglob("*"):
            if path.is_file() and not path.is_symlink():
                local[f"{prefix}{path.relative_to(root).as_posix()}"] = path.read_bytes()
        self.delete_prefix(prefix)
        self.objects.update(local)


class EmptyCheckpointer:
    def delete_thread(self, thread_id):
        return None


@pytest.fixture
def postgres_database():
    with IsolatedPostgresDatabase() as database:
        yield database
