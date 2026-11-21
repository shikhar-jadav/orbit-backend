import os
import sys
import asyncio
import subprocess
import contextvars
import base64
import mimetypes
import logging
import queue
import threading
import time
import tempfile
import urllib.parse
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable, Literal
from object_storage import S3ObjectStorage

# Callback hook for agent write tracking to suppress file event self-trigger loops
_agent_file_write_listener: Optional[Callable[[str], None]] = None
logger = logging.getLogger("orbit.workspace")

_workspace_change_lock = threading.Lock()
_workspace_change_subscribers: Dict[str, set[queue.Queue]] = defaultdict(set)
_recent_workspace_changes: Dict[tuple, float] = {}
_workspace_watcher: Optional["WorkspacePollingWatcher"] = None
_workspace_storage: Optional[S3ObjectStorage] = None
_hydrated_workspaces: set[str] = set()
_workspace_hydration_lock = threading.Lock()
_ARTIFACT_EXTENSIONS = {
    ".bmp", ".csv", ".docx", ".gif", ".htm", ".html", ".ico", ".ipynb", ".jpeg",
    ".jpg", ".js", ".json", ".jsx", ".md", ".markdown", ".mp3", ".mp4",
    ".m4a", ".pdf", ".png", ".pptx", ".py", ".svg", ".ts", ".tsx", ".txt",
    ".mov", ".wav", ".webm", ".webp", ".xlsx", ".xls", ".yaml", ".yml", ".css",
}
_IGNORED_WORKSPACE_DIRS = {
    ".git", ".next", ".pytest_cache", ".venv", ".vite", "__pycache__",
    "build", "coverage", "dist", "node_modules", "target", "venv",
}
_TEMP_FILE_SUFFIXES = {".part", ".swp", ".tmp", ".temp"}

def set_agent_file_write_listener(listener: Optional[Callable[[str], None]]) -> None:
    global _agent_file_write_listener
    _agent_file_write_listener = listener

# Context variable to track active thread in async execution contexts
active_thread_id_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "active_thread_id", default="default"
)

# Local execution cache; durable workspace files live in object storage.
WORKSPACE_ROOT = Path(tempfile.mkdtemp(prefix="orbit-workspaces-")).resolve()


def configure_workspace_storage(storage: S3ObjectStorage) -> None:
    global _workspace_storage
    with _workspace_hydration_lock:
        _workspace_storage = storage
        _hydrated_workspaces.clear()


def _safe_thread_id(thread_id: str) -> str:
    return "".join(c for c in thread_id if c.isalnum() or c in ("-", "_")) or "default"


def _enforce_inputs_readonly(workspace_dir: Path) -> None:
    """Ensure all files inside the inputs directory are marked read-only."""
    import stat
    inputs_dir = workspace_dir / "inputs"
    if not inputs_dir.exists():
        return
    for item in inputs_dir.rglob("*"):
        if item.is_file():
            try:
                os.chmod(item, stat.S_IREAD | stat.S_IRGRP | stat.S_IROTH)
            except Exception:
                pass


def _is_in_inputs_dir(target_path: Path, thread_id: Optional[str] = None) -> bool:
    """Checks if a resolved path is inside the read-only 'inputs' directory."""
    ws_dir = get_workspace_dir(thread_id)
    try:
        rel = target_path.resolve().relative_to(ws_dir)
        return bool(rel.parts and rel.parts[0].lower() == "inputs")
    except ValueError:
        return False


def save_upload_to_workspace_inputs(
    thread_id: str,
    filename: str,
    content: bytes,
) -> Path:
    """
    Saves an uploaded user file into the thread workspace's read-only 'inputs/' directory.
    Sets read-only filesystem attributes and durably syncs to Tigris object storage.
    """
    import stat
    ws_dir = get_workspace_dir(thread_id)
    inputs_dir = ws_dir / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)

    safe_name = Path(filename).name
    if not safe_name or safe_name in {".", ".."}:
        safe_name = "upload"
    target_file = (inputs_dir / safe_name).resolve()

    if target_file.exists():
        try:
            os.chmod(target_file, stat.S_IWRITE | stat.S_IREAD)
        except Exception:
            pass

    target_file.write_bytes(content)

    try:
        os.chmod(target_file, stat.S_IREAD | stat.S_IRGRP | stat.S_IROTH)
    except Exception:
        pass

    sync_workspace_dir(thread_id)
    return target_file


def copy_file(
    source_path: str,
    destination_path: str,
    thread_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Copies a file within the workspace, e.g. from the read-only 'inputs/' folder
    to the main working directory so it can be edited or modified.
    """
    import shutil
    import stat
    try:
        ws_dir = get_workspace_dir(thread_id)
        source = resolve_safe_path(source_path, thread_id)
        if not source.exists():
            return {"error": f"Source file '{source_path}' does not exist."}
        if not source.is_file():
            return {"error": f"Source path '{source_path}' is not a file."}

        destination = resolve_safe_path(destination_path, thread_id)
        if _is_in_inputs_dir(destination, thread_id):
            return {
                "error": "Cannot copy into the 'inputs' folder because it is strictly read-only."
            }

        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            try:
                os.chmod(destination, stat.S_IWRITE | stat.S_IREAD)
            except Exception:
                pass

        shutil.copy2(source, destination)

        # Ensure destination is writable so the agent can modify it
        try:
            os.chmod(destination, stat.S_IWRITE | stat.S_IREAD)
        except Exception:
            pass

        sync_workspace_dir(thread_id)
        metadata = _workspace_file_metadata(destination, ws_dir)
        return {
            "status": "copied",
            "source": source_path,
            "destination": destination_path,
            "message": f"Successfully copied '{source_path}' to '{destination_path}'. The file is now editable.",
            "file": metadata,
        }
    except Exception as e:
        return {"error": f"Failed to copy '{source_path}' to '{destination_path}': {str(e)}"}


def get_workspace_dir(thread_id: Optional[str] = None) -> Path:
    """Returns and creates the isolated workspace directory for a thread with inputs folder."""
    tid = thread_id or active_thread_id_var.get()
    safe_tid = _safe_thread_id(tid)
    target_dir = (WORKSPACE_ROOT / safe_tid).resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    inputs_dir = target_dir / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    if _workspace_storage is not None and safe_tid not in _hydrated_workspaces:
        with _workspace_hydration_lock:
            if safe_tid not in _hydrated_workspaces:
                _workspace_storage.download_prefix(
                    f"workspaces/{safe_tid}", target_dir
                )
                _hydrated_workspaces.add(safe_tid)
                inputs_dir.mkdir(parents=True, exist_ok=True)
                _enforce_inputs_readonly(target_dir)
    return target_dir


def sync_workspace_dir(thread_id: Optional[str] = None) -> None:
    if _workspace_storage is None:
        return
    tid = thread_id or active_thread_id_var.get()
    safe_tid = _safe_thread_id(tid)
    workspace_dir = get_workspace_dir(tid)
    _workspace_storage.sync_directory(workspace_dir, f"workspaces/{safe_tid}")


def _relative_workspace_path(path: Path, thread_id: str) -> str:
    return path.relative_to(get_workspace_dir(thread_id)).as_posix()


def _is_internal_or_temporary(path: Path, root: Path) -> bool:
    relative = path.relative_to(root)
    if any(part.lower() in _IGNORED_WORKSPACE_DIRS or part.startswith(".") for part in relative.parts):
        return True
    return path.name.endswith("~") or path.suffix.lower() in _TEMP_FILE_SUFFIXES


def _artifact_kind(path: Path) -> Optional[str]:
    ext = path.suffix.lower()
    mime_type = mimetypes.guess_type(path.name)[0] or ""
    if ext not in _ARTIFACT_EXTENSIONS:
        if mime_type.startswith("image/"):
            return "image"
        if mime_type.startswith(("audio/", "video/")):
            return "media"
        return {
            "text/html": "application",
            "text/javascript": "code",
            "text/typescript": "code",
            "text/markdown": "document",
            "application/pdf": "pdf",
            "application/vnd.openxmlformats-officedocument.presentationml.presentation": "presentation",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": "spreadsheet",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "document",
        }.get(mime_type)
    if ext in {".html", ".htm", ".jsx", ".tsx"}:
        return "application"
    if ext in {".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp", ".ico"}:
        return "image"
    if ext == ".pdf":
        return "pdf"
    if ext == ".pptx":
        return "presentation"
    if ext in {".xlsx", ".xls", ".csv"}:
        return "spreadsheet"
    if ext == ".docx":
        return "document"
    if ext in {".py", ".js", ".ts", ".css", ".json", ".yaml", ".yml", ".ipynb"}:
        return "code"
    if ext in {".mp3", ".mp4", ".m4a", ".mov", ".wav", ".webm"}:
        return "media"
    return "document"


def _workspace_file_metadata(path: Path, root: Path) -> Dict[str, Any]:
    stat = path.stat()
    extension = path.suffix.lower()
    fallback_mime_types = {
        ".md": "text/markdown",
        ".ts": "text/typescript",
        ".tsx": "text/typescript",
        ".jsx": "text/javascript",
    }
    return {
        "file_path": path.relative_to(root).as_posix(),
        "name": path.name,
        "extension": extension.lstrip("."),
        "mime_type": mimetypes.guess_type(path.name)[0] or fallback_mime_types.get(extension, "application/octet-stream"),
        "artifact_type": _artifact_kind(path),
        "size_bytes": stat.st_size,
        "modified": stat.st_mtime,
        "modified_ns": stat.st_mtime_ns,
        "file_identity": f"{stat.st_dev}:{stat.st_ino}",
    }


def list_workspace_artifacts(thread_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """Discover previewable workspace outputs without storing a second copy of their content."""
    root = get_workspace_dir(thread_id)
    files: Dict[str, Dict[str, Any]] = {}
    project_dirs: set[Path] = set()
    for current, dirs, names in os.walk(root):
        current_path = Path(current)
        dirs[:] = sorted(
            name for name in dirs
            if name.lower() not in _IGNORED_WORKSPACE_DIRS and not name.startswith(".")
        )
        if "package.json" in names:
            project_dirs.add(current_path)
        for name in names:
            path = current_path / name
            if path.is_symlink() or _is_internal_or_temporary(path, root) or not _artifact_kind(path):
                continue
            try:
                files[path.relative_to(root).as_posix()] = _workspace_file_metadata(path, root)
            except OSError:
                continue

    grouped_paths: set[str] = set()
    artifacts: List[Dict[str, Any]] = []
    for project_dir in sorted(project_dirs, key=lambda value: len(value.parts)):
        if any(project_dir == parent or parent in project_dir.parents for parent in project_dirs if parent != project_dir):
            continue
        relative_dir = project_dir.relative_to(root).as_posix()
        entry_candidates = (
            "dist/index.html", "build/index.html", "out/index.html", "index.html",
            "src/main.tsx", "src/main.jsx", "src/main.ts",
            "src/main.js", "src/index.tsx", "src/index.jsx", "src/index.ts",
            "src/index.js", "app/page.tsx", "app/page.jsx", "pages/index.tsx",
            "pages/index.jsx", "src/app/page.tsx", "index.js", "index.tsx", "index.jsx",
        )
        entry_point = next((candidate for candidate in entry_candidates if (project_dir / candidate).is_file()), None)
        if entry_point is None:
            continue
        entry_path = project_dir / entry_point
        entry_meta = _workspace_file_metadata(entry_path, root)
        member_paths = sorted(
            path for path in files
            if relative_dir == "." or path.startswith(relative_dir + "/")
        )
        member_metadata = [files[path] for path in member_paths]
        source_signature = [
            (path, files[path]["modified_ns"], files[path]["size_bytes"])
            for path in member_paths
        ]
        entry_relative_path = entry_path.relative_to(root).as_posix()
        if entry_relative_path not in files:
            source_signature.append((
                entry_relative_path,
                entry_meta["modified_ns"],
                entry_meta["size_bytes"],
            ))
            member_metadata.append(entry_meta)
        artifacts.append({
            **entry_meta,
            "file_path": entry_path.relative_to(root).as_posix(),
            "project_path": relative_dir,
            "entry_point": entry_path.relative_to(root).as_posix(),
            "name": project_dir.name or "Workspace application",
            "artifact_type": "application",
            "mime_type": entry_meta["mime_type"],
            "is_project": True,
            "source_signature": source_signature,
            "modified_ns": max((item["modified_ns"] for item in member_metadata), default=entry_meta["modified_ns"]),
            "modified": max((item["modified"] for item in member_metadata), default=entry_meta["modified"]),
        })
        grouped_paths.update(
            path for path in files
            if path == (project_dir / "package.json").relative_to(root).as_posix()
            or relative_dir == "."
            or path.startswith(relative_dir + "/")
        )

    for file_path, metadata in files.items():
        if file_path in grouped_paths:
            continue
        if Path(file_path).name.lower() == "index.html":
            directory = Path(file_path).parent
            siblings = [
                candidate for candidate in files
                if Path(candidate).parent == directory and candidate != file_path
            ]
            related = [files[file_path], *(files[path] for path in siblings)]
            metadata = {
                **metadata,
                "artifact_type": "application",
                "entry_point": file_path,
                "is_project": bool(siblings),
                "source_signature": [
                    (path, files[path]["modified_ns"], files[path]["size_bytes"])
                    for path in sorted([file_path, *siblings])
                ],
                "modified_ns": max(item["modified_ns"] for item in related),
            }
            grouped_paths.update([file_path, *siblings])
            artifacts.append(metadata)
            continue
        artifacts.append(metadata)

    artifacts.sort(key=lambda item: (item.get("modified", 0), item["file_path"]), reverse=True)
    return artifacts


def snapshot_workspace_artifacts(thread_id: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    return {
        item["file_path"]: item
        for item in list_workspace_artifacts(thread_id)
    }


def diff_workspace_artifacts(
    thread_id: str,
    previous: Dict[str, Dict[str, Any]],
    current: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Return create/update/delete/move events by comparing workspace file snapshots."""
    removed = set(previous) - set(current)
    added = set(current) - set(previous)
    changes: List[Dict[str, Any]] = []

    old_by_identity: Dict[str, List[str]] = defaultdict(list)
    for path in removed:
        old_by_identity[str(previous[path].get("file_identity", ""))].append(path)
    matched_new: set[str] = set()
    matched_old: set[str] = set()
    for new_path in added:
        identity = str(current[new_path].get("file_identity", ""))
        old_paths = old_by_identity.get(identity, [])
        if identity and len(old_paths) == 1:
            old_path = old_paths[0]
            changes.append({
                "thread_id": thread_id,
                "operation": "moved",
                "old_path": old_path,
                "file_path": new_path,
                "file": current[new_path],
            })
            matched_new.add(new_path)
            matched_old.add(old_path)

    for path in sorted(added - matched_new):
        changes.append({"thread_id": thread_id, "operation": "created", "file_path": path, "file": current[path]})
    for path in sorted(set(previous) & set(current)):
        if (
            previous[path].get("modified_ns") != current[path].get("modified_ns")
            or previous[path].get("size_bytes") != current[path].get("size_bytes")
            or previous[path].get("source_signature") != current[path].get("source_signature")
        ):
            changes.append({"thread_id": thread_id, "operation": "modified", "file_path": path, "file": current[path]})
    for path in sorted(removed - matched_old):
        changes.append({"thread_id": thread_id, "operation": "deleted", "file_path": path, "file": previous[path]})
    return changes


def publish_workspace_change(event: Dict[str, Any]) -> None:
    """Fan out debounced file changes to live workspace event subscribers."""
    thread_id = str(event.get("thread_id") or "")
    if not thread_id:
        return
    signature = (
        thread_id,
        event.get("operation"),
        event.get("old_path"),
        event.get("file_path"),
        (event.get("file") or {}).get("modified_ns"),
        (event.get("file") or {}).get("size_bytes"),
    )
    now = time.monotonic()
    with _workspace_change_lock:
        if now - _recent_workspace_changes.get(signature, 0) < 5.0:
            return
        _recent_workspace_changes[signature] = now
        for stale_key, timestamp in list(_recent_workspace_changes.items()):
            if now - timestamp > 10:
                del _recent_workspace_changes[stale_key]
        subscribers = tuple(_workspace_change_subscribers.get(thread_id, ()))
    if "file" in event:
        event = {
            **event,
            "file": {
                key: value
                for key, value in event["file"].items()
                if key not in {"file_identity", "source_signature", "modified_ns"}
            },
        }
    event = {**event, "timestamp": time.time()}
    for subscriber in subscribers:
        try:
            subscriber.put_nowait(event)
        except queue.Full:
            logger.warning("Workspace event subscriber queue is full for thread %s", thread_id)


def subscribe_workspace_changes(thread_id: str) -> queue.Queue:
    subscriber: queue.Queue = queue.Queue(maxsize=100)
    with _workspace_change_lock:
        _workspace_change_subscribers[thread_id].add(subscriber)
    return subscriber


def unsubscribe_workspace_changes(thread_id: str, subscriber: queue.Queue) -> None:
    with _workspace_change_lock:
        subscribers = _workspace_change_subscribers.get(thread_id)
        if subscribers:
            subscribers.discard(subscriber)
            if not subscribers:
                _workspace_change_subscribers.pop(thread_id, None)


def _emit_workspace_file_change(
    file_path: Path,
    thread_id: Optional[str],
    operation: str,
) -> None:
    tid = thread_id or active_thread_id_var.get()
    root = get_workspace_dir(tid)
    if _is_internal_or_temporary(file_path, root) or not _artifact_kind(file_path):
        return
    try:
        metadata = _workspace_file_metadata(file_path, root)
    except OSError:
        metadata = {"file_path": file_path.relative_to(root).as_posix()}
    publish_workspace_change({
        "thread_id": tid,
        "operation": operation,
        "file_path": metadata["file_path"],
        "file": metadata,
    })


def create_or_update_file(
    file_path: str,
    content: str,
    operation: Literal["create", "replace", "append", "edit"] = "create",
    find_text: Optional[str] = None,
    replacement: Optional[str] = None,
    content_encoding: Literal["text", "base64"] = "text",
    thread_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Atomically create or update a UTF-8 or base64-encoded workspace file."""
    try:
        if operation not in {"create", "replace", "append", "edit"}:
            return {"error": "operation must be 'create', 'replace', 'append', or 'edit'."}
        if content_encoding not in {"text", "base64"}:
            return {"error": "content_encoding must be 'text' or 'base64'."}
        target = resolve_safe_path(file_path, thread_id)
        if _is_in_inputs_dir(target, thread_id):
            return {
                "error": (
                    "The 'inputs' folder is read-only. User uploads cannot be modified directly. "
                    "Please copy the file to the main workspace (e.g. using action='copy') to edit or modify it."
                )
            }
        exists = target.is_file()
        if operation == "create" and target.exists():
            return {"error": f"File '{file_path}' already exists; use replace, append, or edit."}
        if operation in {"replace", "append", "edit"} and not exists:
            return {"error": f"File '{file_path}' does not exist; use create to make it."}
        if operation == "edit":
            if not find_text:
                return {"error": "find_text is required for edit operations."}
            if content_encoding != "text":
                return {"error": "Edit operations support text content only."}
            current_text = target.read_text(encoding="utf-8")
            if find_text not in current_text:
                return {"error": f"find_text was not found in '{file_path}'."}
            next_content: str | bytes = current_text.replace(find_text, replacement or "")
            occurrences = current_text.count(find_text)
        elif content_encoding == "base64":
            if operation == "append":
                return {"error": "Base64 content cannot be appended; replace the complete file instead."}
            next_content = base64.b64decode(content, validate=True)
            occurrences = 0
        elif content_encoding == "text":
            if operation == "append":
                next_content = target.read_bytes() + content.encode("utf-8")
            else:
                next_content = content
            occurrences = 0
        else:
            return {"error": "content_encoding must be 'text' or 'base64'."}

        target.parent.mkdir(parents=True, exist_ok=True)
        temp_path = target.with_name(f".{target.name}.{os.getpid()}.{threading.get_ident()}.tmp")
        try:
            if isinstance(next_content, bytes):
                temp_path.write_bytes(next_content)
            else:
                temp_path.write_text(next_content, encoding="utf-8")
            if operation == "create":
                try:
                    os.link(temp_path, target)
                except FileExistsError:
                    return {"error": f"File '{file_path}' already exists; use replace, append, or edit."}
            else:
                os.replace(temp_path, target)
        finally:
            temp_path.unlink(missing_ok=True)

        sync_workspace_dir(thread_id)
        metadata = _workspace_file_metadata(target, get_workspace_dir(thread_id))
        if _agent_file_write_listener:
            try:
                _agent_file_write_listener(str(target))
                _agent_file_write_listener(file_path)
            except Exception:
                logger.exception("Legacy workspace write listener failed for %s", file_path)
        _emit_workspace_file_change(
            target,
            thread_id,
            "created" if not exists else "modified",
        )
        return {
            "status": "created" if not exists else "updated",
            "operation": operation,
            **{
                key: value
                for key, value in metadata.items()
                if key not in {"file_identity", "modified_ns"}
            },
            "occurrences_replaced": occurrences if operation == "edit" else None,
            "message": f"Successfully {('created' if not exists else 'updated')} {file_path} ({metadata['size_bytes']} bytes).",
        }
    except (ValueError, OSError, UnicodeDecodeError, base64.binascii.Error) as e:
        return {"error": f"Failed to write '{file_path}': {str(e)}"}


class WorkspacePollingWatcher:
    """Cross-platform, debounced workspace watcher for changes outside tool writes."""

    def __init__(self, interval_seconds: float = 1.0, stable_scans: int = 2):
        self.interval_seconds = interval_seconds
        self.stable_scans = stable_scans
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._snapshots: Dict[str, Dict[str, Dict[str, Any]]] = {}
        self._pending: Dict[tuple, tuple[Dict[str, Any], int]] = {}

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._snapshots = {}
        for workspace in WORKSPACE_ROOT.iterdir() if WORKSPACE_ROOT.exists() else ():
            if workspace.is_dir():
                self._snapshots[workspace.name] = snapshot_workspace_artifacts(workspace.name)
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="workspace-file-watcher", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3)
        self._thread = None

    def _run(self) -> None:
        while not self._stop_event.wait(self.interval_seconds):
            try:
                self._scan()
            except Exception:
                logger.exception("Workspace file watcher scan failed")

    def _scan(self) -> None:
        WORKSPACE_ROOT.mkdir(parents=True, exist_ok=True)
        thread_ids = {entry.name for entry in WORKSPACE_ROOT.iterdir() if entry.is_dir()}
        thread_ids.update(self._snapshots)
        for thread_id in thread_ids:
            current = snapshot_workspace_artifacts(thread_id)
            previous = self._snapshots.setdefault(thread_id, {})
            changes = diff_workspace_artifacts(thread_id, previous, current)
            stable_changes = []
            observed_keys = set()
            for change in changes:
                key = (thread_id, change["operation"], change.get("old_path"), change.get("file_path"))
                observed_keys.add(key)
                last_change, seen_count = self._pending.get(key, (change, 0))
                if last_change.get("file") == change.get("file"):
                    seen_count += 1
                else:
                    last_change, seen_count = change, 1
                if seen_count >= self.stable_scans:
                    stable_changes.append(change)
                    self._pending.pop(key, None)
                else:
                    self._pending[key] = (last_change, seen_count)
            for pending_key in list(self._pending):
                if pending_key[0] == thread_id and pending_key not in observed_keys:
                    self._pending.pop(pending_key, None)
            for change in stable_changes:
                publish_workspace_change(change)
                if change["operation"] == "moved":
                    previous.pop(change["old_path"], None)
                    previous[change["file_path"]] = current[change["file_path"]]
                elif change["operation"] == "deleted":
                    previous.pop(change["file_path"], None)
                else:
                    previous[change["file_path"]] = current[change["file_path"]]


def start_workspace_watcher() -> None:
    global _workspace_watcher
    if _workspace_watcher is None:
        _workspace_watcher = WorkspacePollingWatcher()
    _workspace_watcher.start()


def stop_workspace_watcher() -> None:
    if _workspace_watcher:
        _workspace_watcher.stop()


def resolve_safe_path(file_path: str, thread_id: Optional[str] = None) -> Path:
    """
    Safely resolves a relative path within the thread workspace.
    Raises ValueError if the path escapes the sandbox.
    """
    ws_dir = get_workspace_dir(thread_id)
    # Strip leading slashes to keep it relative
    clean_rel = file_path.lstrip("/\\")
    resolved = (ws_dir / clean_rel).resolve()
    
    try:
        resolved.relative_to(ws_dir)
    except ValueError:
        raise ValueError(f"Access denied: path '{file_path}' traverses outside workspace sandbox.")
    return resolved


def list_files(relative_path: str = ".", thread_id: Optional[str] = None) -> Dict[str, Any]:
    """Lists files and folders inside the workspace sandbox."""
    ws_dir = get_workspace_dir(thread_id)
    target = resolve_safe_path(relative_path, thread_id)
    
    if not target.exists():
        return {"error": f"Path '{relative_path}' does not exist in workspace."}
    
    if target.is_file():
        stat = target.stat()
        return {
            "type": "file",
            "path": str(target.relative_to(ws_dir)).replace("\\", "/"),
            "size_bytes": stat.st_size,
            "modified": stat.st_mtime,
        }
    
    items = []
    for entry in target.iterdir():
        rel = str(entry.relative_to(ws_dir)).replace("\\", "/")
        stat = entry.stat()
        items.append({
            "name": entry.name,
            "path": rel,
            "is_dir": entry.is_dir(),
            "size_bytes": stat.st_size if entry.is_file() else 0,
            "modified": stat.st_mtime,
        })
    
    # Sort directories first, then alphabetically
    items.sort(key=lambda x: (not x["is_dir"], x["name"].lower()))
    return {
        "status": "ok",
        "workspace_root": str(target.relative_to(ws_dir)).replace("\\", "/") or ".",
        "items": items,
        "count": len(items),
    }


def read_file(file_path: str, start_line: int = 1, end_line: int = 120, thread_id: Optional[str] = None) -> Dict[str, Any]:
    """Reads a range of lines from a workspace file."""
    try:
        target = resolve_safe_path(file_path, thread_id)
        if not target.is_file():
            return {"error": f"File '{file_path}' does not exist."}
        
        content = target.read_text(encoding="utf-8", errors="replace")
        all_lines = content.splitlines()
        total_lines = len(all_lines)
        
        # Bounded line range (1-indexed)
        start_idx = max(1, start_line)
        end_idx = min(total_lines, max(start_idx, end_line))
        
        selected_lines = all_lines[start_idx - 1:end_idx]
        formatted = "\n".join(f"{i + start_idx:4d} | {line}" for i, line in enumerate(selected_lines))
        
        return {
            "status": "ok",
            "file_path": file_path,
            "total_lines": total_lines,
            "start_line": start_idx,
            "end_line": end_idx,
            "content": formatted,
            "raw_content": "\n".join(selected_lines),
        }
    except Exception as e:
        return {"error": f"Failed to read '{file_path}': {str(e)}"}


def write_file(file_path: str, content: str, thread_id: Optional[str] = None) -> Dict[str, Any]:
    """Creates or overwrites a file in the workspace sandbox."""
    try:
        target = resolve_safe_path(file_path, thread_id)
        if _is_in_inputs_dir(target, thread_id):
            return {
                "error": (
                    "The 'inputs' folder is read-only. User uploads cannot be modified directly. "
                    "Please copy the file to the main workspace (e.g. using action='copy') to edit or modify it."
                )
            }
        existed = target.is_file()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8", errors="replace")
        sync_workspace_dir(thread_id)
        
        if _agent_file_write_listener:
            try:
                _agent_file_write_listener(str(target))
                _agent_file_write_listener(file_path)
            except Exception:
                logger.exception("Legacy workspace write listener failed for %s", file_path)

        _emit_workspace_file_change(
            target,
            thread_id,
            "modified" if existed else "created",
        )

        lines_count = len(content.splitlines())
        size_bytes = len(content.encode("utf-8"))
        
        return {
            "status": "written",
            "file_path": file_path,
            "lines": lines_count,
            "size_bytes": size_bytes,
            "message": f"Successfully wrote {lines_count} lines ({size_bytes} bytes) to {file_path}",
        }
    except Exception as e:
        return {"error": f"Failed to write '{file_path}': {str(e)}"}


def edit_file(file_path: str, target: str, replacement: str, thread_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Performs deterministic exact-match string replacement on a workspace file.
    Follows Antigravity/Claude exact-string replacement semantics.
    """
    try:
        file_obj = resolve_safe_path(file_path, thread_id)
        if _is_in_inputs_dir(file_obj, thread_id):
            return {
                "error": (
                    "The 'inputs' folder is read-only. User uploads cannot be modified directly. "
                    "Please copy the file to the main workspace (e.g. using action='copy') to edit or modify it."
                )
            }
        if not file_obj.is_file():
            return {"error": f"File '{file_path}' does not exist to edit."}
        
        content = file_obj.read_text(encoding="utf-8", errors="replace")
        
        if target not in content:
            return {
                "error": f"Target string not found in '{file_path}'. Please read the file first to verify exact indentation and line breaks."
            }
        
        count = content.count(target)
        new_content = content.replace(target, replacement)
        file_obj.write_text(new_content, encoding="utf-8", errors="replace")
        sync_workspace_dir(thread_id)
        
        if _agent_file_write_listener:
            try:
                _agent_file_write_listener(str(file_obj))
                _agent_file_write_listener(file_path)
            except Exception:
                logger.exception("Legacy workspace write listener failed for %s", file_path)

        _emit_workspace_file_change(file_obj, thread_id, "modified")

        return {
            "status": "edited",
            "file_path": file_path,
            "occurrences_replaced": count,
            "message": f"Successfully replaced {count} occurrence(s) in {file_path}.",
        }
    except Exception as e:
        return {"error": f"Failed to edit '{file_path}': {str(e)}"}


async def run_code(
    command_or_file: str,
    args: Optional[List[str]] = None,
    timeout_seconds: int = 15,
    thread_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Asynchronously executes a Python script or command inside the workspace directory.
    Guarded with strict timeout, directory sandboxing, and output truncation.
    """
    ws_dir = get_workspace_dir(thread_id)
    _enforce_inputs_readonly(ws_dir)
    cmd_args = args or []
    
    # If the user passed a python file path (e.g. 'main.py' or 'script.py')
    clean_target = command_or_file.strip()
    if clean_target.endswith(".py") or (ws_dir / clean_target).is_file():
        safe_file = resolve_safe_path(clean_target, thread_id)
        executable = sys.executable  # Current python interpreter
        full_command = [executable, str(safe_file)] + cmd_args
    else:
        # Run command directly in workspace shell
        full_command = [sys.executable, "-c", command_or_file] if not clean_target.startswith("python") else [sys.executable] + clean_target.split()[1:]

    start_time = asyncio.get_event_loop().time()
    try:
        proc = await asyncio.create_subprocess_exec(
            *full_command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(ws_dir),
            env={
                **os.environ,
                "PYTHONUNBUFFERED": "1",
                "PYTHONDONTWRITEBYTECODE": "1",
            }
        )
        
        try:
            stdout_data, stderr_data = await asyncio.wait_for(
                proc.communicate(), timeout=float(timeout_seconds)
            )
        except asyncio.TimeoutError:
            try:
                proc.kill()
            except ProcessLookupError:
                pass
            sync_workspace_dir(thread_id)
            return {
                "status": "timeout",
                "exit_code": -1,
                "duration_seconds": timeout_seconds,
                "error": f"Execution timed out after {timeout_seconds}s.",
                "stdout": "",
                "stderr": f"Process killed after {timeout_seconds} seconds.",
            }
        
        duration = round(asyncio.get_event_loop().time() - start_time, 3)
        stdout_str = stdout_data.decode("utf-8", errors="replace")
        stderr_str = stderr_data.decode("utf-8", errors="replace")
        sync_workspace_dir(thread_id)
        
        # Truncate output to prevent LLM context saturation (max 5,000 characters)
        max_chars = 5000
        truncated = False
        if len(stdout_str) > max_chars:
            stdout_str = stdout_str[:max_chars] + f"\n... [Output truncated; total {len(stdout_str)} chars]"
            truncated = True
            
        return {
            "status": "completed" if proc.returncode == 0 else "failed",
            "exit_code": proc.returncode,
            "duration_seconds": duration,
            "stdout": stdout_str,
            "stderr": stderr_str,
            "truncated": truncated,
        }
    except Exception as e:
        return {
            "status": "error",
            "exit_code": -1,
            "duration_seconds": round(asyncio.get_event_loop().time() - start_time, 3),
            "error": str(e),
            "stdout": "",
            "stderr": str(e),
        }


def get_file_preview_data(file_path: str, thread_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Generates rich preview data for any workspace file across all formats:
    - Markdown / Code / Plain text (.md, .py, .js, .json, .txt, etc.)
    - Word Documents (.docx) via mammoth HTML conversion
    - Excel Spreadsheets (.xlsx, .xls) via openpyxl
    - Delimited data (.csv, .tsv) via standard csv parser
    - Presentations (.pptx) via python-pptx
    - PDF Documents (.pdf) via pypdf
    - Video, Audio, and Image media files
    """
    try:
        target = resolve_safe_path(file_path, thread_id)
        if not target.is_file():
            return {"error": f"File '{file_path}' does not exist in workspace."}

        ext = target.suffix.lower()
        stat = target.stat()
        file_size = stat.st_size
        base_name = target.name

        # 1. Word Documents (.docx)
        if ext == ".docx":
            try:
                import mammoth
                with open(target, "rb") as f:
                    res = mammoth.convert_to_html(f)
                    html_content = res.value
                return {
                    "type": "docx",
                    "file_path": file_path,
                    "filename": base_name,
                    "extension": "docx",
                    "size_bytes": file_size,
                    "html": html_content,
                }
            except Exception as e:
                return {
                    "type": "error",
                    "error": f"Failed to parse Word document: {str(e)}",
                    "file_path": file_path,
                }

        # 2. Excel Spreadsheets (.xlsx, .xls)
        elif ext in (".xlsx", ".xls"):
            try:
                import openpyxl
                wb = openpyxl.load_workbook(target, data_only=True)
                sheets = []
                for s_name in wb.sheetnames[:10]:
                    ws = wb[s_name]
                    rows = []
                    for row in ws.iter_rows(max_row=250, values_only=True):
                        clean_row = [str(cell) if cell is not None else "" for cell in row]
                        if any(clean_row):
                            rows.append(clean_row)
                    sheets.append({"sheet_name": s_name, "rows": rows})
                return {
                    "type": "excel",
                    "file_path": file_path,
                    "filename": base_name,
                    "extension": ext.lstrip("."),
                    "size_bytes": file_size,
                    "sheets": sheets,
                }
            except Exception as e:
                return {
                    "type": "error",
                    "error": f"Failed to parse Excel workbook: {str(e)}",
                    "file_path": file_path,
                }

        # 3. Delimited Data (.csv, .tsv)
        elif ext in (".csv", ".tsv"):
            try:
                import csv
                delimiter = "\t" if ext == ".tsv" else ","
                text = target.read_text(encoding="utf-8", errors="replace")
                reader = csv.reader(text.splitlines(), delimiter=delimiter)
                rows = [row for row in reader if any(row)][:300]
                return {
                    "type": "csv",
                    "file_path": file_path,
                    "filename": base_name,
                    "extension": ext.lstrip("."),
                    "size_bytes": file_size,
                    "rows": rows,
                    "raw_text": text,
                }
            except Exception as e:
                return {
                    "type": "error",
                    "error": f"Failed to parse CSV/TSV: {str(e)}",
                    "file_path": file_path,
                }

        # 4. Presentations (.pptx)
        elif ext == ".pptx":
            try:
                from pptx import Presentation
                prs = Presentation(target)
                slides = []
                for idx, slide in enumerate(prs.slides):
                    title = ""
                    text_blocks = []
                    for shape in slide.shapes:
                        if shape.has_text_frame:
                            txt = shape.text.strip()
                            if txt:
                                if shape == slide.shapes.title or (not title and len(txt) < 80):
                                    title = txt
                                else:
                                    text_blocks.append(txt)
                    notes = ""
                    if slide.has_notes_slide and slide.notes_slide.notes_text_frame:
                        notes = slide.notes_slide.notes_text_frame.text.strip()
                    slides.append({
                        "slide_index": idx + 1,
                        "title": title or f"Slide {idx + 1}",
                        "text_blocks": text_blocks,
                        "notes": notes,
                    })
                return {
                    "type": "pptx",
                    "file_path": file_path,
                    "filename": base_name,
                    "extension": "pptx",
                    "size_bytes": file_size,
                    "slides": slides,
                    "total_slides": len(slides),
                }
            except Exception as e:
                return {
                    "type": "error",
                    "error": f"Failed to parse PowerPoint presentation: {str(e)}",
                    "file_path": file_path,
                }

        # 5. PDF Documents (.pdf)
        elif ext == ".pdf":
            try:
                import pypdf
                reader = pypdf.PdfReader(target)
                total_pages = len(reader.pages)
                pages_text = []
                for i in range(min(total_pages, 25)):
                    pages_text.append(reader.pages[i].extract_text() or "")
                return {
                    "type": "pdf",
                    "file_path": file_path,
                    "filename": base_name,
                    "extension": "pdf",
                    "size_bytes": file_size,
                    "total_pages": total_pages,
                    "pages_text": pages_text,
                    "raw_url": f"/threads/{thread_id or 'default'}/workspace/raw?path={file_path}",
                }
            except Exception as e:
                return {
                    "type": "error",
                    "error": f"Failed to parse PDF: {str(e)}",
                    "file_path": file_path,
                }

        # 6. Video Files (.mp4, .webm, .ogg, .mov)
        elif ext in (".mp4", ".webm", ".ogg", ".mov"):
            return {
                "type": "media",
                "category": "video",
                "file_path": file_path,
                "filename": base_name,
                "extension": ext.lstrip("."),
                "size_bytes": file_size,
                "raw_url": f"/threads/{thread_id or 'default'}/workspace/raw?path={file_path}",
            }

        # 7. Audio Files (.mp3, .wav, .m4a, .flac, .aac)
        elif ext in (".mp3", ".wav", ".m4a", ".flac", ".aac"):
            return {
                "type": "media",
                "category": "audio",
                "file_path": file_path,
                "filename": base_name,
                "extension": ext.lstrip("."),
                "size_bytes": file_size,
                "raw_url": f"/threads/{thread_id or 'default'}/workspace/raw?path={file_path}",
            }

        # 8. Images (.png, .jpg, .jpeg, .gif, .svg, .webp, .bmp, .ico)
        elif ext in (".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp", ".bmp", ".ico"):
            svg_content = None
            if ext == ".svg":
                try:
                    svg_content = target.read_text(encoding="utf-8", errors="replace")
                except Exception:
                    pass
            return {
                "type": "media",
                "category": "image",
                "file_path": file_path,
                "filename": base_name,
                "extension": ext.lstrip("."),
                "size_bytes": file_size,
                "svg_content": svg_content,
                "raw_url": f"/threads/{thread_id or 'default'}/workspace/raw?path={file_path}",
            }

        elif ext in (".html", ".htm"):
            return {
                "type": "html",
                "file_path": file_path,
                "filename": base_name,
                "extension": ext.lstrip("."),
                "size_bytes": file_size,
                "content": target.read_text(encoding="utf-8", errors="replace"),
                "serve_url": f"/threads/{thread_id or 'default'}/workspace/serve/{urllib.parse.quote(file_path, safe='/')}",
            }

        # 9. Markdown / Code / Text files (.md, .py, .js, .json, etc.)
        else:
            text = target.read_text(encoding="utf-8", errors="replace")
            lines_count = len(text.splitlines())
            return {
                "type": "text",
                "file_path": file_path,
                "filename": base_name,
                "extension": ext.lstrip(".") or "txt",
                "size_bytes": file_size,
                "lines": lines_count,
                "content": text,
            }
    except Exception as e:
        return {"error": f"Preview error for '{file_path}': {str(e)}"}
