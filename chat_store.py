import copy
import time
import secrets
import json
import re
import uuid
from uuid import uuid4
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Optional, Dict, List

from database import PostgresDatabase
from credential_vault import CredentialVault


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _redact_sensitive_payload(value: Any, key: str = "") -> Any:
    secret_key = re.compile(
        r"(?:secret|token|password|passwd|api[_-]?key|access[_-]?key|"
        r"private[_-]?key|credential|authorization)",
        re.IGNORECASE,
    )
    if secret_key.search(key):
        return "[REDACTED]"
    if isinstance(value, dict):
        result = {}
        for item_key, item_value in value.items():
            if str(item_key).lower() in {"headers", "request_headers", "authorization_headers"}:
                result[str(item_key)] = "[REDACTED]"
            else:
                result[str(item_key)] = _redact_sensitive_payload(item_value, str(item_key))
        return result
    if isinstance(value, list):
        return [_redact_sensitive_payload(item, key) for item in value]
    if isinstance(value, tuple):
        return [_redact_sensitive_payload(item, key) for item in value]
    if isinstance(value, str):
        return re.sub(
            r"(?i)(bearer\s+)[A-Za-z0-9._~+/=-]+",
            r"\1[REDACTED]",
            value,
        )
    return value


def _mask_secret_fields(value: Any, vault: CredentialVault) -> Any:
    secret_key = re.compile(
        r"(?:secret|token|password|passwd|api[_-]?key|access[_-]?key|"
        r"private[_-]?key|credential|authorization)",
        re.IGNORECASE,
    )
    if isinstance(value, dict):
        return {
            str(key): (
                "••••••"
                if isinstance(item, str) and item
                else _mask_secret_fields(item, vault)
            )
            if secret_key.search(str(key))
            else _mask_secret_fields(item, vault)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_mask_secret_fields(item, vault) for item in value]
    return value


_thread_cache: dict[str, tuple[dict[str, Any], float]] = {}
THREAD_CACHE_TTL = 10.0  # seconds


def invalidate_thread_cache(thread_id: Optional[str] = None) -> None:
    """Invalidates the thread cache for a specific thread or all threads."""
    if thread_id is None:
        _thread_cache.clear()
    else:
        to_del = [k for k in _thread_cache if k == thread_id or k.startswith(f"{thread_id}:")]
        for k in to_del:
            _thread_cache.pop(k, None)


class PostgresChatStore:
    def __init__(self, database: PostgresDatabase):
        self.database = database

    @property
    def credential_vault(self) -> CredentialVault:
        # Resolve from the current process environment so tests and key rotation
        # can supply the active key without recreating the store instance.
        return CredentialVault()

    @contextmanager
    def _connection(self) -> Iterator:
        with self.database.connection() as connection:
            yield connection

    def create_thread(
        self,
        thread_id: str | None = None,
        title: str = "New chat",
        user_id: Optional[str] = None,
    ) -> dict[str, Any]:
        now = _utc_now()
        safe_title = (title or "").strip() or "New chat"
        with self._connection() as connection:
            if thread_id:
                connection.execute(
                    """
                    INSERT INTO threads (thread_id, title, created_at, updated_at, user_id)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (thread_id) DO NOTHING
                    """,
                    (thread_id, safe_title, now, now, user_id),
                )
            else:
                while True:
                    thread_id = str(secrets.randbelow(900_000_000_000) + 100_000_000_000)
                    inserted = connection.execute(
                        """
                        INSERT INTO threads (thread_id, title, created_at, updated_at, user_id)
                        VALUES (%s, %s, %s, %s, %s)
                        ON CONFLICT (thread_id) DO NOTHING
                        RETURNING thread_id
                        """,
                        (thread_id, safe_title, now, now, user_id),
                    ).fetchone()
                    if inserted:
                        break
        return {
            "thread_id": thread_id,
            "title": safe_title,
            "created_at": now,
            "updated_at": now,
            "preview": "",
            "message_count": 0,
            "user_id": user_id,
        }

    def cleanup_empty_threads(self) -> int:
        """Removes ghost threads that have no messages, artifacts, or canvases."""
        with self._connection() as connection:
            cursor = connection.execute(
                """
                DELETE FROM threads
                WHERE parent_thread_id IS NULL
                  AND (SELECT COUNT(*) FROM messages WHERE thread_id = threads.thread_id) = 0
                  AND (SELECT COUNT(*) FROM artifacts WHERE thread_id = threads.thread_id) = 0
                  AND (SELECT COUNT(*) FROM canvases WHERE thread_id = threads.thread_id) = 0
                """
            )
            return cursor.rowcount

    def list_threads(
        self,
        include_empty: bool = False,
        user_id: Optional[str] = None,
        limit: Optional[int] = None,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        with self._connection() as connection:
            where_clauses: list[str] = []
            params: list[Any] = []
            if user_id is not None:
                where_clauses.append("t.user_id = %s")
                params.append(user_id)
            if not include_empty:
                where_clauses.append(
                    """(
                        t.parent_thread_id IS NOT NULL
                        OR EXISTS (SELECT 1 FROM messages m WHERE m.thread_id = t.thread_id AND m.is_active = 1)
                        OR EXISTS (SELECT 1 FROM artifacts a WHERE a.thread_id = t.thread_id)
                        OR EXISTS (SELECT 1 FROM canvases c WHERE c.thread_id = t.thread_id)
                    )"""
                )
            where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
            page_sql = ""
            if limit is not None:
                safe_limit = max(1, min(int(limit), 101))
                safe_offset = max(0, int(offset))
                page_sql = " LIMIT %s OFFSET %s"
                params.extend((safe_limit, safe_offset))
            rows = connection.execute(
                f"""
                SELECT t.thread_id, t.title, t.created_at, t.updated_at,
                    t.parent_thread_id, t.branch_from_message_id, t.branch_label,
                    t.user_id,
                    parent.title AS parent_title,
                    (SELECT content FROM messages m
                     WHERE m.thread_id = t.thread_id AND m.is_active = 1
                     ORDER BY m.id DESC LIMIT 1) AS preview,
                    (SELECT COUNT(*) FROM messages m
                     WHERE m.thread_id = t.thread_id AND m.is_active = 1) AS message_count
                FROM threads t
                LEFT JOIN threads parent ON parent.thread_id = t.parent_thread_id
                {where_sql}
                ORDER BY t.updated_at DESC, t.id DESC
                {page_sql}
                """,
                params,
            ).fetchall()
        return [dict(row) for row in rows]

    def get_thread(
        self,
        thread_id: str,
        user_id: Optional[str] = None,
        *,
        limit: Optional[int] = None,
        before_id: Optional[int] = None,
        include_extras: bool = True,
    ) -> Optional[dict[str, Any]]:
        page_key = f"{limit or 'all'}:{before_id or 'latest'}:{int(include_extras)}"
        cache_key = f"{thread_id}:{user_id or ''}:{page_key}"
        now_ts = time.time()
        if cache_key in _thread_cache:
            data, expires_at = _thread_cache[cache_key]
            if now_ts < expires_at:
                return copy.deepcopy(data)
            _thread_cache.pop(cache_key, None)

        with self._connection() as connection:
            query = """
                SELECT t.thread_id, t.title, t.created_at, t.updated_at,
                    t.parent_thread_id, t.branch_from_message_id, t.branch_label,
                    t.user_id,
                    parent.title AS parent_title
                FROM threads t
                LEFT JOIN threads parent ON parent.thread_id = t.parent_thread_id
                WHERE t.thread_id = %s
            """
            params: list[Any] = [thread_id]
            if user_id is not None:
                query += " AND t.user_id = %s"
                params.append(user_id)
            thread = connection.execute(query, params).fetchone()
            if thread is None:
                return None
            message_query = """
                SELECT id, role, content, created_at, selected_version, failed,
                    tools_json, parts_json
                FROM messages
                WHERE thread_id = %s AND is_active = 1
            """
            message_params: list[Any] = [thread_id]
            if before_id is not None:
                message_query += " AND id < %s"
                message_params.append(int(before_id))
            message_query += " ORDER BY id DESC"
            if limit is not None:
                safe_limit = max(1, min(int(limit), 100))
                message_query += " LIMIT %s"
                message_params.append(safe_limit + 1)
            messages = connection.execute(message_query, message_params).fetchall()
            has_more_messages = limit is not None and len(messages) > safe_limit
            if limit is not None:
                messages = messages[:safe_limit]
            messages = list(reversed(messages))
            message_ids = [int(row["id"]) for row in messages]
            id_slots = ", ".join(["%s"] * len(message_ids))

            if message_ids:
                variants_query = """
                    SELECT v.user_message_id, v.assistant_message_id, v.version_index,
                        v.user_content, v.assistant_content, v.created_at, v.failed,
                        v.tools_json, v.parts_json,
                        u.selected_version
                    FROM turn_versions v
                    JOIN messages u ON u.id = v.user_message_id
                    WHERE u.thread_id = %s AND u.is_active = 1
                """
                variants_params: list[Any] = [thread_id]
                if limit is not None:
                    variants_query += f" AND u.id IN ({id_slots})"
                    variants_params.extend(message_ids)
                variants_query += " ORDER BY v.user_message_id, v.version_index"
                variants = connection.execute(variants_query, variants_params).fetchall()

                attachment_query = """
                    SELECT attachment_id, message_id, filename, content_type, size_bytes
                    FROM attachments
                    WHERE thread_id = %s
                """
                attachment_params: list[Any] = [thread_id]
                if limit is not None:
                    attachment_query += f" AND message_id IN ({id_slots})"
                    attachment_params.extend(message_ids)
                attachment_query += " ORDER BY created_at, attachment_id"
                attachment_rows = connection.execute(attachment_query, attachment_params).fetchall()
            else:
                variants = []
                attachment_rows = []

            if include_extras:
                artifact_query = """
                    SELECT artifact_id, thread_id, turn_id, title, artifact_type, language, content, created_at
                    FROM artifacts
                    WHERE thread_id = %s
                """
                artifact_params: list[Any] = [thread_id]
                selected_user_ids = [int(row["id"]) for row in messages if row["role"] == "user"]
                if limit is not None:
                    if selected_user_ids:
                        artifact_slots = ", ".join(["%s"] * len(selected_user_ids))
                        artifact_query += f" AND turn_id IN ({artifact_slots})"
                        artifact_params.extend(selected_user_ids)
                    else:
                        artifact_query += " AND 1 = 0"
                artifact_query += " ORDER BY id ASC"
                artifact_rows = connection.execute(artifact_query, artifact_params).fetchall()
                canvas_rows = connection.execute(
                    """
                    SELECT canvas_id, thread_id, title, spec_json, created_at
                    FROM canvases
                    WHERE thread_id = %s
                    ORDER BY id ASC
                    """,
                    (thread_id,),
                ).fetchall()
            else:
                artifact_rows = []
                canvas_rows = []

            attachments_by_message: dict[int, list[dict[str, Any]]] = {}
            for attachment in attachment_rows:
                if attachment["message_id"] is None:
                    continue
                attachments_by_message.setdefault(attachment["message_id"], []).append(
                    {
                        "attachment_id": attachment["attachment_id"],
                        "filename": attachment["filename"],
                        "content_type": attachment["content_type"],
                        "size_bytes": attachment["size_bytes"],
                    }
                )
            variants_by_user: dict[int, list[dict[str, Any]]] = {}
            variants_by_assistant: dict[int, list[dict[str, Any]]] = {}
            selected_versions: dict[int, int] = {}
            user_by_assistant: dict[int, int] = {}
            for variant in variants:
                raw_v_tools = variant["tools_json"] if "tools_json" in variant.keys() else None
                raw_v_parts = variant["parts_json"] if "parts_json" in variant.keys() else None
                try:
                    v_tools = json.loads(raw_v_tools) if raw_v_tools else []
                except Exception:
                    v_tools = []
                try:
                    v_parts = json.loads(raw_v_parts) if raw_v_parts else []
                except Exception:
                    v_parts = []
                item = {
                    "version": variant["version_index"],
                    "user_content": variant["user_content"],
                    "assistant_content": variant["assistant_content"],
                    "created_at": variant["created_at"],
                    "failed": bool(variant["failed"]),
                    "tools": v_tools,
                    "parts": v_parts,
                }
                user_id_v = variant["user_message_id"]
                assistant_id = variant["assistant_message_id"]
                variants_by_user.setdefault(user_id_v, []).append(item)
                variants_by_assistant.setdefault(assistant_id, []).append(item)
                selected_versions[user_id_v] = variant["selected_version"]
                user_by_assistant[assistant_id] = user_id_v
            thread_artifacts = [dict(row) for row in artifact_rows]
            thread_canvases = []
            for row in canvas_rows:
                try:
                    spec = json.loads(row["spec_json"])
                except Exception:
                    spec = row["spec_json"]
                thread_canvases.append({
                    "canvas_id": row["canvas_id"],
                    "thread_id": row["thread_id"],
                    "title": row["title"],
                    "spec": spec,
                    "created_at": row["created_at"],
                })
            # Workspace scans are only needed to reconstruct legacy tool events.
            # The fast history API sets include_extras=False, so do not touch the
            # filesystem on the chat's critical loading path.
            workspace_files: dict[str, Path] = {}
            workspace_file_pattern = None
            if include_extras:
                workspace_dir = Path("data/workspaces") / thread_id
                if workspace_dir.is_dir():
                    workspace_files = {
                        entry.name.casefold(): entry
                        for entry in workspace_dir.iterdir()
                        if entry.is_file() and entry.name != "__pycache__"
                    }
                workspace_file_pattern = (
                    re.compile(
                        r"\b(?:"
                        + "|".join(
                            re.escape(name)
                            for name in sorted(workspace_files, key=len, reverse=True)
                        )
                        + r")\b",
                        re.IGNORECASE,
                    )
                    if workspace_files
                    else None
                )
            loaded_messages = []
            messages_to_backfill: list[tuple[str, str, int]] = []
            latest_user_id: Optional[int] = None
            for row in messages:
                message = dict(row)
                raw_tools = message.pop("tools_json", None)
                raw_parts = message.pop("parts_json", None)
                try:
                    message["tools"] = json.loads(raw_tools) if raw_tools else []
                except Exception:
                    message["tools"] = []
                try:
                    message["parts"] = json.loads(raw_parts) if raw_parts else []
                except Exception:
                    message["parts"] = []

                message["attachments"] = attachments_by_message.get(row["id"], [])
                if row["role"] == "user":
                    latest_user_id = row["id"]
                    message["turn_id"] = row["id"]
                    message["variants"] = variants_by_user.get(row["id"], [])
                    message["selected_version"] = selected_versions.get(row["id"], 1)
                else:
                    matching_user = user_by_assistant.get(row["id"], latest_user_id)
                    if matching_user is not None:
                        message["turn_id"] = matching_user
                        message["variants"] = variants_by_assistant.get(row["id"], [])
                        message["selected_version"] = selected_versions.get(matching_user, 1)
                    latest_user_id = None

                    # Smart fallback for historical assistant messages created before persistence
                    if include_extras and not message["tools"] and not message["parts"]:
                        synthesized_tools: list[dict[str, Any]] = []
                        # 1. Check workspace files mentioned in content
                        if workspace_file_pattern:
                            matched_names = {
                                match.group().casefold()
                                for match in workspace_file_pattern.finditer(message["content"])
                            }
                            for matched_name in matched_names:
                                fpath = workspace_files[matched_name]
                                fname = fpath.name
                                if fname == "app.py" and "app.py" not in message["content"]:
                                    continue
                                try:
                                    size_bytes = fpath.stat().st_size
                                except OSError:
                                    size_bytes = 0
                                synthesized_tools.append({
                                    "id": f"hist-file-{message['id']}-{fname}",
                                    "name": "write_workspace_file",
                                    "args": {"file_path": fname},
                                    "output": json.dumps({
                                        "status": "written",
                                        "file_path": fname,
                                        "size_bytes": size_bytes,
                                        "message": f"Successfully wrote {size_bytes} bytes to {fname}",
                                    }),
                                    "status": "done",
                                })
                        # 2. Check artifacts for this turn
                        if matching_user is not None:
                            for art in thread_artifacts:
                                if art.get("turn_id") == matching_user:
                                    synthesized_tools.append({
                                        "id": f"hist-art-{art['artifact_id']}",
                                        "name": "create_artifact",
                                        "args": {
                                            "title": art["title"],
                                            "artifact_type": art["artifact_type"],
                                            "language": art.get("language") or "",
                                            "content": art["content"],
                                        },
                                        "output": json.dumps(art),
                                        "status": "done",
                                    })

                        if synthesized_tools:
                            message["tools"] = synthesized_tools
                            synth_parts: list[dict[str, Any]] = [
                                {"type": "tool", "tool": t} for t in synthesized_tools
                            ]
                            if message["content"]:
                                synth_parts.append({"type": "text", "text": message["content"]})
                            message["parts"] = synth_parts
                            # Backfill into DB so future requests are pre-populated
                            messages_to_backfill.append(
                                (
                                    json.dumps(message["tools"], ensure_ascii=False),
                                    json.dumps(message["parts"], ensure_ascii=False),
                                    message["id"],
                                )
                            )

                loaded_messages.append(message)
            if messages_to_backfill:
                try:
                    connection.executemany(
                        "UPDATE messages SET tools_json = %s, parts_json = %s WHERE id = %s",
                        messages_to_backfill,
                    )
                except Exception:
                    pass

        result = {
            "thread": dict(thread),
            "messages": loaded_messages,
            "canvases": thread_canvases,
            "artifacts": thread_artifacts,
            "has_more_messages": bool(has_more_messages),
        }
        _thread_cache[cache_key] = (result, time.time() + THREAD_CACHE_TTL)
        return copy.deepcopy(result)

    def get_thread_updates(
        self, thread_id: str, after_id: int, user_id: Optional[str] = None
    ) -> Optional[dict[str, Any]]:
        """Return only newly created active messages for lightweight cross-tab sync."""
        with self._connection() as connection:
            owner_query = "SELECT 1 FROM threads WHERE thread_id = %s"
            owner_params: list[Any] = [thread_id]
            if user_id is not None:
                owner_query += " AND user_id = %s"
                owner_params.append(user_id)
            if connection.execute(owner_query, owner_params).fetchone() is None:
                return None

            rows = connection.execute(
                """
                SELECT id, role, content, created_at, selected_version, failed,
                    tools_json, parts_json
                FROM messages
                WHERE thread_id = %s AND is_active = 1 AND id > %s
                ORDER BY id ASC
                LIMIT 101
                """,
                (thread_id, max(0, int(after_id))),
            ).fetchall()
            has_more = len(rows) > 100
            rows = rows[:100]
            message_ids = [int(row["id"]) for row in rows]
            attachments_by_message: dict[int, list[dict[str, Any]]] = {}
            if message_ids:
                slots = ", ".join(["%s"] * len(message_ids))
                attachments = connection.execute(
                    f"""
                    SELECT attachment_id, message_id, filename, content_type, size_bytes
                    FROM attachments
                    WHERE thread_id = %s AND message_id IN ({slots})
                    ORDER BY created_at, attachment_id
                    """,
                    [thread_id, *message_ids],
                ).fetchall()
                for attachment in attachments:
                    attachments_by_message.setdefault(attachment["message_id"], []).append(
                        {
                            "attachment_id": attachment["attachment_id"],
                            "filename": attachment["filename"],
                            "content_type": attachment["content_type"],
                            "size_bytes": attachment["size_bytes"],
                        }
                    )

        messages = []
        for row in rows:
            item = dict(row)
            raw_tools = item.pop("tools_json", None)
            raw_parts = item.pop("parts_json", None)
            try:
                item["tools"] = json.loads(raw_tools) if raw_tools else []
            except Exception:
                item["tools"] = []
            try:
                item["parts"] = json.loads(raw_parts) if raw_parts else []
            except Exception:
                item["parts"] = []
            item["attachments"] = attachments_by_message.get(row["id"], [])
            messages.append(item)
        return {"messages": messages, "has_more": has_more}

    def thread_exists(
        self, thread_id: str, user_id: Optional[str] = None
    ) -> bool:
        query = "SELECT 1 FROM threads WHERE thread_id = %s"
        params: list[Any] = [thread_id]
        if user_id is not None:
            query += " AND user_id = %s"
            params.append(user_id)
        with self._connection() as connection:
            row = connection.execute(query, params).fetchone()
        return row is not None

    def add_canvas(self, thread_id: str, title: str, spec: Any) -> dict[str, Any]:
        canvas_id = f"canvas-{uuid4().hex[:10]}"
        now = _utc_now()
        spec_json = json.dumps(spec, ensure_ascii=False) if isinstance(spec, (dict, list)) else str(spec)
        with self._connection() as connection:
            thread = connection.execute(
                "SELECT id FROM threads WHERE thread_id = %s", (thread_id,)
            ).fetchone()
            if thread is None:
                raise KeyError(thread_id)
            connection.execute(
                """
                INSERT INTO canvases (canvas_id, thread_id, title, spec_json, created_at)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (canvas_id, thread_id, title, spec_json, now),
            )
        invalidate_thread_cache(thread_id)
        return {
            "canvas_id": canvas_id,
            "thread_id": thread_id,
            "title": title,
            "spec": spec if isinstance(spec, (dict, list)) else json.loads(spec_json),
            "created_at": now,
        }

    def list_canvases(self, thread_id: str) -> list[dict[str, Any]]:
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT canvas_id, thread_id, title, spec_json, created_at
                FROM canvases
                WHERE thread_id = %s
                ORDER BY id ASC
                """,
                (thread_id,),
            ).fetchall()
            result = []
            for row in rows:
                try:
                    spec = json.loads(row["spec_json"])
                except Exception:
                    spec = row["spec_json"]
                result.append({
                    "canvas_id": row["canvas_id"],
                    "thread_id": row["thread_id"],
                    "title": row["title"],
                    "spec": spec,
                    "created_at": row["created_at"],
                })
            return result

    def add_artifact(
        self,
        thread_id: str,
        title: str,
        artifact_type: str,
        content: str,
        language: Optional[str] = None,
        turn_id: Optional[int] = None,
    ) -> dict[str, Any]:
        artifact_id = f"artifact-{uuid4().hex[:10]}"
        now = _utc_now()
        with self._connection() as connection:
            thread = connection.execute(
                "SELECT id FROM threads WHERE thread_id = %s", (thread_id,)
            ).fetchone()
            if thread is None:
                raise KeyError(thread_id)
            connection.execute(
                """
                INSERT INTO artifacts (artifact_id, thread_id, turn_id, title, artifact_type, language, content, created_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (artifact_id, thread_id, turn_id, title, artifact_type, language, content, now),
            )
        invalidate_thread_cache(thread_id)
        return {
            "artifact_id": artifact_id,
            "thread_id": thread_id,
            "turn_id": turn_id,
            "title": title,
            "artifact_type": artifact_type,
            "language": language,
            "content": content,
            "created_at": now,
        }

    def list_artifacts(self, thread_id: str) -> list[dict[str, Any]]:
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT artifact_id, thread_id, turn_id, title, artifact_type, language, content, created_at
                FROM artifacts
                WHERE thread_id = %s
                ORDER BY id ASC
                """,
                (thread_id,),
            ).fetchall()
            return [dict(row) for row in rows]

    def get_artifact(self, artifact_id: str) -> Optional[dict[str, Any]]:
        with self._connection() as connection:
            row = connection.execute(
                """
                SELECT artifact_id, thread_id, turn_id, title, artifact_type, language, content, created_at
                FROM artifacts
                WHERE artifact_id = %s
                """,
                (artifact_id,),
            ).fetchone()
            return dict(row) if row else None

    def delete_artifact(self, artifact_id: str) -> bool:
        with self._connection() as connection:
            cursor = connection.execute(
                "DELETE FROM artifacts WHERE artifact_id = %s", (artifact_id,)
            )
            invalidate_thread_cache()
            return cursor.rowcount > 0

    def add_message(
        self,
        thread_id: str,
        role: str,
        content: str,
        tools: Optional[list[dict[str, Any]]] = None,
        parts: Optional[list[dict[str, Any]]] = None,
    ) -> dict[str, Any]:
        if role not in {"user", "assistant"}:
            raise ValueError("Message role must be 'user' or 'assistant'.")
        now = _utc_now()
        tools_json = json.dumps(tools, ensure_ascii=False) if tools else None
        parts_json = json.dumps(parts, ensure_ascii=False) if parts else None
        with self._connection() as connection:
            thread = connection.execute(
                "SELECT id FROM threads WHERE thread_id = %s", (thread_id,)
            ).fetchone()
            if thread is None:
                raise KeyError(thread_id)
            cursor = connection.execute(
                """
                INSERT INTO messages (thread_id, role, content, created_at, tools_json, parts_json)
                VALUES (%s, %s, %s, %s, %s, %s)
                RETURNING id
                """,
                (thread_id, role, content, now, tools_json, parts_json),
            )
            message_id = cursor.fetchone()["id"]
            if role == "user":
                count = connection.execute(
                    "SELECT COUNT(*) AS count FROM messages WHERE thread_id = %s",
                    (thread_id,),
                ).fetchone()["count"]
                if count == 1:
                    title = " ".join(content.split())[:72] or "New chat"
                    connection.execute(
                        "UPDATE threads SET title = %s WHERE thread_id = %s",
                        (title, thread_id),
                    )
            connection.execute(
                "UPDATE threads SET updated_at = %s WHERE thread_id = %s",
                (now, thread_id),
            )
        invalidate_thread_cache(thread_id)
        return {
            "id": message_id,
            "role": role,
            "content": content,
            "created_at": now,
            "tools": tools or [],
            "parts": parts or [],
        }

    def add_attachment(
        self,
        thread_id: str,
        attachment_id: str,
        filename: str,
        content_type: str,
        storage_name: str,
        size_bytes: int,
        extracted_text: str,
    ) -> dict[str, Any]:
        now = _utc_now()
        with self._connection() as connection:
            thread = connection.execute(
                "SELECT 1 FROM threads WHERE thread_id = %s", (thread_id,)
            ).fetchone()
            if thread is None:
                raise KeyError(thread_id)
            connection.execute(
                """
                INSERT INTO attachments (
                    attachment_id, thread_id, filename, content_type,
                    storage_name, size_bytes, extracted_text, created_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    attachment_id,
                    thread_id,
                    filename,
                    content_type,
                    storage_name,
                    size_bytes,
                    extracted_text,
                    now,
                ),
            )
        invalidate_thread_cache(thread_id)
        return {
            "attachment_id": attachment_id,
            "filename": filename,
            "content_type": content_type,
            "size_bytes": size_bytes,
        }

    def get_attachments(
        self, thread_id: str, attachment_ids: list[str]
    ) -> list[dict[str, Any]]:
        if not attachment_ids:
            return []
        placeholders = ",".join("%s" for _ in attachment_ids)
        with self._connection() as connection:
            rows = connection.execute(
                f"""
                SELECT attachment_id, thread_id, filename, content_type,
                    storage_name, size_bytes, extracted_text, message_id
                FROM attachments
                WHERE thread_id = %s AND attachment_id IN ({placeholders})
                """,
                (thread_id, *attachment_ids),
            ).fetchall()
        return [dict(row) for row in rows]

    def attach_to_message(
        self, thread_id: str, message_id: int, attachment_ids: list[str]
    ) -> None:
        if not attachment_ids:
            return
        placeholders = ",".join("%s" for _ in attachment_ids)
        with self._connection() as connection:
            cursor = connection.execute(
                f"""
                UPDATE attachments SET message_id = %s
                WHERE thread_id = %s AND message_id IS NULL
                    AND attachment_id IN ({placeholders})
                """,
                (message_id, thread_id, *attachment_ids),
            )
            if cursor.rowcount != len(set(attachment_ids)):
                raise ValueError("One or more uploaded files are no longer available.")

    def add_turn_version(
        self,
        user_message_id: int,
        assistant_message_id: int,
        user_content: str,
        assistant_content: str,
        tools: Optional[list[dict[str, Any]]] = None,
        parts: Optional[list[dict[str, Any]]] = None,
    ) -> None:
        tools_json = json.dumps(tools, ensure_ascii=False) if tools else None
        parts_json = json.dumps(parts, ensure_ascii=False) if parts else None
        with self._connection() as connection:
            connection.execute(
                """
                INSERT INTO turn_versions (
                    user_message_id, assistant_message_id, version_index,
                    user_content, assistant_content, created_at, tools_json, parts_json
                ) VALUES (%s, %s, 1, %s, %s, %s, %s, %s)
                ON CONFLICT (user_message_id, version_index) DO NOTHING
                """,
                (
                    user_message_id,
                    assistant_message_id,
                    user_content,
                    assistant_content,
                    _utc_now(),
                    tools_json,
                    parts_json,
                ),
            )

    def get_turn_context(self, thread_id: str, user_message_id: int) -> dict[str, Any]:
        with self._connection() as connection:
            user = connection.execute(
                """
                SELECT id, content FROM messages
                WHERE id = %s AND thread_id = %s AND role = 'user' AND is_active = 1
                """,
                (user_message_id, thread_id),
            ).fetchone()
            if user is None:
                raise KeyError(user_message_id)
            assistant = connection.execute(
                """
                SELECT id FROM messages
                WHERE thread_id = %s AND role = 'assistant' AND is_active = 1 AND id > %s
                ORDER BY id LIMIT 1
                """,
                (thread_id, user_message_id),
            ).fetchone()
            if assistant is None:
                raise ValueError("The selected user message has no completed response.")
            history = connection.execute(
                """
                SELECT role, content FROM messages
                WHERE thread_id = %s AND is_active = 1 AND id < %s
                ORDER BY id
                """,
                (thread_id, user_message_id),
            ).fetchall()
        return {
            "user_content": user["content"],
            "assistant_message_id": assistant["id"],
            "history": [dict(row) for row in history],
        }

    def save_turn_version(
        self,
        thread_id: str,
        user_message_id: int,
        user_content: str,
        assistant_content: str,
        tools: Optional[list[dict[str, Any]]] = None,
        parts: Optional[list[dict[str, Any]]] = None,
    ) -> dict[str, int]:
        now = _utc_now()
        tools_json = json.dumps(tools, ensure_ascii=False) if tools else None
        parts_json = json.dumps(parts, ensure_ascii=False) if parts else None
        with self._connection() as connection:
            user = connection.execute(
                """
                SELECT id, content FROM messages
                WHERE id = %s AND thread_id = %s AND role = 'user' AND is_active = 1
                """,
                (user_message_id, thread_id),
            ).fetchone()
            if user is None:
                raise KeyError(user_message_id)
            assistant = connection.execute(
                """
                SELECT id, content, tools_json, parts_json FROM messages
                WHERE thread_id = %s AND role = 'assistant' AND is_active = 1 AND id > %s
                ORDER BY id LIMIT 1
                """,
                (thread_id, user_message_id),
            ).fetchone()
            if assistant is None:
                raise ValueError("The selected user message has no completed response.")
            maximum = connection.execute(
                "SELECT MAX(version_index) AS maximum FROM turn_versions WHERE user_message_id = %s",
                (user_message_id,),
            ).fetchone()["maximum"]
            if maximum is None:
                connection.execute(
                    """
                    INSERT INTO turn_versions (
                        user_message_id, assistant_message_id, version_index,
                        user_content, assistant_content, created_at, tools_json, parts_json
                    ) VALUES (%s, %s, 1, %s, %s, %s, %s, %s)
                    """,
                    (
                        user_message_id,
                        assistant["id"],
                        user["content"],
                        assistant["content"],
                        now,
                        assistant["tools_json"],
                        assistant["parts_json"],
                    ),
                )
                maximum = 1
            version_index = maximum + 1
            connection.execute(
                "UPDATE messages SET is_active = 0 WHERE thread_id = %s AND id > %s",
                (thread_id, assistant["id"]),
            )
            connection.execute(
                "UPDATE messages SET content = %s, selected_version = %s WHERE id = %s",
                (user_content, version_index, user_message_id),
            )
            connection.execute(
                "UPDATE messages SET content = %s, tools_json = %s, parts_json = %s WHERE id = %s",
                (assistant_content, tools_json, parts_json, assistant["id"]),
            )
            connection.execute(
                """
                INSERT INTO turn_versions (
                    user_message_id, assistant_message_id, version_index,
                    user_content, assistant_content, created_at, tools_json, parts_json
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    user_message_id,
                    assistant["id"],
                    version_index,
                    user_content,
                    assistant_content,
                    now,
                    tools_json,
                    parts_json,
                ),
            )
            connection.execute(
                "UPDATE threads SET updated_at = %s WHERE thread_id = %s",
                (now, thread_id),
            )
        invalidate_thread_cache(thread_id)
        return {"version": version_index, "version_count": version_index}

    def prepare_turn_version(
        self, thread_id: str, user_message_id: int, user_content: str
    ) -> dict[str, int]:
        now = _utc_now()
        with self._connection() as connection:
            user = connection.execute(
                """
                SELECT id, content FROM messages
                WHERE id = %s AND thread_id = %s AND role = 'user' AND is_active = 1
                """,
                (user_message_id, thread_id),
            ).fetchone()
            if user is None:
                raise KeyError(user_message_id)
            assistant = connection.execute(
                """
                SELECT id, content, tools_json, parts_json FROM messages
                WHERE thread_id = %s AND role = 'assistant' AND is_active = 1 AND id > %s
                ORDER BY id LIMIT 1
                """,
                (thread_id, user_message_id),
            ).fetchone()
            if assistant is None:
                raise ValueError("The selected user message has no completed response.")
            maximum = connection.execute(
                "SELECT MAX(version_index) AS maximum FROM turn_versions WHERE user_message_id = %s",
                (user_message_id,),
            ).fetchone()["maximum"]
            if maximum is None:
                connection.execute(
                    """
                    INSERT INTO turn_versions (
                        user_message_id, assistant_message_id, version_index,
                        user_content, assistant_content, created_at, tools_json, parts_json
                    ) VALUES (%s, %s, 1, %s, %s, %s, %s, %s)
                    """,
                    (
                        user_message_id,
                        assistant["id"],
                        user["content"],
                        assistant["content"],
                        now,
                        assistant["tools_json"],
                        assistant["parts_json"],
                    ),
                )
                maximum = 1
            version_index = maximum + 1
            connection.execute(
                "UPDATE messages SET is_active = 0 WHERE thread_id = %s AND id > %s",
                (thread_id, assistant["id"]),
            )
            connection.execute(
                """
                UPDATE messages
                SET content = %s, selected_version = %s, failed = 0
                WHERE id = %s
                """,
                (user_content, version_index, user_message_id),
            )
            connection.execute(
                "UPDATE messages SET content = '', tools_json = NULL, parts_json = NULL, failed = 0 WHERE id = %s",
                (assistant["id"],),
            )
            connection.execute(
                """
                INSERT INTO turn_versions (
                    user_message_id, assistant_message_id, version_index,
                    user_content, assistant_content, created_at
                ) VALUES (%s, %s, %s, %s, '', %s)
                """,
                (
                    user_message_id,
                    assistant["id"],
                    version_index,
                    user_content,
                    now,
                ),
            )
            connection.execute(
                "UPDATE threads SET updated_at = %s WHERE thread_id = %s",
                (now, thread_id),
            )
            version_count = version_index
        return {"version": version_index, "version_count": version_count}

    def complete_turn_version(
        self,
        thread_id: str,
        user_message_id: int,
        version_index: int,
        assistant_content: str,
        tools: Optional[list[dict[str, Any]]] = None,
        parts: Optional[list[dict[str, Any]]] = None,
    ) -> None:
        tools_json = json.dumps(tools, ensure_ascii=False) if tools else None
        parts_json = json.dumps(parts, ensure_ascii=False) if parts else None
        with self._connection() as connection:
            variant = connection.execute(
                """
                SELECT v.assistant_message_id
                FROM turn_versions v
                JOIN messages u ON u.id = v.user_message_id
                WHERE v.user_message_id = %s AND v.version_index = %s
                    AND u.thread_id = %s AND u.is_active = 1
                """,
                (user_message_id, version_index, thread_id),
            ).fetchone()
            if variant is None:
                raise KeyError((user_message_id, version_index))
            connection.execute(
                """
                UPDATE turn_versions
                SET assistant_content = %s, failed = 0, tools_json = %s, parts_json = %s
                WHERE user_message_id = %s AND version_index = %s
                """,
                (assistant_content, tools_json, parts_json, user_message_id, version_index),
            )
            connection.execute(
                """
                UPDATE messages SET content = %s, failed = 0, tools_json = %s, parts_json = %s
                WHERE id = %s AND is_active = 1
                """,
                (assistant_content, tools_json, parts_json, variant["assistant_message_id"]),
            )
        invalidate_thread_cache(thread_id)

    def fail_turn_version(
        self, thread_id: str, user_message_id: int, version_index: int
    ) -> None:
        with self._connection() as connection:
            variant = connection.execute(
                """
                SELECT v.assistant_message_id
                FROM turn_versions v
                JOIN messages u ON u.id = v.user_message_id
                WHERE v.user_message_id = %s AND v.version_index = %s
                    AND u.thread_id = %s AND u.is_active = 1
                """,
                (user_message_id, version_index, thread_id),
            ).fetchone()
            if variant is None:
                return
            connection.execute(
                """
                UPDATE turn_versions SET failed = 1
                WHERE user_message_id = %s AND version_index = %s
                """,
                (user_message_id, version_index),
            )
            connection.execute(
                """
                UPDATE messages SET failed = 1
                WHERE id = %s AND is_active = 1
                """,
                (variant["assistant_message_id"],),
            )
        invalidate_thread_cache(thread_id)

    def select_turn_version(
        self, thread_id: str, user_message_id: int, version_index: int
    ) -> bool:
        with self._connection() as connection:
            user = connection.execute(
                """
                SELECT id FROM messages
                WHERE id = %s AND thread_id = %s AND role = 'user' AND is_active = 1
                """,
                (user_message_id, thread_id),
            ).fetchone()
            variant = connection.execute(
                """
                SELECT assistant_message_id, user_content, assistant_content, failed,
                    tools_json, parts_json
                FROM turn_versions
                WHERE user_message_id = %s AND version_index = %s
                """,
                (user_message_id, version_index),
            ).fetchone()
            if user is None or variant is None:
                return False
            connection.execute(
                "UPDATE messages SET is_active = 0 WHERE thread_id = %s AND id > %s",
                (thread_id, variant["assistant_message_id"]),
            )
            connection.execute(
                "UPDATE messages SET content = %s, selected_version = %s, failed = 0 WHERE id = %s",
                (variant["user_content"], version_index, user_message_id),
            )
            connection.execute(
                "UPDATE messages SET content = %s, failed = %s, tools_json = %s, parts_json = %s WHERE id = %s",
                (
                    variant["assistant_content"],
                    variant["failed"],
                    variant["tools_json"],
                    variant["parts_json"],
                    variant["assistant_message_id"],
                ),
            )
            connection.execute(
                "UPDATE threads SET updated_at = %s WHERE thread_id = %s",
                (_utc_now(), thread_id),
            )
        invalidate_thread_cache(thread_id)
        return True

    def branch_thread(
        self,
        thread_id: str,
        user_message_id: int,
        version_index: int,
        user_id: Optional[str] = None,
    ) -> Optional[dict[str, Any]]:
        now = _utc_now()
        with self._connection() as connection:
            if user_id is not None:
                owner_row = connection.execute(
                    "SELECT user_id FROM threads WHERE thread_id = %s", (thread_id,)
                ).fetchone()
                if owner_row is None or owner_row["user_id"] != user_id:
                    return None
            active_turn = connection.execute(
                """
                SELECT u.id AS user_id, u.content AS user_content,
                    a.id AS assistant_id, a.content AS assistant_content,
                    a.tools_json AS assistant_tools_json, a.parts_json AS assistant_parts_json
                FROM messages u
                JOIN messages a ON a.thread_id = u.thread_id
                    AND a.role = 'assistant' AND a.is_active = 1 AND a.id > u.id
                WHERE u.id = %s AND u.thread_id = %s AND u.role = 'user' AND u.is_active = 1
                ORDER BY a.id LIMIT 1
                """,
                (user_message_id, thread_id),
            ).fetchone()
            if active_turn is None:
                return None
            existing_version = connection.execute(
                """
                SELECT 1 FROM turn_versions
                WHERE user_message_id = %s AND version_index = %s
                """,
                (user_message_id, version_index),
            ).fetchone()
            if existing_version is None and version_index == 1:
                connection.execute(
                    """
                    INSERT INTO turn_versions (
                        user_message_id, assistant_message_id, version_index,
                        user_content, assistant_content, created_at, tools_json, parts_json
                    ) VALUES (%s, %s, 1, %s, %s, %s, %s, %s)
                    """,
                    (
                        user_message_id,
                        active_turn["assistant_id"],
                        active_turn["user_content"],
                        active_turn["assistant_content"],
                        now,
                        active_turn["assistant_tools_json"],
                        active_turn["assistant_parts_json"],
                    ),
                )
            target = connection.execute(
                """
                SELECT u.id AS user_id, a.id AS assistant_id,
                    v.user_content, v.assistant_content, v.failed,
                    v.tools_json, v.parts_json
                FROM messages u
                JOIN turn_versions v ON v.user_message_id = u.id
                    AND v.version_index = %s
                JOIN messages a ON a.id = v.assistant_message_id
                WHERE u.id = %s AND u.thread_id = %s AND u.is_active = 1
                    AND a.is_active = 1
                """,
                (version_index, user_message_id, thread_id),
            ).fetchone()
            if target is None:
                return None
            source_messages = connection.execute(
                """
                SELECT id, role, content, created_at, selected_version, failed,
                    tools_json, parts_json
                FROM messages
                WHERE thread_id = %s AND is_active = 1 AND id <= %s
                ORDER BY id
                """,
                (thread_id, target["assistant_id"]),
            ).fetchall()
            new_thread_id = str(secrets.randbelow(900_000_000_000) + 100_000_000_000)
            source_thread = connection.execute(
                "SELECT title, user_id FROM threads WHERE thread_id = %s", (thread_id,)
            ).fetchone()
            parent_title = source_thread["title"] if source_thread else "conversation"
            source_user_id = source_thread["user_id"] if source_thread else None
            branch_title = f"Branch · {parent_title.removeprefix('Branch · ')[:62]}"
            connection.execute(
                """
                INSERT INTO threads (
                    thread_id, title, created_at, updated_at, parent_thread_id,
                    branch_label, user_id
                ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    new_thread_id,
                    branch_title,
                    now,
                    now,
                    thread_id,
                    (target["assistant_content"].splitlines() or ["Branch"])[0][:80],
                    source_user_id,
                ),
            )
            copied_ids: dict[int, int] = {}
            for message in source_messages:
                content = message["content"]
                selected_version = message["selected_version"]
                failed = message["failed"]
                tools_json = message["tools_json"]
                parts_json = message["parts_json"]
                if message["id"] == target["user_id"]:
                    content = target["user_content"]
                    selected_version = version_index
                elif message["id"] == target["assistant_id"]:
                    content = target["assistant_content"]
                    failed = target["failed"]
                    tools_json = target["tools_json"]
                    parts_json = target["parts_json"]
                cursor = connection.execute(
                    """
                    INSERT INTO messages (
                        thread_id, role, content, created_at, is_active, selected_version, failed,
                        tools_json, parts_json
                    ) VALUES (%s, %s, %s, %s, 1, %s, %s, %s, %s)
                    RETURNING id
                    """,
                    (
                        new_thread_id,
                        message["role"],
                        content,
                        message["created_at"],
                        selected_version,
                        failed,
                        tools_json,
                        parts_json,
                    ),
                )
                copied_ids[message["id"]] = cursor.fetchone()["id"]
            for source_message_id, copied_message_id in copied_ids.items():
                source_attachments = connection.execute(
                    """
                    SELECT filename, content_type, storage_name, size_bytes,
                        extracted_text, created_at
                    FROM attachments WHERE message_id = %s
                    """,
                    (source_message_id,),
                ).fetchall()
                for attachment in source_attachments:
                    connection.execute(
                        """
                        INSERT INTO attachments (
                            attachment_id, thread_id, message_id, filename,
                            content_type, storage_name, size_bytes, extracted_text,
                            created_at
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """,
                        (
                            secrets.token_hex(16),
                            new_thread_id,
                            copied_message_id,
                            attachment["filename"],
                            attachment["content_type"],
                            attachment["storage_name"],
                            attachment["size_bytes"],
                            attachment["extracted_text"],
                            attachment["created_at"],
                        ),
                    )
            connection.execute(
                """
                UPDATE threads SET branch_from_message_id = %s
                WHERE thread_id = %s
                """,
                (copied_ids[target["assistant_id"]], new_thread_id),
            )
            for message in source_messages:
                if message["role"] != "user":
                    continue
                assistant = connection.execute(
                    """
                    SELECT id FROM messages
                    WHERE thread_id = %s AND role = 'assistant' AND id > %s
                    ORDER BY id LIMIT 1
                    """,
                    (thread_id, message["id"]),
                ).fetchone()
                if assistant is None or assistant["id"] not in copied_ids:
                    continue
                stored_variants = connection.execute(
                    """
                    SELECT version_index, user_content, assistant_content, created_at, failed
                    FROM turn_versions WHERE user_message_id = %s
                    """,
                    (message["id"],),
                ).fetchall()
                for variant in stored_variants:
                    variant_user = variant["user_content"]
                    variant_assistant = variant["assistant_content"]
                    if message["id"] == target["user_id"] and variant["version_index"] == version_index:
                        variant_user = target["user_content"]
                        variant_assistant = target["assistant_content"]
                    connection.execute(
                        """
                        INSERT INTO turn_versions (
                            user_message_id, assistant_message_id, version_index,
                            user_content, assistant_content, created_at, failed
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                        """,
                        (
                            copied_ids[message["id"]],
                            copied_ids[assistant["id"]],
                            variant["version_index"],
                            variant_user,
                            variant_assistant,
                            variant["created_at"],
                            variant["failed"],
                        ),
                    )
        return self.get_thread(new_thread_id)

    def get_thread_cleanup_info(self, thread_id: str) -> dict[str, list[str]]:
        """Collects attachment storage names for a thread prior to deletion."""
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT storage_name FROM attachments WHERE thread_id = %s",
                (thread_id,),
            ).fetchall()
        return {"storage_names": [row["storage_name"] for row in rows]}

    def delete_thread(
        self, thread_id: str, user_id: Optional[str] = None
    ) -> bool:
        with self._connection() as connection:
            if user_id is not None:
                cursor = connection.execute(
                    "DELETE FROM threads WHERE thread_id = %s AND user_id = %s",
                    (thread_id, user_id),
                )
            else:
                cursor = connection.execute(
                    "DELETE FROM threads WHERE thread_id = %s", (thread_id,)
                )
        invalidate_thread_cache(thread_id)
        return cursor.rowcount > 0

    def rename_thread(
        self, thread_id: str, title: str, user_id: Optional[str] = None
    ) -> bool:
        with self._connection() as connection:
            if user_id is not None:
                cursor = connection.execute(
                    "UPDATE threads SET title = %s WHERE thread_id = %s AND user_id = %s",
                    (title, thread_id, user_id),
                )
            else:
                cursor = connection.execute(
                    "UPDATE threads SET title = %s WHERE thread_id = %s",
                    (title, thread_id),
                )
        invalidate_thread_cache(thread_id)
        return cursor.rowcount > 0

    def get_settings(self, user_id: Optional[str] = None) -> dict[str, Any]:
        if user_id is not None:
            with self._connection() as connection:
                rows = connection.execute(
                    """
                    SELECT key, value FROM user_settings
                    WHERE user_id = %s AND key IN ('model', 'provider_index')
                    """,
                    (user_id,),
                ).fetchall()
        else:
            with self._connection() as connection:
                rows = connection.execute(
                    "SELECT key, value FROM app_settings WHERE key IN ('model', 'provider_index')"
                ).fetchall()
        values = {row["key"]: row["value"] for row in rows}
        provider_index = values.get("provider_index", "")
        return {
            "model": values.get("model") or None,
            "provider_index": int(provider_index) if provider_index else None,
        }

    def save_settings(
        self,
        model: Optional[str],
        provider_index: Optional[int],
        user_id: Optional[str] = None,
    ) -> dict[str, Any]:
        values = {
            "model": model or "",
            "provider_index": str(provider_index) if provider_index is not None else "",
        }
        with self._connection() as connection:
            if user_id is not None:
                connection.cursor().executemany(
                    """
                    INSERT INTO user_settings (user_id, key, value) VALUES (%s, %s, %s)
                    ON CONFLICT(user_id, key) DO UPDATE SET value = excluded.value
                    """,
                    [(user_id, k, v) for k, v in values.items()],
                )
            else:
                connection.cursor().executemany(
                    """
                    INSERT INTO app_settings (key, value) VALUES (%s, %s)
                    ON CONFLICT(key) DO UPDATE SET value = excluded.value
                    """,
                    values.items(),
                )
        return {"model": model, "provider_index": provider_index}

    def get_setting(
        self,
        key: str,
        default: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> Optional[str]:
        with self._connection() as connection:
            if user_id is not None:
                row = connection.execute(
                    "SELECT value FROM user_settings WHERE user_id = %s AND key = %s",
                    (user_id, key),
                ).fetchone()
            else:
                row = connection.execute(
                    "SELECT value FROM app_settings WHERE key = %s", (key,)
                ).fetchone()
            return row["value"] if row else default

    def set_setting(
        self, key: str, value: str, user_id: Optional[str] = None
    ) -> None:
        with self._connection() as connection:
            if user_id is not None:
                connection.execute(
                    """
                    INSERT INTO user_settings (user_id, key, value) VALUES (%s, %s, %s)
                    ON CONFLICT(user_id, key) DO UPDATE SET value = excluded.value
                    """,
                    (user_id, key, value),
                )
            else:
                connection.execute(
                    """
                    INSERT INTO app_settings (key, value) VALUES (%s, %s)
                    ON CONFLICT(key) DO UPDATE SET value = excluded.value
                    """,
                    (key, value),
                )

    def delete_setting(self, key: str, user_id: Optional[str] = None) -> bool:
        with self._connection() as connection:
            if user_id is not None:
                cursor = connection.execute(
                    "DELETE FROM user_settings WHERE user_id = %s AND key = %s",
                    (user_id, key),
                )
            else:
                cursor = connection.execute("DELETE FROM app_settings WHERE key = %s", (key,))
            return cursor.rowcount > 0

    def get_settings_by_prefix(
        self, prefix: str, user_id: Optional[str] = None
    ) -> dict[str, str]:
        with self._connection() as connection:
            if user_id is not None:
                rows = connection.execute(
                    "SELECT key, value FROM user_settings WHERE user_id = %s AND key LIKE %s",
                    (user_id, f"{prefix}%"),
                ).fetchall()
            else:
                rows = connection.execute(
                    "SELECT key, value FROM app_settings WHERE key LIKE %s",
                    (f"{prefix}%",),
                ).fetchall()
            return {row["key"]: row["value"] for row in rows}

    # --------------------------------------------------------------------------
    # Users, Sessions & Recovery Codes
    # --------------------------------------------------------------------------
    @staticmethod
    def _row_to_public_user(row: dict[str, Any]) -> dict[str, Any]:
        return {
            "user_id": row["user_id"],
            "email": row["email"],
            "display_name": row["display_name"],
            "avatar_color": row["avatar_color"],
            "role": row["role"],
            "is_active": bool(row["is_active"]),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "last_login_at": row["last_login_at"],
        }

    def count_users(self) -> int:
        with self._connection() as connection:
            return int(
                connection.execute("SELECT COUNT(*) AS count FROM users").fetchone()["count"]
            )

    def count_active_admins(self) -> int:
        with self._connection() as connection:
            return int(
                connection.execute(
                    "SELECT COUNT(*) AS count FROM users WHERE role = 'admin' AND is_active = 1"
                ).fetchone()["count"]
            )

    def create_user(
        self,
        email: str,
        display_name: str,
        password_hash: str,
        avatar_color: str = "#673ab7",
        recovery_code_hashes: Optional[list[str]] = None,
        initial_installed_skills: Optional[list[str]] = None,
    ) -> tuple[dict[str, Any], bool]:
        """Creates a user account. First account becomes admin and claims legacy data."""
        now = _utc_now()
        user_id = f"usr-{uuid4().hex[:16]}"
        with self._connection() as connection:
            connection.execute(
                "SELECT pg_advisory_xact_lock(hashtextextended('orbit:first-user', 0))"
            )
            existing_count = int(
                connection.execute("SELECT COUNT(*) AS count FROM users").fetchone()["count"]
            )
            is_first_admin = existing_count == 0
            role = "admin" if is_first_admin else "member"

            connection.execute(
                """
                INSERT INTO users (
                    user_id, email, display_name, avatar_color, password_hash,
                    role, is_active, created_at, updated_at, last_login_at
                ) VALUES (%s, %s, %s, %s, %s, %s, 1, %s, %s, %s)
                """,
                (
                    user_id,
                    email,
                    display_name,
                    avatar_color,
                    password_hash,
                    role,
                    now,
                    now,
                    now,
                ),
            )
            if recovery_code_hashes:
                connection.cursor().executemany(
                    """
                    INSERT INTO user_recovery_codes (user_id, code_hash, used, created_at)
                    VALUES (%s, %s, 0, %s)
                    """,
                    [(user_id, code_hash, now) for code_hash in recovery_code_hashes],
                )
            if is_first_admin:
                # Claim legacy threads & seed per-user settings from global defaults
                connection.execute(
                    "UPDATE threads SET user_id = %s WHERE user_id IS NULL",
                    (user_id,),
                )
                global_settings = connection.execute(
                    "SELECT key, value FROM app_settings"
                ).fetchall()
                if global_settings:
                    connection.cursor().executemany(
                        """
                        INSERT INTO user_settings (user_id, key, value)
                        VALUES (%s, %s, %s)
                        ON CONFLICT (user_id, key) DO NOTHING
                        """,
                        [(user_id, r["key"], r["value"]) for r in global_settings],
                    )
                if initial_installed_skills:
                    connection.execute(
                        """
                        INSERT INTO user_settings (user_id, key, value)
                        VALUES (%s, 'marketplace_installed_skills', %s)
                        ON CONFLICT (user_id, key) DO NOTHING
                        """,
                        (user_id, json.dumps(sorted(set(initial_installed_skills)))),
                    )
            row = connection.execute(
                "SELECT * FROM users WHERE user_id = %s", (user_id,)
            ).fetchone()
        return self._row_to_public_user(row), is_first_admin

    def get_user_credentials_by_email(self, email: str) -> Optional[dict[str, Any]]:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM users WHERE email = %s", (email,)
            ).fetchone()
            return dict(row) if row else None

    def get_user_with_hash_by_id(self, user_id: str) -> Optional[dict[str, Any]]:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM users WHERE user_id = %s", (user_id,)
            ).fetchone()
            return dict(row) if row else None

    def get_user_by_id(self, user_id: str) -> Optional[dict[str, Any]]:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM users WHERE user_id = %s", (user_id,)
            ).fetchone()
            return self._row_to_public_user(row) if row else None

    def update_user_profile(
        self,
        user_id: str,
        display_name: str,
        email: str,
        avatar_color: str,
    ) -> Optional[dict[str, Any]]:
        now = _utc_now()
        with self._connection() as connection:
            connection.execute(
                """
                UPDATE users
                SET display_name = %s, email = %s, avatar_color = %s, updated_at = %s
                WHERE user_id = %s
                """,
                (display_name, email, avatar_color, now, user_id),
            )
            row = connection.execute(
                "SELECT * FROM users WHERE user_id = %s", (user_id,)
            ).fetchone()
            return self._row_to_public_user(row) if row else None

    def update_user_password(self, user_id: str, password_hash: str) -> bool:
        now = _utc_now()
        with self._connection() as connection:
            cursor = connection.execute(
                "UPDATE users SET password_hash = %s, updated_at = %s WHERE user_id = %s",
                (password_hash, now, user_id),
            )
            return cursor.rowcount > 0

    def list_users(self) -> list[dict[str, Any]]:
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT u.*,
                    (SELECT COUNT(*) FROM threads t WHERE t.user_id = u.user_id) AS thread_count
                FROM users u
                ORDER BY u.id ASC
                """
            ).fetchall()
            result = []
            for row in rows:
                item = self._row_to_public_user(row)
                item["thread_count"] = row["thread_count"]
                result.append(item)
            return result

    def update_user_admin_fields(
        self,
        user_id: str,
        role: Optional[str] = None,
        is_active: Optional[bool] = None,
    ) -> Optional[dict[str, Any]]:
        updates: list[str] = ["updated_at = %s"]
        params: list[Any] = [_utc_now()]
        if role is not None:
            updates.append("role = %s")
            params.append(role)
        if is_active is not None:
            updates.append("is_active = %s")
            params.append(1 if is_active else 0)
        params.append(user_id)
        with self._connection() as connection:
            cursor = connection.execute(
                f"UPDATE users SET {', '.join(updates)} WHERE user_id = %s",
                params,
            )
            if cursor.rowcount == 0:
                return None
            if is_active is False:
                connection.execute(
                    "UPDATE user_sessions SET revoked = 1 WHERE user_id = %s",
                    (user_id,),
                )
            row = connection.execute(
                "SELECT * FROM users WHERE user_id = %s", (user_id,)
            ).fetchone()
            return self._row_to_public_user(row) if row else None

    def delete_user(self, user_id: str) -> dict[str, Any]:
        """Deletes a user and returns owned thread IDs and attachment filenames for disk cleanup."""
        with self._connection() as connection:
            thread_rows = connection.execute(
                "SELECT thread_id FROM threads WHERE user_id = %s", (user_id,)
            ).fetchall()
            thread_ids = [r["thread_id"] for r in thread_rows]
            attachment_rows = connection.execute(
                """
                SELECT a.storage_name
                FROM attachments a
                JOIN threads t ON t.thread_id = a.thread_id
                WHERE t.user_id = %s
                """,
                (user_id,),
            ).fetchall()
            storage_names = [r["storage_name"] for r in attachment_rows]
            cursor = connection.execute(
                "DELETE FROM users WHERE user_id = %s", (user_id,)
            )
        return {
            "deleted": cursor.rowcount > 0,
            "thread_ids": thread_ids,
            "storage_names": storage_names,
        }

    def create_session(
        self,
        user_id: str,
        token_hash: str,
        expires_at: str,
        remember_me: bool = False,
        user_agent: str = "",
        ip_address: str = "",
    ) -> dict[str, Any]:
        now = _utc_now()
        session_id = f"sess-{uuid4().hex[:16]}"
        with self._connection() as connection:
            connection.execute(
                """
                INSERT INTO user_sessions (
                    session_id, user_id, token_hash, remember_me,
                    user_agent, ip_address, created_at, last_seen_at, expires_at, revoked
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, 0)
                """,
                (
                    session_id,
                    user_id,
                    token_hash,
                    1 if remember_me else 0,
                    user_agent[:255],
                    ip_address[:64],
                    now,
                    now,
                    expires_at,
                ),
            )
            connection.execute(
                "UPDATE users SET last_login_at = %s WHERE user_id = %s",
                (now, user_id),
            )
        return {
            "session_id": session_id,
            "user_id": user_id,
            "remember_me": remember_me,
            "created_at": now,
            "expires_at": expires_at,
        }

    def get_session_with_user(self, token_hash: str) -> Optional[dict[str, Any]]:
        now = _utc_now()
        with self._connection() as connection:
            row = connection.execute(
                """
                SELECT s.session_id, s.remember_me, s.expires_at,
                    u.user_id, u.email, u.display_name, u.avatar_color,
                    u.role, u.is_active, u.created_at, u.updated_at, u.last_login_at
                FROM user_sessions s
                JOIN users u ON u.user_id = s.user_id
                WHERE s.token_hash = %s
                  AND s.revoked = 0
                  AND s.expires_at > %s
                  AND u.is_active = 1
                """,
                (token_hash, now),
            ).fetchone()
            if row is None:
                return None
            connection.execute(
                "UPDATE user_sessions SET last_seen_at = %s WHERE session_id = %s",
                (now, row["session_id"]),
            )
        return {
            "session_id": row["session_id"],
            "remember_me": bool(row["remember_me"]),
            "expires_at": row["expires_at"],
            "user": self._row_to_public_user(row),
        }

    def list_user_sessions(
        self, user_id: str, current_session_id: Optional[str] = None
    ) -> list[dict[str, Any]]:
        now = _utc_now()
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT session_id, remember_me, user_agent, ip_address,
                    created_at, last_seen_at, expires_at
                FROM user_sessions
                WHERE user_id = %s AND revoked = 0 AND expires_at > %s
                ORDER BY last_seen_at DESC, created_at DESC
                """,
                (user_id, now),
            ).fetchall()
        return [
            {
                "session_id": row["session_id"],
                "remember_me": bool(row["remember_me"]),
                "user_agent": row["user_agent"],
                "ip_address": row["ip_address"],
                "created_at": row["created_at"],
                "last_seen_at": row["last_seen_at"],
                "expires_at": row["expires_at"],
                "is_current": row["session_id"] == current_session_id,
            }
            for row in rows
        ]

    def get_session_token_hash(self, user_id: str, session_id: str) -> Optional[str]:
        """Return the hashed token for one session owned by this user."""
        with self._connection() as connection:
            row = connection.execute(
                """
                SELECT token_hash FROM user_sessions
                WHERE user_id = %s AND session_id = %s
                """,
                (user_id, session_id),
            ).fetchone()
        return row["token_hash"] if row else None

    def list_unrevoked_session_token_hashes(
        self, user_id: str, except_session_id: Optional[str] = None
    ) -> list[str]:
        """Return token hashes for sessions this user-revocation operation will revoke."""
        with self._connection() as connection:
            if except_session_id:
                rows = connection.execute(
                    """
                    SELECT token_hash FROM user_sessions
                    WHERE user_id = %s AND revoked = 0 AND session_id != %s
                    """,
                    (user_id, except_session_id),
                ).fetchall()
            else:
                rows = connection.execute(
                    """
                    SELECT token_hash FROM user_sessions
                    WHERE user_id = %s AND revoked = 0
                    """,
                    (user_id,),
                ).fetchall()
        return [row["token_hash"] for row in rows if row["token_hash"]]

    def revoke_session(self, user_id: str, session_id: str) -> bool:
        with self._connection() as connection:
            cursor = connection.execute(
                """
                UPDATE user_sessions SET revoked = 1
                WHERE user_id = %s AND session_id = %s AND revoked = 0
                """,
                (user_id, session_id),
            )
            return cursor.rowcount > 0

    def revoke_session_by_token_hash(self, token_hash: str) -> bool:
        with self._connection() as connection:
            cursor = connection.execute(
                "UPDATE user_sessions SET revoked = 1 WHERE token_hash = %s AND revoked = 0",
                (token_hash,),
            )
            return cursor.rowcount > 0

    def revoke_all_sessions(
        self, user_id: str, except_session_id: Optional[str] = None
    ) -> int:
        with self._connection() as connection:
            if except_session_id:
                cursor = connection.execute(
                    """
                    UPDATE user_sessions SET revoked = 1
                    WHERE user_id = %s AND revoked = 0 AND session_id != %s
                    """,
                    (user_id, except_session_id),
                )
            else:
                cursor = connection.execute(
                    "UPDATE user_sessions SET revoked = 1 WHERE user_id = %s AND revoked = 0",
                    (user_id,),
                )
            return cursor.rowcount

    def replace_recovery_codes(
        self, user_id: str, code_hashes: list[str], keep_existing: bool = False
    ) -> None:
        now = _utc_now()
        with self._connection() as connection:
            if not keep_existing:
                connection.execute(
                    "DELETE FROM user_recovery_codes WHERE user_id = %s", (user_id,)
                )
            connection.cursor().executemany(
                """
                INSERT INTO user_recovery_codes (user_id, code_hash, used, created_at)
                VALUES (%s, %s, 0, %s)
                """,
                [(user_id, code_hash, now) for code_hash in code_hashes],
            )

    def count_unused_recovery_codes(self, user_id: str) -> int:
        with self._connection() as connection:
            return int(
                connection.execute(
                    "SELECT COUNT(*) AS count FROM user_recovery_codes WHERE user_id = %s AND used = 0",
                    (user_id,),
                ).fetchone()["count"]
            )

    def consume_recovery_code(self, user_id: str, code_hash: str) -> bool:
        with self._connection() as connection:
            row = connection.execute(
                """
                SELECT id FROM user_recovery_codes
                WHERE user_id = %s AND code_hash = %s AND used = 0
                LIMIT 1
                FOR UPDATE
                """,
                (user_id, code_hash),
            ).fetchone()
            if row is None:
                return False
            connection.execute(
                "UPDATE user_recovery_codes SET used = 1 WHERE id = %s",
                (row["id"],),
            )
            return True

    def create_otp_code(
        self, user_id: str, purpose: str, code_hash: str, expires_at: str
    ) -> None:
        now = _utc_now()
        with self._connection() as connection:
            connection.execute(
                "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))",
                (f"orbit:otp:{user_id}:{purpose}",),
            )
            connection.execute(
                "UPDATE user_otp_codes SET used = 1 WHERE user_id = %s AND purpose = %s AND used = 0",
                (user_id, purpose),
            )
            connection.execute(
                """
                INSERT INTO user_otp_codes (
                    user_id, purpose, code_hash, attempts, used, expires_at, created_at
                ) VALUES (%s, %s, %s, 0, 0, %s, %s)
                """,
                (user_id, purpose, code_hash, expires_at, now),
            )

    def verify_and_consume_otp_code(
        self,
        user_id: str,
        purpose: str,
        code_hash: str,
        max_attempts: int = 5,
    ) -> bool:
        now = _utc_now()
        with self._connection() as connection:
            row = connection.execute(
                """
                SELECT id, code_hash, attempts, expires_at
                FROM user_otp_codes
                WHERE user_id = %s AND purpose = %s AND used = 0 AND expires_at > %s
                ORDER BY id DESC LIMIT 1
                FOR UPDATE
                """,
                (user_id, purpose, now),
            ).fetchone()
            if row is None:
                return False
            if row["attempts"] >= max_attempts:
                connection.execute(
                    "UPDATE user_otp_codes SET used = 1 WHERE id = %s",
                    (row["id"],),
                )
                return False
            if row["code_hash"] != code_hash:
                next_attempts = int(row["attempts"]) + 1
                connection.execute(
                    "UPDATE user_otp_codes SET attempts = %s, used = %s WHERE id = %s",
                    (next_attempts, 1 if next_attempts >= max_attempts else 0, row["id"]),
                )
                return False
            connection.execute(
                "UPDATE user_otp_codes SET used = 1 WHERE id = %s",
                (row["id"],),
            )
            return True

    # ------------------------------------------------------------------
    # User UI Preferences (General / Appearance)
    # ------------------------------------------------------------------

    def get_user_preferences(self, user_id: str) -> dict[str, Any]:
        values = self.get_settings_by_prefix("pref_", user_id=user_id)
        language = values.get("pref_language") or "auto"
        send_shortcut = values.get("pref_send_shortcut") or "enter"
        if send_shortcut not in {"enter", "ctrl_enter"}:
            send_shortcut = "enter"
        appearance = values.get("pref_appearance") or "dark"
        if appearance not in {"dark", "midnight"}:
            appearance = "dark"
        return {
            "language": language,
            "send_shortcut": send_shortcut,
            "appearance": appearance,
        }

    def save_user_preferences(
        self,
        user_id: str,
        *,
        language: Optional[str] = None,
        send_shortcut: Optional[str] = None,
        appearance: Optional[str] = None,
    ) -> dict[str, Any]:
        if language is not None:
            self.set_setting("pref_language", language.strip() or "auto", user_id=user_id)
        if send_shortcut is not None:
            normalized_shortcut = "ctrl_enter" if send_shortcut == "ctrl_enter" else "enter"
            self.set_setting("pref_send_shortcut", normalized_shortcut, user_id=user_id)
        if appearance is not None:
            normalized_appearance = "midnight" if appearance == "midnight" else "dark"
            self.set_setting("pref_appearance", normalized_appearance, user_id=user_id)
        return self.get_user_preferences(user_id)

    # ------------------------------------------------------------------
    # Personal Skills (Strictly Scoped to User)
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_personal_skill(row: dict[str, Any]) -> dict[str, Any]:
        try:
            files = json.loads(row["files_json"] or "[]")
            if not isinstance(files, list):
                files = []
        except Exception:
            files = []
        try:
            scan_raw = row["security_scan_json"] if "security_scan_json" in row.keys() else "{}"
            security_scan = json.loads(scan_raw or "{}")
            if not isinstance(security_scan, dict) or not security_scan:
                security_scan = None
        except Exception:
            security_scan = None
        return {
            "id": row["id"],
            "name": row["name"],
            "description": row["description"],
            "prompt": row["prompt"],
            "is_enabled": bool(row["is_enabled"]),
            "is_draft": bool(row["is_draft"]),
            "source": row["source"] or "custom",
            "files": files,
            "security_scan": security_scan,
            "created_at": row["created_at"],
            "updated_at": row["updated_at"] or row["created_at"],
        }

    def get_personal_skills(self, user_id: str) -> list[dict[str, Any]]:
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT id, name, description, prompt, is_enabled, is_draft,
                    source, files_json, security_scan_json, created_at, updated_at
                FROM user_personal_skills
                WHERE user_id = %s
                ORDER BY created_at DESC
                """,
                (user_id,),
            ).fetchall()
        return [self._row_to_personal_skill(row) for row in rows]

    def get_personal_skill(self, user_id: str, skill_id: str) -> Optional[dict[str, Any]]:
        with self._connection() as connection:
            row = connection.execute(
                """
                SELECT id, name, description, prompt, is_enabled, is_draft,
                    source, files_json, security_scan_json, created_at, updated_at
                FROM user_personal_skills
                WHERE id = %s AND user_id = %s
                """,
                (skill_id, user_id),
            ).fetchone()
        return self._row_to_personal_skill(row) if row else None

    def create_personal_skill(
        self,
        user_id: str,
        name: str,
        description: str,
        prompt: str,
        *,
        is_enabled: bool = True,
        is_draft: bool = False,
        source: str = "custom",
        files: Optional[list[dict[str, Any]]] = None,
        security_scan: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        skill_id = str(uuid4())
        now = _utc_now()
        safe_files = files if isinstance(files, list) else []
        files_json = json.dumps(safe_files, ensure_ascii=False)
        safe_scan = security_scan if isinstance(security_scan, dict) else {}
        scan_json = json.dumps(safe_scan, ensure_ascii=False)
        safe_source = source if source in {"custom", "upload"} else "custom"
        with self._connection() as connection:
            connection.execute(
                """
                INSERT INTO user_personal_skills (
                    id, user_id, name, description, prompt,
                    is_enabled, is_draft, source, files_json, security_scan_json, created_at, updated_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    skill_id,
                    user_id,
                    name,
                    description,
                    prompt,
                    1 if is_enabled else 0,
                    1 if is_draft else 0,
                    safe_source,
                    files_json,
                    scan_json,
                    now,
                    now,
                ),
            )
        return {
            "id": skill_id,
            "name": name,
            "description": description,
            "prompt": prompt,
            "is_enabled": is_enabled,
            "is_draft": is_draft,
            "source": safe_source,
            "files": safe_files,
            "security_scan": safe_scan or None,
            "created_at": now,
            "updated_at": now,
        }

    def update_personal_skill(
        self,
        user_id: str,
        skill_id: str,
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
        prompt: Optional[str] = None,
        is_enabled: Optional[bool] = None,
        is_draft: Optional[bool] = None,
        files: Optional[list[dict[str, Any]]] = None,
        security_scan: Optional[dict[str, Any]] = None,
    ) -> Optional[dict[str, Any]]:
        updates: list[str] = ["updated_at = %s"]
        params: list[Any] = [_utc_now()]
        if name is not None:
            updates.append("name = %s")
            params.append(name.strip())
        if description is not None:
            updates.append("description = %s")
            params.append(description)
        if prompt is not None:
            updates.append("prompt = %s")
            params.append(prompt)
        if is_enabled is not None:
            updates.append("is_enabled = %s")
            params.append(1 if is_enabled else 0)
        if is_draft is not None:
            updates.append("is_draft = %s")
            params.append(1 if is_draft else 0)
        if files is not None:
            updates.append("files_json = %s")
            params.append(json.dumps(files, ensure_ascii=False))
        if security_scan is not None:
            updates.append("security_scan_json = %s")
            params.append(json.dumps(security_scan, ensure_ascii=False))
        params.extend([skill_id, user_id])
        with self._connection() as connection:
            cursor = connection.execute(
                f"UPDATE user_personal_skills SET {', '.join(updates)} WHERE id = %s AND user_id = %s",
                params,
            )
            if cursor.rowcount == 0:
                return None
        return self.get_personal_skill(user_id, skill_id)

    def delete_personal_skill(self, user_id: str, skill_id: str) -> bool:
        with self._connection() as connection:
            cursor = connection.execute(
                "DELETE FROM user_personal_skills WHERE id = %s AND user_id = %s",
                (skill_id, user_id),
            )
        return cursor.rowcount > 0

    # ------------------------------------------------------------------
    # Personal MCP Connectors (Strictly Scoped to User)
    # ------------------------------------------------------------------

    @staticmethod
    def _mask_header_value(value: str) -> str:
        if not value or len(value) <= 6:
            return "••••••"
        return f"{value[:2]}••••••{value[-2:]}"

    def _row_to_personal_connector(
        self, row: dict[str, Any], include_secrets: bool = False
    ) -> dict[str, Any]:
        try:
            raw_headers = json.loads(row["headers_json"] or "[]")
            if not isinstance(raw_headers, list):
                raw_headers = []
        except Exception:
            raw_headers = []

        vault = self.credential_vault
        if include_secrets:
            headers_out = vault.decrypt_headers(raw_headers)
        else:
            headers_out = [
                {
                    "key": str(item.get("key", "")),
                    "masked_value": "••••••" if item.get("value") else "",
                    "has_value": bool(item.get("value")),
                }
                for item in raw_headers
                if isinstance(item, dict) and item.get("key")
            ]

        try:
            advanced = json.loads(row["advanced_json"] or "{}")
            if not isinstance(advanced, dict):
                advanced = {}
        except Exception:
            advanced = {}

        advanced = (
            vault.decrypt_secret_fields(advanced)
            if include_secrets
            else _mask_secret_fields(advanced, vault)
        )

        try:
            tools = json.loads(row["tools_json"] or "[]")
            if not isinstance(tools, list):
                tools = []
        except Exception:
            tools = []

        try:
            tool_permissions = json.loads(row["tool_permissions_json"] or "{}")
            if not isinstance(tool_permissions, dict):
                tool_permissions = {}
        except Exception:
            tool_permissions = {}

        for t in tools:
            if isinstance(t, dict):
                tname = t.get("name")
                if tname and tname not in tool_permissions:
                    tool_permissions[tname] = "allow" if t.get("readOnlyHint") else "approval"

        return {
            "id": row["id"],
            "name": row["name"],
            "mcp_url": row["mcp_url"],
            "connector_type": row["connector_type"] or "Web",
            "auth_mode": row["auth_mode"] or "none",
            "headers": headers_out,
            "advanced": advanced,
            "tools": tools,
            "tool_permissions": tool_permissions,
            "is_connected": bool(row["is_connected"]),
            "last_status_code": row["last_status_code"],
            "last_checked_at": row["last_checked_at"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def get_personal_connectors(
        self, user_id: str, include_secrets: bool = False
    ) -> list[dict[str, Any]]:
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT * FROM user_personal_connectors
                WHERE user_id = %s
                ORDER BY created_at DESC
                """,
                (user_id,),
            ).fetchall()
        return [
            self._row_to_personal_connector(row, include_secrets=include_secrets)
            for row in rows
        ]

    def get_personal_connector(
        self, user_id: str, connector_id: str, include_secrets: bool = False
    ) -> Optional[dict[str, Any]]:
        with self._connection() as connection:
            row = connection.execute(
                """
                SELECT * FROM user_personal_connectors
                WHERE id = %s AND user_id = %s
                """,
                (connector_id, user_id),
            ).fetchone()
        return (
            self._row_to_personal_connector(row, include_secrets=include_secrets)
            if row
            else None
        )

    def create_personal_connector(
        self,
        user_id: str,
        *,
        name: str,
        mcp_url: str,
        connector_type: str = "Web",
        auth_mode: str = "none",
        headers: Optional[list[dict[str, str]]] = None,
        advanced: Optional[dict[str, Any]] = None,
        tools: Optional[list[dict[str, Any]]] = None,
        tool_permissions: Optional[dict[str, str]] = None,
        is_connected: bool = False,
        last_status_code: Optional[int] = None,
    ) -> dict[str, Any]:
        connector_id = str(uuid4())
        now = _utc_now()
        safe_auth = auth_mode if auth_mode in {"now", "later", "none"} else "none"
        clean_headers = [
            {"key": str(h["key"]).strip(), "value": str(h["value"])}
            for h in (headers or [])[:4]
            if isinstance(h, dict) and str(h.get("key", "")).strip()
        ]
        safe_tools = tools if isinstance(tools, list) else []
        vault = self.credential_vault
        encrypted_headers = vault.encrypt_headers(clean_headers)
        encrypted_advanced = vault.encrypt_secret_fields(advanced or {})
        perms = dict(tool_permissions or {})
        for t in safe_tools:
            tname = t.get("name") if isinstance(t, dict) else None
            if tname and tname not in perms:
                is_read_only = bool(t.get("readOnlyHint"))
                perms[tname] = "allow" if is_read_only else "approval"

        with self._connection() as connection:
            connection.execute(
                """
                INSERT INTO user_personal_connectors (
                    id, user_id, name, mcp_url, connector_type, auth_mode,
                    headers_json, advanced_json, tools_json, tool_permissions_json,
                    is_connected, last_status_code, last_checked_at, created_at, updated_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    connector_id,
                    user_id,
                    name.strip(),
                    mcp_url.strip(),
                    connector_type or "Web",
                    safe_auth,
                    json.dumps(encrypted_headers, ensure_ascii=False),
                    json.dumps(encrypted_advanced, ensure_ascii=False),
                    json.dumps(safe_tools, ensure_ascii=False),
                    json.dumps(perms, ensure_ascii=False),
                    1 if is_connected else 0,
                    last_status_code,
                    now if last_status_code is not None else None,
                    now,
                    now,
                ),
            )
        return self.get_personal_connector(user_id, connector_id, include_secrets=False)  # type: ignore[return-value]

    def update_personal_connector(
        self,
        user_id: str,
        connector_id: str,
        *,
        name: Optional[str] = None,
        mcp_url: Optional[str] = None,
        auth_mode: Optional[str] = None,
        headers: Optional[list[dict[str, str]]] = None,
        advanced: Optional[dict[str, Any]] = None,
        tools: Optional[list[dict[str, Any]]] = None,
        tool_permissions: Optional[dict[str, str]] = None,
        is_connected: Optional[bool] = None,
        last_status_code: Optional[int] = None,
    ) -> Optional[dict[str, Any]]:
        existing = self.get_personal_connector(user_id, connector_id, include_secrets=True)
        if not existing:
            return None

        now = _utc_now()
        updates: list[str] = ["updated_at = %s"]
        params: list[Any] = [now]

        if name is not None:
            updates.append("name = %s")
            params.append(name.strip())
        if mcp_url is not None:
            updates.append("mcp_url = %s")
            params.append(mcp_url.strip())
        if auth_mode is not None and auth_mode in {"now", "later", "none"}:
            updates.append("auth_mode = %s")
            params.append(auth_mode)
        if headers is not None:
            existing_by_key = {
                str(h.get("key", "")).lower(): str(h.get("value", ""))
                for h in existing.get("headers", [])
            }
            merged_headers: list[dict[str, str]] = []
            for h in headers[:4]:
                if not isinstance(h, dict):
                    continue
                k = str(h.get("key", "")).strip()
                if not k:
                    continue
                v = str(h.get("value", ""))
                if not v and k.lower() in existing_by_key:
                    v = existing_by_key[k.lower()]
                merged_headers.append({"key": k, "value": v})
            updates.append("headers_json = %s")
            params.append(json.dumps(self.credential_vault.encrypt_headers(merged_headers), ensure_ascii=False))
        if advanced is not None:
            merged_adv = dict(existing.get("advanced", {}))
            for k, v in advanced.items():
                if isinstance(v, str) and "••••" in v:
                    continue
                merged_adv[k] = v
            updates.append("advanced_json = %s")
            params.append(json.dumps(self.credential_vault.encrypt_secret_fields(merged_adv), ensure_ascii=False))
        if tools is not None:
            updates.append("tools_json = %s")
            params.append(json.dumps(tools, ensure_ascii=False))
            perms = dict(existing.get("tool_permissions", {}))
            for t in tools:
                tname = t.get("name") if isinstance(t, dict) else None
                if tname:
                    is_read_only = bool(t.get("readOnlyHint"))
                    if tname not in perms:
                        perms[tname] = "allow" if is_read_only else "approval"
                    elif is_read_only and perms.get(tname) == "approval" and tool_permissions is None:
                        perms[tname] = "allow"
            updates.append("tool_permissions_json = %s")
            params.append(json.dumps(perms, ensure_ascii=False))
        if tool_permissions is not None:
            merged_perms = dict(existing.get("tool_permissions", {}))
            for tname, perm in tool_permissions.items():
                if perm in {"allow", "approval", "block"}:
                    merged_perms[str(tname)] = perm
            updates.append("tool_permissions_json = %s")
            params.append(json.dumps(merged_perms, ensure_ascii=False))
        if is_connected is not None:
            updates.append("is_connected = %s")
            params.append(1 if is_connected else 0)
        if last_status_code is not None:
            updates.append("last_status_code = %s")
            params.append(last_status_code)
            updates.append("last_checked_at = %s")
            params.append(now)

        params.extend([connector_id, user_id])
        with self._connection() as connection:
            connection.execute(
                f"UPDATE user_personal_connectors SET {', '.join(updates)} WHERE id = %s AND user_id = %s",
                params,
            )
        return self.get_personal_connector(user_id, connector_id, include_secrets=False)

    def delete_personal_connector(self, user_id: str, connector_id: str) -> bool:
        with self._connection() as connection:
            cursor = connection.execute(
                "DELETE FROM user_personal_connectors WHERE id = %s AND user_id = %s",
                (connector_id, user_id),
            )
        return cursor.rowcount > 0

    def migrate_legacy_connector_credentials(self) -> dict[str, int]:
        """Encrypt pre-vault custom connector and marketplace secrets in place."""
        vault = self.credential_vault
        if not vault.configured:
            raise RuntimeError(
                "Credential migration requires ORBIT_CREDENTIAL_ENCRYPTION_KEY "
                "or INTEGRATION_TOKEN_ENCRYPTION_KEY."
            )

        personal_count = 0
        marketplace_count = 0
        with self._connection() as connection:
            personal_rows = connection.execute(
                "SELECT id, user_id, headers_json, advanced_json FROM user_personal_connectors"
            ).fetchall()
            for row in personal_rows:
                headers = json.loads(row["headers_json"] or "[]")
                advanced = json.loads(row["advanced_json"] or "{}")
                encrypted_headers = vault.encrypt_headers(headers)
                encrypted_advanced = vault.encrypt_secret_fields(advanced)
                if encrypted_headers != headers or encrypted_advanced != advanced:
                    connection.execute(
                        """
                        UPDATE user_personal_connectors
                        SET headers_json = %s, advanced_json = %s, updated_at = %s
                        WHERE id = %s AND user_id = %s
                        """,
                        (
                            json.dumps(encrypted_headers, ensure_ascii=False),
                            json.dumps(encrypted_advanced, ensure_ascii=False),
                            _utc_now(),
                            row["id"],
                            row["user_id"],
                        ),
                    )
                    personal_count += 1

            from marketplace import AVAILABLE_CONNECTORS

            catalog = {item["id"]: item for item in AVAILABLE_CONNECTORS}
            setting_rows = connection.execute(
                """
                SELECT user_id, key, value FROM user_settings
                WHERE key LIKE 'connector_%_config'
                """
            ).fetchall()
            for row in setting_rows:
                setting_key = str(row["key"])
                connector_id = setting_key[len("connector_"):-len("_config")]
                connector = catalog.get(connector_id)
                if not connector:
                    continue
                try:
                    config = json.loads(row["value"] or "{}")
                except (TypeError, ValueError):
                    continue
                changed = False
                for field in connector.get("configFields", []):
                    if field.get("type") != "password":
                        continue
                    key = field.get("key")
                    value = config.get(key) if key else None
                    if value:
                        encrypted = vault.encrypt_text(str(value))
                        if encrypted != value:
                            config[key] = encrypted
                            changed = True
                if changed:
                    connection.execute(
                        """
                        UPDATE user_settings SET value = %s
                        WHERE user_id = %s AND key = %s
                        """,
                        (
                            json.dumps(config, ensure_ascii=False),
                            row["user_id"],
                            setting_key,
                        ),
                    )
                    marketplace_count += 1

        return {
            "personal_connectors_migrated": personal_count,
            "marketplace_connectors_migrated": marketplace_count,
        }

    # --------------------------------------------------------------------------
    # Trust Engine: Approvals & Immutable Audit Logging
    # --------------------------------------------------------------------------

    def interrupt_pending_approvals(self) -> int:
        """Fail closed on approvals whose live stream was lost during a restart."""
        now = _utc_now()
        decision_json = json.dumps(
            {"reason": "Orbit restarted before this approval was completed. Retry the action from chat."},
            ensure_ascii=False,
        )
        with self._connection() as connection:
            cursor = connection.execute(
                """
                UPDATE approvals
                SET status = 'interrupted', decision_json = %s, resolved_at = %s
                WHERE status = 'pending'
                """,
                (decision_json, now),
            )
        return int(cursor.rowcount or 0)

    def create_approval(
        self,
        approval_id: str,
        user_id: Optional[str],
        thread_id: Optional[str],
        tool_name: str,
        action_summary: str,
        risk_level: str = "medium",
        connector_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        now = _utc_now()
        details_json = json.dumps(_redact_sensitive_payload(details or {}), ensure_ascii=False)
        with self._connection() as connection:
            connection.execute(
                """
                INSERT INTO approvals (
                    id, user_id, thread_id, tool_name, connector_id,
                    action_summary, risk_level, status, details_json, decision_json,
                    created_at, resolved_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, 'pending', %s, '{}', %s, NULL)
                """,
                (
                    approval_id,
                    user_id,
                    thread_id,
                    tool_name,
                    connector_id,
                    action_summary,
                    risk_level,
                    details_json,
                    now,
                ),
            )
        return {
            "id": approval_id,
            "user_id": user_id,
            "thread_id": thread_id,
            "tool_name": tool_name,
            "connector_id": connector_id,
            "action_summary": action_summary,
            "risk_level": risk_level,
            "status": "pending",
            "details": details or {},
            "decision": {},
            "created_at": now,
            "resolved_at": None,
        }

    def update_approval(
        self,
        approval_id: str,
        status: str,
        decision: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        now = _utc_now()
        decision_json = json.dumps(_redact_sensitive_payload(decision or {}), ensure_ascii=False)
        with self._connection() as connection:
            if user_id is None:
                cursor = connection.execute(
                    """
                    UPDATE approvals
                    SET status = %s, decision_json = %s, resolved_at = %s
                    WHERE id = %s AND status = 'pending'
                    """,
                    (status, decision_json, now, approval_id),
                )
            else:
                cursor = connection.execute(
                    """
                    UPDATE approvals
                    SET status = %s, decision_json = %s, resolved_at = %s
                    WHERE id = %s AND user_id = %s AND status = 'pending'
                    """,
                    (status, decision_json, now, approval_id, user_id),
                )
            if cursor.rowcount == 0:
                return None
        return self.get_approval(approval_id)

    def get_approval(self, approval_id: str) -> Optional[Dict[str, Any]]:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM approvals WHERE id = %s",
                (approval_id,),
            ).fetchone()
        if not row:
            return None
        return {
            "id": row["id"],
            "user_id": row["user_id"],
            "thread_id": row["thread_id"],
            "tool_name": row["tool_name"],
            "connector_id": row["connector_id"],
            "action_summary": row["action_summary"],
            "risk_level": row["risk_level"],
            "status": row["status"],
            "details": json.loads(row["details_json"] or "{}"),
            "decision": json.loads(row["decision_json"] or "{}"),
            "created_at": row["created_at"],
            "resolved_at": row["resolved_at"],
        }

    def list_approvals(
        self,
        user_id: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        query = "SELECT * FROM approvals"
        params: List[Any] = []
        conds: List[str] = []
        if user_id:
            conds.append("user_id = %s")
            params.append(user_id)
        if status:
            conds.append("status = %s")
            params.append(status)
        if conds:
            query += f" WHERE {' AND '.join(conds)}"
        query += " ORDER BY created_at DESC LIMIT %s"
        params.append(max(1, min(limit, 200)))

        with self._connection() as connection:
            rows = connection.execute(query, params).fetchall()

        return [
            {
                "id": r["id"],
                "user_id": r["user_id"],
                "thread_id": r["thread_id"],
                "tool_name": r["tool_name"],
                "connector_id": r["connector_id"],
                "action_summary": r["action_summary"],
                "risk_level": r["risk_level"],
                "status": r["status"],
                "details": json.loads(r["details_json"] or "{}"),
                "decision": json.loads(r["decision_json"] or "{}"),
                "created_at": r["created_at"],
                "resolved_at": r["resolved_at"],
            }
            for r in rows
        ]

    def record_audit_log(
        self,
        user_id: Optional[str],
        thread_id: Optional[str],
        tool_name: str,
        action_name: str,
        policy_evaluated: str,
        risk_level: str,
        status: str,
        actor: str = "system",
        input_payload: Optional[Any] = None,
        output_payload: Optional[Any] = None,
    ) -> Dict[str, Any]:
        log_id = f"aud-{uuid.uuid4().hex[:12]}"
        now = _utc_now()
        in_json = json.dumps(_redact_sensitive_payload(input_payload if input_payload is not None else {}), ensure_ascii=False)
        out_json = json.dumps(_redact_sensitive_payload(output_payload if output_payload is not None else {}), ensure_ascii=False)

        with self._connection() as connection:
            connection.execute(
                """
                INSERT INTO audit_logs (
                    id, user_id, thread_id, tool_name, action_name,
                    policy_evaluated, risk_level, status, actor,
                    input_payload_json, output_payload_json, created_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    log_id,
                    user_id,
                    thread_id,
                    tool_name,
                    action_name,
                    policy_evaluated,
                    risk_level,
                    status,
                    actor,
                    in_json,
                    out_json,
                    now,
                ),
            )

        return {
            "id": log_id,
            "user_id": user_id,
            "thread_id": thread_id,
            "tool_name": tool_name,
            "action_name": action_name,
            "policy_evaluated": policy_evaluated,
            "risk_level": risk_level,
            "status": status,
            "actor": actor,
            "input_payload": input_payload,
            "output_payload": output_payload,
            "created_at": now,
        }

    def list_audit_logs(
        self,
        user_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        query = "SELECT * FROM audit_logs"
        params: List[Any] = []
        if user_id:
            query += " WHERE user_id = %s"
            params.append(user_id)
        query += " ORDER BY created_at DESC LIMIT %s"
        params.append(max(1, min(limit, 500)))

        with self._connection() as connection:
            rows = connection.execute(query, params).fetchall()

        return [
            {
                "id": r["id"],
                "user_id": r["user_id"],
                "thread_id": r["thread_id"],
                "tool_name": r["tool_name"],
                "action_name": r["action_name"],
                "policy_evaluated": r["policy_evaluated"],
                "risk_level": r["risk_level"],
                "status": r["status"],
                "actor": r["actor"],
                "input_payload": json.loads(r["input_payload_json"] or "{}"),
                "output_payload": json.loads(r["output_payload_json"] or "{}"),
                "created_at": r["created_at"],
            }
            for r in rows
        ]
