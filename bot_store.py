"""Persistent per-user named bot profiles and bot-owned chat threads."""
from __future__ import annotations

import secrets
from datetime import datetime, timezone
from typing import Any, Optional
from uuid import uuid4

from database import PostgresDatabase


class BotStore:
    def __init__(self, database: PostgresDatabase):
        self.database = database

    def create_bot(self, user_id: str, name: str, instructions: str) -> dict[str, Any]:
        bot_id = f"bot_{uuid4().hex}"
        now = datetime.now(timezone.utc)
        with self.database.connection() as connection:
            row = connection.execute(
                """
                INSERT INTO bots (id, user_id, name, instructions, created_at, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s)
                RETURNING id, user_id, name, instructions, created_at, updated_at
                """,
                (bot_id, user_id, name.strip(), instructions.strip(), now, now),
            ).fetchone()
        return dict(row)

    def list_bots(self, user_id: str) -> list[dict[str, Any]]:
        with self.database.connection() as connection:
            rows = connection.execute(
                """SELECT id, user_id, name, instructions, created_at, updated_at
                   FROM bots WHERE user_id = %s ORDER BY created_at ASC""",
                (user_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_bot(self, user_id: str, bot_id: str) -> Optional[dict[str, Any]]:
        with self.database.connection() as connection:
            row = connection.execute(
                """SELECT id, user_id, name, instructions, created_at, updated_at
                   FROM bots WHERE user_id = %s AND id = %s""",
                (user_id, bot_id),
            ).fetchone()
        return dict(row) if row else None

    def update_bot(self, user_id: str, bot_id: str, *, name: Optional[str] = None,
                   instructions: Optional[str] = None) -> Optional[dict[str, Any]]:
        if name is None and instructions is None:
            return self.get_bot(user_id, bot_id)
        with self.database.connection() as connection:
            row = connection.execute(
                """UPDATE bots SET name = COALESCE(%s, name),
                       instructions = COALESCE(%s, instructions), updated_at = NOW()
                   WHERE user_id = %s AND id = %s
                   RETURNING id, user_id, name, instructions, created_at, updated_at""",
                (name.strip() if name is not None else None,
                 instructions.strip() if instructions is not None else None,
                 user_id, bot_id),
            ).fetchone()
        return dict(row) if row else None

    def delete_bot(self, user_id: str, bot_id: str) -> bool:
        with self.database.connection() as connection:
            result = connection.execute(
                "DELETE FROM bots WHERE user_id = %s AND id = %s",
                (user_id, bot_id),
            )
        return result.rowcount > 0

    def create_bot_thread(self, user_id: str, bot_id: str) -> Optional[dict[str, Any]]:
        bot = self.get_bot(user_id, bot_id)
        if not bot:
            return None
        now = datetime.now(timezone.utc)
        with self.database.connection() as connection:
            while True:
                thread_id = str(secrets.randbelow(900_000_000_000) + 100_000_000_000)
                inserted = connection.execute(
                    """INSERT INTO threads (thread_id, title, created_at, updated_at, user_id)
                       VALUES (%s, %s, %s, %s, %s)
                       ON CONFLICT (thread_id) DO NOTHING RETURNING thread_id""",
                    (thread_id, bot["name"], now, now, user_id),
                ).fetchone()
                if inserted:
                    break
            connection.execute(
                """INSERT INTO bot_threads (thread_id, bot_id, user_id, created_at)
                   VALUES (%s, %s, %s, %s)""",
                (thread_id, bot_id, user_id, now),
            )
        return {"thread_id": thread_id, "title": bot["name"], "created_at": now,
                "updated_at": now, "preview": "", "message_count": 0,
                "user_id": user_id, "bot_id": bot_id}

    def get_latest_thread(self, user_id: str, bot_id: str) -> Optional[dict[str, Any]]:
        with self.database.connection() as connection:
            row = connection.execute(
                """SELECT t.thread_id, t.title, t.created_at, t.updated_at, t.user_id,
                          COALESCE((SELECT LEFT(m.content, 180) FROM messages m
                                    WHERE m.thread_id = t.thread_id AND m.role = 'user'
                                    ORDER BY m.id DESC LIMIT 1), '') AS preview,
                          (SELECT COUNT(*) FROM messages m WHERE m.thread_id = t.thread_id) AS message_count,
                          bt.bot_id
                   FROM bot_threads bt JOIN threads t ON t.thread_id = bt.thread_id
                   WHERE bt.user_id = %s AND bt.bot_id = %s
                   ORDER BY t.updated_at DESC LIMIT 1""",
                (user_id, bot_id),
            ).fetchone()
        return dict(row) if row else None

    def get_bot_for_thread(self, user_id: str, thread_id: str) -> Optional[dict[str, Any]]:
        with self.database.connection() as connection:
            row = connection.execute(
                """SELECT b.id, b.user_id, b.name, b.instructions, b.created_at, b.updated_at
                   FROM bot_threads bt JOIN bots b ON b.id = bt.bot_id
                   WHERE bt.user_id = %s AND bt.thread_id = %s""",
                (user_id, thread_id),
            ).fetchone()
        return dict(row) if row else None
