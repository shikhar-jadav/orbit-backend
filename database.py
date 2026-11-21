"""PostgreSQL connection pool and versioned SQL migrations."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Iterator
from contextlib import contextmanager

import psycopg
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool


class PostgresDatabase:
    def __init__(
        self, database_url: str | None = None, *, schema: str | None = None
    ):
        url = (database_url or os.getenv("DATABASE_URL", "")).strip()
        if not url:
            raise RuntimeError("DATABASE_URL must be configured for PostgreSQL.")
        if url.startswith("postgres://"):
            url = "postgresql://" + url[len("postgres://") :]

        self.connection_string = url
        if schema is not None and not re.fullmatch(r"[a-z][a-z0-9_]*", schema):
            raise ValueError("PostgreSQL schema names must be lowercase identifiers.")
        self.connection_options = (
            f"-c search_path={schema}" if schema is not None else None
        )
        pool_kwargs = {"row_factory": dict_row}
        if self.connection_options:
            pool_kwargs["options"] = self.connection_options
        self.pool = ConnectionPool(
            conninfo=url,
            kwargs=pool_kwargs,
            min_size=1,
            max_size=int(os.getenv("DATABASE_POOL_MAX_SIZE", "10")),
            open=False,
        )
        self.migrations_dir = Path(__file__).resolve().parent / "migrations"

    def open(self) -> None:
        self.pool.open(wait=True)

    def close(self) -> None:
        self.pool.close()

    def connect(self, **kwargs):
        if self.connection_options:
            kwargs.setdefault("options", self.connection_options)
        return psycopg.connect(self.connection_string, **kwargs)

    @contextmanager
    def connection(self) -> Iterator:
        with self.pool.connection() as connection:
            yield connection

    def migrate(self) -> None:
        with self.connection() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    version TEXT PRIMARY KEY,
                    applied_at TIMESTAMPTZ NOT NULL DEFAULT now()
                )
                """
            )
            applied = {
                row["version"]
                for row in connection.execute(
                    "SELECT version FROM schema_migrations"
                ).fetchall()
            }

        for migration_path in sorted(self.migrations_dir.glob("*.sql")):
            version = migration_path.name
            if version in applied:
                continue
            statements = _split_sql_statements(
                migration_path.read_text(encoding="utf-8")
            )
            with self.connection() as connection:
                for statement in statements:
                    connection.execute(statement)
                connection.execute(
                    "INSERT INTO schema_migrations (version) VALUES (%s)",
                    (version,),
                )


def _split_sql_statements(script: str) -> list[str]:
    """Split migration SQL at semicolons outside quoted strings and comments."""
    statements: list[str] = []
    current: list[str] = []
    quote: str | None = None
    line_comment = False
    block_comment = False
    index = 0

    while index < len(script):
        char = script[index]
        next_char = script[index + 1] if index + 1 < len(script) else ""

        if line_comment:
            current.append(char)
            if char == "\n":
                line_comment = False
        elif block_comment:
            current.append(char)
            if char == "*" and next_char == "/":
                current.append(next_char)
                index += 1
                block_comment = False
        elif quote:
            current.append(char)
            if char == quote:
                if next_char == quote:
                    current.append(next_char)
                    index += 1
                else:
                    quote = None
            elif char == "\\" and next_char:
                current.append(next_char)
                index += 1
        elif char == "-" and next_char == "-":
            current.extend((char, next_char))
            index += 1
            line_comment = True
        elif char == "/" and next_char == "*":
            current.extend((char, next_char))
            index += 1
            block_comment = True
        elif char in {"'", '"'}:
            quote = char
            current.append(char)
        elif char == ";":
            statement = "".join(current).strip()
            if statement:
                statements.append(statement)
            current.clear()
        else:
            current.append(char)
        index += 1

    statement = "".join(current).strip()
    if statement:
        statements.append(statement)
    if quote or block_comment:
        raise ValueError("Unterminated quoted string or block comment in migration.")
    return statements
