"""Isolated PostgreSQL schemas for database-backed tests."""

from __future__ import annotations

import os
import unittest
from uuid import uuid4

import psycopg
from psycopg import sql

from database import PostgresDatabase, _split_sql_statements


class IsolatedPostgresDatabase:
    def __init__(self) -> None:
        database_url = os.getenv("TEST_DATABASE_URL", "").strip()
        if not database_url:
            raise unittest.SkipTest("Set TEST_DATABASE_URL to run PostgreSQL tests.")

        schema = f"orbit_test_{uuid4().hex}"
        normalized_url = database_url
        if normalized_url.startswith("postgres://"):
            normalized_url = "postgresql://" + normalized_url[len("postgres://") :]

        with psycopg.connect(normalized_url, autocommit=True) as connection:
            connection.execute(
                sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema))
            )

        self.database = PostgresDatabase(normalized_url, schema=schema)
        self.schema = schema
        self.database.open()
        try:
            self.database.migrate()
        except Exception:
            self.close()
            raise

    def close(self) -> None:
        self.database.close()
        with psycopg.connect(
            self.database.connection_string, autocommit=True
        ) as connection:
            connection.execute(
                sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(
                    sql.Identifier(self.schema)
                )
            )

    def __enter__(self) -> PostgresDatabase:
        return self.database

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


def test_migration_splitter_preserves_quoted_semicolons_and_comments():
    statements = _split_sql_statements(
        "-- ignored ;\n"
        "CREATE TABLE sample (value TEXT DEFAULT 'first;second');"
        "/* another ; */ INSERT INTO sample VALUES ('done');"
    )

    assert len(statements) == 2
    assert "'first;second'" in statements[0]
    assert "/* another ; */" in statements[1]
