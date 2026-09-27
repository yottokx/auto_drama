"""SQLite migrations and transactions, owned exclusively by the coordinator."""

from __future__ import annotations

import hashlib
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

MIGRATIONS = Path(__file__).with_name("migrations")


class Database:
    def __init__(self, path: Path):
        self.path = path.resolve()

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=15, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA busy_timeout=15000")
        connection.execute("PRAGMA synchronous=FULL")
        return connection

    @contextmanager
    def transaction(self, *, migration: bool = False) -> Iterator[sqlite3.Connection]:
        connection = self.connect()
        try:
            # SQLite's documented table-rebuild procedure requires FK enforcement
            # disabled before BEGIN. Validate the complete graph before committing.
            if migration:
                connection.execute("PRAGMA foreign_keys=OFF")
            connection.execute("BEGIN IMMEDIATE")
            yield connection
            if migration and connection.execute("PRAGMA foreign_key_check").fetchone():
                raise RuntimeError("Migration would leave broken foreign key references")
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    def migrate(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = self.connect()
        try:
            connection.execute("PRAGMA journal_mode=WAL")
        finally:
            connection.close()
        with self.transaction(migration=True) as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS schema_migration "
                "(version INTEGER PRIMARY KEY, filename TEXT NOT NULL, sha256 TEXT NOT NULL)"
            )
            files = sorted(MIGRATIONS.glob("[0-9]*.sql"))
            applied = {
                row["version"]: row for row in connection.execute("SELECT * FROM schema_migration")
            }
            versions = {int(path.name.split("_", 1)[0]) for path in files}
            if set(applied) - versions:
                raise RuntimeError("Database schema is newer than this application")
            for path in files:
                version = int(path.name.split("_", 1)[0])
                # Git's Windows checkout may change line endings, not SQL semantics.
                source = path.read_bytes().replace(b"\r\n", b"\n")
                digest = hashlib.sha256(source).hexdigest()
                if version in applied:
                    if applied[version]["sha256"] != digest:
                        raise RuntimeError(f"Applied migration {version} has changed")
                    continue
                # execute(), unlike executescript(), does not commit the surrounding
                # transaction. A failed migration is rolled back in its entirety.
                statement = ""
                for line in source.decode("utf-8").splitlines(keepends=True):
                    statement += line
                    if sqlite3.complete_statement(statement):
                        connection.execute(statement)
                        statement = ""
                if statement.strip():
                    raise RuntimeError(f"Incomplete SQL migration: {path.name}")
                connection.execute(
                    "INSERT INTO schema_migration VALUES (?, ?, ?)", (version, path.name, digest)
                )
