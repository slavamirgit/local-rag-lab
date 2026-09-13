"""SQLite persistence for ordered generated chunks and their derived FTS index."""

from contextlib import closing
from pathlib import Path
import sqlite3


def _rebuild_fts(connection: sqlite3.Connection) -> None:
    connection.execute("INSERT INTO chunks_fts(chunks_fts) VALUES ('rebuild')")


def build_rag_db(chunks: list[dict], db_path: str | Path) -> None:
    """Replace chunk storage and FTS at the supplied path in one transaction.

    Global IDs are zero-based list positions; chunk_id stays the per-source
    ordinal. Construction errors propagate after rolling back the transaction.
    """
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(path)) as connection, connection:
        # Begin before DDL so a failed build also rolls back schema replacement.
        connection.execute("BEGIN")
        connection.execute("DROP TABLE IF EXISTS chunks_fts")
        connection.execute("DROP TABLE IF EXISTS chunks")
        connection.execute("""
            CREATE TABLE chunks (
                id INTEGER PRIMARY KEY,
                text TEXT NOT NULL,
                source TEXT NOT NULL,
                chunk_id INTEGER NOT NULL
            )
        """)
        connection.execute("""
            CREATE VIRTUAL TABLE chunks_fts USING fts5(
                text,
                content='chunks',
                content_rowid='id',
                tokenize='unicode61'
            )
        """)
        connection.executemany(
            "INSERT INTO chunks(id, text, source, chunk_id) VALUES (?, ?, ?, ?)",
            ((position, chunk["text"], chunk["source"], chunk["chunk_id"])
             for position, chunk in enumerate(chunks)),
        )
        _rebuild_fts(connection)


def load_chunks(db_path: str | Path) -> list[dict]:
    """Load public chunk dictionaries in global ID order, without creating a DB.

    SQLite errors propagate; invalid global IDs raise ValueError instead of
    returning a mapping that could resolve vector positions to the wrong chunks.
    """
    uri = Path(db_path).resolve().as_uri() + "?mode=ro"
    with closing(sqlite3.connect(uri, uri=True)) as connection:
        rows = connection.execute(
            "SELECT id, text, source, chunk_id FROM chunks ORDER BY id"
        ).fetchall()
    chunks = []
    for expected_id, (global_id, text, source, chunk_id) in enumerate(rows):
        if global_id != expected_id:
            raise ValueError(
                f"Invalid chunk identity: expected global id {expected_id}, got {global_id}"
            )
        chunks.append({"text": text, "source": source, "chunk_id": chunk_id})
    return chunks
