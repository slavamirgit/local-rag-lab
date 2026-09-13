"""SQLite persistence for generated chunks, FTS, and generation identity."""

from contextlib import closing
from pathlib import Path
import re
import sqlite3


def _rebuild_fts(connection: sqlite3.Connection) -> None:
    connection.execute("INSERT INTO chunks_fts(chunks_fts) VALUES ('rebuild')")


def _validate_faiss_sha256(value: str) -> None:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("Invalid faiss_sha256: expected 64 lowercase hexadecimal characters")


def _insert_generation_metadata(
    connection: sqlite3.Connection, faiss_sha256: str, chunk_count: int
) -> None:
    connection.execute(
        "INSERT INTO generation_meta(id, faiss_sha256, chunk_count) VALUES (1, ?, ?)",
        (faiss_sha256, chunk_count),
    )


def build_rag_db(
    chunks: list[dict], db_path: str | Path, *, faiss_sha256: str | None = None
) -> None:
    """Replace chunks, FTS, and generation metadata in one transaction.

    Global IDs are zero-based list positions; chunk_id stays the per-source
    ordinal. Construction errors propagate after rolling back the transaction.
    Omitting the SHA leaves metadata empty for legacy/transitional callers.
    """
    if faiss_sha256 is not None:
        _validate_faiss_sha256(faiss_sha256)
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
        connection.execute("DROP TABLE IF EXISTS generation_meta")
        connection.execute("""
            CREATE TABLE generation_meta (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                faiss_sha256 TEXT NOT NULL,
                chunk_count INTEGER NOT NULL
            )
        """)
        connection.executemany(
            "INSERT INTO chunks(id, text, source, chunk_id) VALUES (?, ?, ?, ?)",
            ((position, chunk["text"], chunk["source"], chunk["chunk_id"])
             for position, chunk in enumerate(chunks)),
        )
        _rebuild_fts(connection)
        if faiss_sha256 is not None:
            _insert_generation_metadata(connection, faiss_sha256, len(chunks))


def load_generation_metadata(db_path: str | Path) -> dict | None:
    """Read generation identity without creating or modifying the database.

    A missing table or empty table means identity is unproven. Invalid metadata
    raises ValueError; SQLite/storage errors propagate independently of chunks.
    """
    uri = Path(db_path).resolve().as_uri() + "?mode=ro"
    with closing(sqlite3.connect(uri, uri=True)) as connection:
        if connection.execute(
            "SELECT 1 FROM sqlite_master "
            "WHERE type = 'table' AND name = 'generation_meta' COLLATE NOCASE"
        ).fetchone() is None:
            return None
        rows = connection.execute(
            "SELECT id, faiss_sha256, chunk_count FROM generation_meta"
        ).fetchmany(2)
    if not rows:
        return None
    if len(rows) != 1:
        raise ValueError("Invalid generation metadata: expected exactly one row")
    identity, faiss_sha256, chunk_count = rows[0]
    if type(identity) is not int or identity != 1:
        raise ValueError("Invalid generation metadata: expected id 1")
    _validate_faiss_sha256(faiss_sha256)
    if type(chunk_count) is not int or chunk_count < 0:
        raise ValueError("Invalid generation metadata: expected nonnegative integer chunk_count")
    return {"faiss_sha256": faiss_sha256, "chunk_count": chunk_count}


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
