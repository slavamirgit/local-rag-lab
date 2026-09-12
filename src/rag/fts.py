"""Persistent lexical search over the same ordered chunks used by FAISS."""

from collections.abc import Iterable
from contextlib import closing
import logging
from pathlib import Path
import re
import sqlite3

try:  # Support both src.rag imports and running with src on PYTHONPATH.
    from ..config import FTS_CANDIDATES, FTS_INDEX_PATH
except ImportError:
    from config import FTS_CANDIDATES, FTS_INDEX_PATH

logger = logging.getLogger(__name__)


def build_fts_index(chunks: list[dict], index_path=None) -> None:
    """Replace the index; rowid is list position + 1, not a chunk's chunk_id.

    Configured and explicit paths use normal Path semantics, relative to the cwd.
    A failed rebuild rolls back, preserving the previous index.
    """
    path = Path(index_path) if index_path is not None else Path(FTS_INDEX_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(path)) as connection, connection:
        # Explicit BEGIN also makes the schema replacement transactional.
        connection.execute("BEGIN")
        connection.execute("DROP TABLE IF EXISTS chunks_fts")
        connection.execute("CREATE VIRTUAL TABLE chunks_fts USING fts5(text, tokenize='unicode61')")
        connection.executemany(
            "INSERT INTO chunks_fts(rowid, text) VALUES (?, ?)",
            ((position + 1, chunk["text"]) for position, chunk in enumerate(chunks)),
        )


def search_fts(search_inputs: str | Iterable[str], limit=None, index_path=None) -> list[int]:
    """Return zero-based chunk positions ordered by BM25, then position.

    Inputs are OR-ed lexical tokens, never raw MATCH syntax. Runtime index
    failures return no candidates. Paths follow build_fts_index's convention.
    """
    limit = FTS_CANDIDATES if limit is None else limit
    if limit <= 0:
        return []
    if isinstance(search_inputs, str):
        search_inputs = [search_inputs]
    # Split punctuation (including underscores) for recall on technical terms.
    tokens = dict.fromkeys(token for text in search_inputs for token in re.findall(r"[^\W_]+", text))
    if not tokens:
        return []
    expression = " OR ".join('"' + token + '"' for token in tokens)
    path = Path(index_path) if index_path is not None else Path(FTS_INDEX_PATH)
    try:
        # mode=ro prevents both writes and implicit creation of a missing database.
        with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)) as connection:
            rows = connection.execute(
                "SELECT rowid - 1 FROM chunks_fts WHERE chunks_fts MATCH ? "
                "ORDER BY bm25(chunks_fts), rowid LIMIT ?",
                (expression, limit),
            ).fetchall()
        return [row[0] for row in rows]
    except (sqlite3.Error, OSError) as exc:
        logger.warning("FTS search failed for %s: %s", path, exc)
        return []
