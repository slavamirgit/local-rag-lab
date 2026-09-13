"""Persistent lexical search over the same ordered chunks used by FAISS."""

from collections.abc import Iterable
from contextlib import closing
import logging
from pathlib import Path
import re
import sqlite3

try:  # Support both src.rag imports and running with src on PYTHONPATH.
    from ..config import FTS_CANDIDATES, RAG_DB_PATH
except ImportError:
    from config import FTS_CANDIDATES, RAG_DB_PATH

logger = logging.getLogger(__name__)


def search_fts(search_inputs: str | Iterable[str], limit=None, db_path=None) -> list[int]:
    """Return zero-based chunk positions ordered by BM25, then position.

    Inputs are OR-ed lexical tokens, never raw MATCH syntax. Runtime database
    failures return no candidates. Paths use normal cwd-relative Path semantics.
    The external-content FTS rowid equals chunks.id and the FAISS position.
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
    path = Path(db_path) if db_path is not None else Path(RAG_DB_PATH)
    try:
        # mode=ro prevents both writes and implicit creation of a missing database.
        with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)) as connection:
            rows = connection.execute(
                "SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH ? "
                "ORDER BY bm25(chunks_fts), rowid LIMIT ?",
                (expression, limit),
            ).fetchall()
        return [row[0] for row in rows]
    except (sqlite3.Error, OSError) as exc:
        logger.warning("FTS search failed for %s: %s", path, exc)
        return []
