from contextlib import chdir, closing
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from src.rag import store


class StoreTests(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)
        self.path = self.directory / "nested" / "rag.db"
        self.chunks = [
            {"text": "zirconium café", "source": "z.txt", "chunk_id": 0},
            {"text": "hafnium\nsecond line", "source": "z.txt", "chunk_id": 1},
            {"text": "tantalum 中文", "source": "a.txt", "chunk_id": 0},
        ]

    def match(self, token):
        with closing(sqlite3.connect(self.path)) as connection:
            return connection.execute(
                "SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH ? ORDER BY rowid",
                (token,),
            ).fetchall()

    def assert_old_generation(self):
        self.assertEqual(store.load_chunks(self.path), self.chunks)
        for position, token in enumerate(("zirconium", "hafnium", "tantalum")):
            self.assertEqual(self.match(token), [(position,)])
        self.assertEqual(self.match("newtoken"), [])

    def test_schema_and_round_trip(self):
        store.build_rag_db(self.chunks, self.path)
        with closing(sqlite3.connect(self.path)) as connection:
            self.assertEqual(connection.execute("PRAGMA table_info(chunks)").fetchall(), [
                (0, "id", "INTEGER", 0, None, 1),
                (1, "text", "TEXT", 1, None, 0),
                (2, "source", "TEXT", 1, None, 0),
                (3, "chunk_id", "INTEGER", 1, None, 0),
            ])
            self.assertEqual(connection.execute(
                "SELECT id, text, source, chunk_id FROM chunks ORDER BY id"
            ).fetchall(), [
                (i, chunk["text"], chunk["source"], chunk["chunk_id"])
                for i, chunk in enumerate(self.chunks)
            ])
            fts_schema, = connection.execute(
                "SELECT sql FROM sqlite_master WHERE name = 'chunks_fts'"
            ).fetchone()
            self.assertEqual("".join(fts_schema.split()),
                             "CREATEVIRTUALTABLEchunks_ftsUSINGfts5("
                             "text,content='chunks',content_rowid='id',tokenize='unicode61')")
            self.assertEqual(connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'trigger'"
            ).fetchall(), [])
        loaded = store.load_chunks(self.path)
        self.assertEqual(loaded, self.chunks)
        self.assertTrue(all(type(chunk) is dict for chunk in loaded))

    def test_zero_id_is_searchable_and_resolvable(self):
        store.build_rag_db(self.chunks, self.path)
        self.assertEqual(self.match("zirconium"), [(0,)])
        with closing(sqlite3.connect(self.path)) as connection:
            self.assertEqual(connection.execute(
                "SELECT chunks.id, chunks.text, chunks.source, chunks.chunk_id "
                "FROM chunks_fts JOIN chunks ON chunks.id = chunks_fts.rowid "
                "WHERE chunks_fts MATCH ?", ("zirconium",),
            ).fetchall(), [(0, "zirconium café", "z.txt", 0)])
        self.assertEqual(store.load_chunks(self.path)[0], self.chunks[0])

    def test_external_content_requires_explicit_rebuild_for_match(self):
        with closing(sqlite3.connect(":memory:")) as connection:
            connection.execute(
                "CREATE TABLE chunks (id INTEGER PRIMARY KEY, text TEXT NOT NULL, "
                "source TEXT NOT NULL, chunk_id INTEGER NOT NULL)"
            )
            connection.execute(
                "CREATE VIRTUAL TABLE chunks_fts USING fts5("
                "text, content='chunks', content_rowid='id', tokenize='unicode61')"
            )
            connection.execute(
                "INSERT INTO chunks VALUES (0, 'needle', 'document.txt', 7)"
            )
            query = "SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH ?"
            self.assertEqual(connection.execute(query, ("needle",)).fetchall(), [])
            connection.execute("INSERT INTO chunks_fts(chunks_fts) VALUES ('rebuild')")
            self.assertEqual(connection.execute(query, ("needle",)).fetchall(), [(0,)])

    def test_insertion_failure_rolls_back_old_generation(self):
        store.build_rag_db(self.chunks, self.path)
        # executemany inserts the first row before the second violates NOT NULL.
        new_chunks = [
            {"text": "newtoken", "source": "new.txt", "chunk_id": 0},
            {"text": None, "source": "new.txt", "chunk_id": 1},
        ]
        with self.assertRaises(sqlite3.IntegrityError):
            store.build_rag_db(new_chunks, self.path)
        self.assert_old_generation()

    def test_fts_construction_failure_rolls_back_old_generation(self):
        store.build_rag_db(self.chunks, self.path)
        rebuild_fts = store._rebuild_fts
        error = sqlite3.OperationalError("injected FTS construction failure")

        def fail_fts(connection):
            self.assertTrue(connection.in_transaction)
            self.assertEqual(connection.execute(
                "SELECT id, text, source, chunk_id FROM chunks"
            ).fetchall(), [(0, "newtoken", "new.txt", 9)])
            # A separate reader must still see the complete old generation.
            self.assert_old_generation()
            if after_rebuild:
                rebuild_fts(connection)
                self.assertEqual(connection.execute(
                    "SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH 'newtoken'"
                ).fetchall(), [(0,)])
            raise error

        for after_rebuild in (False, True):
            with self.subTest(after_rebuild=after_rebuild):
                with patch.object(store, "_rebuild_fts", side_effect=fail_fts) as rebuild:
                    with self.assertRaises(sqlite3.OperationalError) as raised:
                        store.build_rag_db([
                            {"text": "newtoken", "source": "new.txt", "chunk_id": 9}
                        ], self.path)
                    self.assertIs(raised.exception, error)
                    rebuild.assert_called_once()
                self.assert_old_generation()

    def test_failed_first_build_rolls_back_schema(self):
        with patch.object(store, "_rebuild_fts", side_effect=RuntimeError("FTS failure")):
            with self.assertRaisesRegex(RuntimeError, "FTS failure"):
                store.build_rag_db(self.chunks, self.path)
        with closing(sqlite3.connect(self.path)) as connection:
            self.assertEqual(connection.execute(
                "SELECT name FROM sqlite_master"
            ).fetchall(), [])

    def test_successful_rebuild_replaces_rows_and_fts(self):
        store.build_rag_db(self.chunks, self.path)
        replacement = [{"text": "newtoken", "source": "new.txt", "chunk_id": 4}]
        store.build_rag_db(replacement, self.path)
        self.assertEqual(store.load_chunks(self.path), replacement)
        self.assertEqual(self.match("newtoken"), [(0,)])
        for token in ("zirconium", "hafnium", "tantalum"):
            self.assertEqual(self.match(token), [])

    def test_empty_generation(self):
        store.build_rag_db(self.chunks, self.path)
        store.build_rag_db([], self.path)
        self.assertEqual(store.load_chunks(self.path), [])
        self.assertEqual(self.match("zirconium"), [])

    def test_missing_database_is_not_created(self):
        for parent_exists in (False, True):
            with self.subTest(parent_exists=parent_exists):
                if parent_exists:
                    self.path.parent.mkdir()
                with self.assertRaises(sqlite3.OperationalError):
                    store.load_chunks(self.path)
                self.assertFalse(self.path.exists())
                self.assertEqual(self.path.parent.exists(), parent_exists)

    def test_corrupt_database_raises(self):
        self.path.parent.mkdir()
        self.path.write_bytes(b"not a SQLite database")
        with self.assertRaises(sqlite3.DatabaseError):
            store.load_chunks(self.path)

    def test_missing_chunks_table_raises(self):
        self.path.parent.mkdir()
        with closing(sqlite3.connect(self.path)) as connection:
            connection.execute("CREATE TABLE unrelated (text TEXT)")
        with self.assertRaises(sqlite3.OperationalError):
            store.load_chunks(self.path)

    def test_noncontiguous_global_ids_are_rejected(self):
        for ids in ((1, 2, 3), (0, 2, 3), (-1, 0, 1)):
            with self.subTest(ids=ids):
                store.build_rag_db(self.chunks, self.path)
                with closing(sqlite3.connect(self.path)) as connection, connection:
                    connection.execute("DELETE FROM chunks")
                    connection.executemany(
                        "INSERT INTO chunks VALUES (?, 'text', 'source.txt', 0)",
                        ((global_id,) for global_id in reversed(ids)),
                    )
                with self.assertRaisesRegex(ValueError, "Invalid chunk identity"):
                    store.load_chunks(self.path)

    def test_exact_relative_path_with_uri_characters(self):
        relative_path = "custom/store #1?.db"
        with chdir(self.directory):
            store.build_rag_db(self.chunks, relative_path)
            self.assertEqual(store.load_chunks(relative_path), self.chunks)
        self.assertEqual(
            list(self.directory.rglob("*.db")), [self.directory / relative_path]
        )


if __name__ == "__main__":
    unittest.main()
