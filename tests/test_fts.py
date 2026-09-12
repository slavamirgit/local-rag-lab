from contextlib import closing
import os
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from src.rag import fts


class FTSTests(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "nested" / "fts_index.db"

    def build(self, texts):
        # Per-document chunk IDs deliberately differ from global list positions.
        chunks = [{"text": text, "source": f"source_{i}.txt", "chunk_id": 0}
                  for i, text in enumerate(texts)]
        fts.build_fts_index(chunks, self.path)
        return chunks

    def search(self, inputs, limit=None):
        return fts.search_fts(inputs, limit=limit, index_path=self.path)

    def test_persistent_index_maps_to_supplied_list_positions(self):
        chunks = self.build(["zirconium", "hafnium", "tantalum"])
        self.assertTrue(self.path.is_file())
        for position, chunk in enumerate(chunks):
            with self.subTest(position=position):
                self.assertEqual(self.search(chunk["text"]), [position])
                self.assertEqual(chunks[self.search(chunk["text"])[0]], chunk)
        with closing(sqlite3.connect(self.path)) as connection:
            self.assertEqual(connection.execute(
                "SELECT rowid - 1, text FROM chunks_fts ORDER BY rowid"
            ).fetchall(), list(enumerate(chunk["text"] for chunk in chunks)))

    def test_technical_terms_and_unicode(self):
        self.build([
            "QW_E17_D2 queue notice",
            "NQ-7B recovery hold",
            "src/relay-routes.v2b.yaml manifest",
            "retry_ack_ms configuration key",
            "rivetctl recover --apply --only-pending",
            "Привет мир café 中文",
        ])
        for term, position in (
            ("QW_E17_D2", 0), ("NQ-7B", 1), ("src/relay-routes.v2b.yaml", 2),
            ("retry_ack_ms", 3), ("rivetctl recover --apply --only-pending", 4),
            ("мир", 5), ("café", 5), ("中文", 5),
        ):
            with self.subTest(term=term), self.assertNoLogs(fts.logger, level="WARNING"):
                self.assertEqual(self.search(term), [position])

    def test_punctuation_and_fts_operators_are_safe(self):
        self.build(["needle", "unrelated"])
        for query in (
            'needle_:--./()"\'[]{}*^:+!?', '"needle" OR (',
            'text:needle NOT missing', 'NEAR(needle, 5)',
            'needle"; DROP TABLE chunks_fts; --',
        ):
            with self.subTest(query=query), self.assertNoLogs(fts.logger, level="WARNING"):
                self.assertEqual(self.search(query), [0])
        self.assertEqual(self.search("unrelated"), [1])

    def test_or_tokens_favor_recall(self):
        self.build(["rarecomponent", "othercomponent", "unrelated"])
        self.assertEqual(self.search("rarecomponent/othercomponent.missing"), [0, 1])

    def test_bm25_ranking(self):
        # Equal document lengths isolate term-frequency ranking; list order differs.
        self.build(["needle filler filler", "needle needle needle", "unrelated filler filler"])
        self.assertEqual(self.search("needle"), [1, 0])

    def test_ties_are_ordered_by_position(self):
        self.build(["same text"] * 4)
        for _ in range(3):
            self.assertEqual(self.search("same"), [0, 1, 2, 3])

    def test_default_and_explicit_candidate_limits(self):
        self.build(["needle"] * (fts.FTS_CANDIDATES + 3))
        self.assertEqual(self.search("needle"), list(range(fts.FTS_CANDIDATES)))
        self.assertEqual(self.search("needle", limit=2), [0, 1])

    def test_duplicate_inputs_and_tokens_do_not_change_ranking(self):
        self.build(["alpha", "beta", "alpha beta"])
        expected = self.search(["alpha", "beta"])
        self.assertEqual(self.search(["alpha", "beta", "alpha", "alpha alpha"]), expected)
        self.assertEqual(self.search(("alpha", "beta")), expected)
        self.assertEqual(len(expected), len(set(expected)))

    def test_blank_and_punctuation_only_inputs(self):
        for inputs in ("", " \t\n", [], ["", " "], "_-./:\"'()"):
            with self.subTest(inputs=inputs), self.assertNoLogs(fts.logger, level="WARNING"):
                self.assertEqual(self.search(inputs), [])
        self.assertFalse(self.path.exists())

    def test_nonpositive_limits(self):
        self.build(["needle"])
        for limit in (0, -1):
            self.assertEqual(self.search("needle", limit=limit), [])

    def test_empty_index(self):
        self.build([])
        with self.assertNoLogs(fts.logger, level="WARNING"):
            self.assertEqual(self.search("needle"), [])

    def test_rebuild_removes_stale_documents_and_resets_positions(self):
        self.build(["oldtoken", "retainedtoken"])
        self.build(["retainedtoken"])
        self.assertEqual(self.search("oldtoken"), [])
        self.assertEqual(self.search("retainedtoken"), [0])
        self.build([])
        with self.assertNoLogs(fts.logger, level="WARNING"):
            self.assertEqual(self.search("retainedtoken"), [])

    def test_failed_build_raises_and_preserves_previous_index(self):
        self.build(["oldtoken"])
        with self.assertRaises(KeyError):
            fts.build_fts_index([{"text": "newtoken"}, {}], self.path)
        self.assertEqual(self.search("oldtoken"), [0])
        self.assertEqual(self.search("newtoken"), [])

    def test_missing_index_logs_and_does_not_create_database(self):
        with self.assertLogs(fts.logger, level="WARNING") as logs:
            self.assertEqual(self.search("needle"), [])
        self.assertIn(str(self.path), logs.output[0])
        self.assertFalse(self.path.exists())
        self.assertFalse(self.path.parent.exists())

    def test_corrupt_index(self):
        self.path.parent.mkdir()
        self.path.write_bytes(b"not a SQLite database")
        with self.assertLogs(fts.logger, level="WARNING"):
            self.assertEqual(self.search("needle"), [])

    def test_missing_fts_table(self):
        self.path.parent.mkdir()
        with closing(sqlite3.connect(self.path)) as connection:
            connection.execute("CREATE TABLE unrelated (text TEXT)")
        with self.assertLogs(fts.logger, level="WARNING"):
            self.assertEqual(self.search("needle"), [])

    def test_sqlite_query_error(self):
        self.path.parent.mkdir()
        with closing(sqlite3.connect(self.path)) as connection:
            connection.execute("CREATE TABLE chunks_fts (text TEXT)")
        with self.assertLogs(fts.logger, level="WARNING"):
            self.assertEqual(self.search("needle"), [])

    def test_relative_configured_path_uses_current_working_directory(self):
        original_cwd = Path.cwd()
        with TemporaryDirectory() as directory:
            try:
                os.chdir(directory)
                with patch.object(fts, "FTS_INDEX_PATH", "relative_index.db"):
                    fts.build_fts_index([{"text": "needle"}])
                    self.assertTrue((Path(directory) / "relative_index.db").is_file())
                    self.assertEqual(fts.search_fts("needle"), [0])
            finally:
                os.chdir(original_cwd)

    def test_configured_path_and_uri_special_characters(self):
        path = self.path.parent / "index #1?.db"
        with patch.object(fts, "FTS_INDEX_PATH", str(path)):
            fts.build_fts_index([{"text": "needle"}])
            self.assertEqual(fts.search_fts("needle"), [0])


if __name__ == "__main__":
    unittest.main()
