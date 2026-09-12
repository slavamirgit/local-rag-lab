"""Hybrid integration tests with real threads and no external model calls."""

from contextlib import ExitStack
import importlib.util
from pathlib import Path
import pickle
import sys
from tempfile import TemporaryDirectory
from threading import Barrier, Event
from types import ModuleType
import unittest
from unittest.mock import Mock, patch

import faiss
import numpy as np


class HybridQueryTests(unittest.TestCase):
    def setUp(self):
        stack = ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(patch.object(sys, "path", sys.path.copy()))
        root = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(root / "src"))
        import config

        directory = Path(stack.enter_context(TemporaryDirectory()))
        index_path = directory / "index.faiss"
        index_path.touch()
        chunks_path = directory / "chunks.pkl"
        self.chunks = [
            {"text": f"chunk {i}", "source": f"source-{i}.txt", "chunk_id": 0}
            for i in range(8)
        ]
        chunks_path.write_bytes(pickle.dumps(self.chunks))
        embedding = ModuleType("sentence_transformers")
        embedding.SentenceTransformer = Mock()
        with (
            patch.dict(sys.modules, {"sentence_transformers": embedding}),
            patch.object(config, "FAISS_INDEX_PATH", str(index_path)),
            patch.object(config, "CHUNKS_PATH", str(chunks_path)),
            patch.object(faiss, "read_index", return_value=Mock(ntotal=8)),
        ):
            spec = importlib.util.spec_from_file_location(
                "hybrid_query_under_test", root / "src/rag/query.py"
            )
            self.query = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(self.query)
        self.chunks = self.query.chunks
        self.expand = stack.enter_context(patch.object(
            self.query, "expand_query", return_value=["original query", "term"]
        ))
        self.fts = stack.enter_context(patch.object(self.query, "search_fts", return_value=[]))
        self.answer = stack.enter_context(patch.object(
            self.query, "ask_llm", side_effect=AssertionError("answer generation forbidden")
        ))
        self.post = stack.enter_context(patch.object(
            self.query.requests, "post", side_effect=AssertionError("HTTP forbidden")
        ))

    def tearDown(self):
        self.answer.assert_not_called()
        self.post.assert_not_called()

    def test_inputs_depths_fusion_order_and_original_chunk_objects(self):
        original = "What must we fix before connecting the worker again?"
        expanded = [original, "worker", "connecting"]
        self.expand.return_value = expanded
        self.fts.return_value = [4, 2, 1]
        with (
            patch.object(self.query, "_vector_search_ids", return_value=[3, 1, 2]) as vector,
            patch.object(self.query, "reciprocal_rank_fusion", return_value=[2, 4, 3]) as fusion,
        ):
            result = self.query.retrieve(original)
        self.expand.assert_called_once_with(original)
        vector.assert_called_once_with(original, self.query.VECTOR_CANDIDATES)
        self.fts.assert_called_once_with(expanded, limit=self.query.FTS_CANDIDATES)
        fusion.assert_called_once_with([[3, 1, 2], [4, 2, 1]], limit=self.query.TOP_K)
        for actual, position in zip(result, [2, 4, 3]):
            self.assertIs(actual, self.chunks[position])
            self.assertIsInstance(actual, dict)
        self.assertEqual(len(result), 3)

    def test_real_fusion_enforces_top_k(self):
        with patch.object(self.query, "_vector_search_ids", return_value=list(range(8))):
            result = self.query.retrieve("original query")
        self.assertEqual(len(result), self.query.TOP_K)
        self.assertEqual(result, self.chunks[:self.query.TOP_K])

    def test_searches_overlap_and_start_after_expansion(self):
        expanded = Event()
        rendezvous = Barrier(2, timeout=5)
        vector_finished = Event()
        fts_finished = Event()

        def expand(query):
            expanded.set()
            return [query, "term"]

        def search(finished):
            def run(*args, **kwargs):
                self.assertTrue(expanded.is_set())
                rendezvous.wait()
                finished.set()
                return [0]
            return run

        self.expand.side_effect = expand
        self.fts.side_effect = search(fts_finished)
        with patch.object(self.query, "_vector_search_ids", side_effect=search(vector_finished)):
            result = self.query.retrieve("original query")
        # These assertions also detect failures swallowed at the branch boundary.
        self.assertTrue(vector_finished.is_set())
        self.assertTrue(fts_finished.is_set())
        self.assertEqual(result, [self.chunks[0]])

    def test_expansion_exception_falls_back_without_retry(self):
        self.expand.side_effect = RuntimeError("expansion unavailable")
        with (
            patch.object(self.query, "_vector_search_ids", return_value=[2]) as vector,
            self.assertLogs(self.query.logger, level="WARNING") as logs,
        ):
            self.assertEqual(self.query.retrieve("original query"), [self.chunks[2]])
        self.expand.assert_called_once_with("original query")
        self.fts.assert_called_once_with(["original query"], limit=self.query.FTS_CANDIDATES)
        vector.assert_called_once_with("original query", self.query.VECTOR_CANDIDATES)
        self.assertIn("expansion unavailable", " ".join(logs.output))

    def test_each_branch_can_fail_independently_through_fusion(self):
        for failing in ("vector", "fts"):
            with self.subTest(failing=failing):
                self.fts.side_effect = RuntimeError("fts unavailable") if failing == "fts" else None
                self.fts.return_value = [3, 1]
                with (
                    patch.object(self.query, "_vector_search_ids", return_value=[3, 1],
                                 side_effect=RuntimeError("vector unavailable") if failing == "vector" else None),
                    patch.object(self.query, "reciprocal_rank_fusion",
                                 wraps=self.query.reciprocal_rank_fusion) as fusion,
                    self.assertLogs(self.query.logger, level="WARNING") as logs,
                ):
                    self.assertEqual(self.query.retrieve("original query"), [self.chunks[3], self.chunks[1]])
                rankings = [[], [3, 1]] if failing == "vector" else [[3, 1], []]
                fusion.assert_called_once_with(rankings, limit=self.query.TOP_K)
                self.assertIn(f"{failing} unavailable", " ".join(logs.output))

    def test_empty_branch_retains_other_ranking(self):
        for vector_ids, fts_ids in (([3, 1], []), ([], [3, 1])):
            with self.subTest(vector=vector_ids, fts=fts_ids):
                self.fts.return_value = fts_ids
                with patch.object(self.query, "_vector_search_ids", return_value=vector_ids):
                    self.assertEqual(self.query.retrieve("query"), [self.chunks[3], self.chunks[1]])

    def test_both_branches_empty_or_failed(self):
        for vector_fails in (False, True):
            for fts_fails in (False, True):
                with self.subTest(vector_fails=vector_fails, fts_fails=fts_fails):
                    self.fts.side_effect = RuntimeError("fts") if fts_fails else None
                    with patch.object(self.query, "_vector_search_ids", return_value=[],
                                      side_effect=RuntimeError("vector") if vector_fails else None):
                        self.assertEqual(self.query.retrieve("query"), [])

    def test_invalid_candidates_are_filtered_before_fusion(self):
        self.fts.return_value = [-1, True, 8, 3, 3, False, 1.0, "2", None, np.int64(2)]
        with (
            patch.object(self.query, "_vector_search_ids", return_value=[7, -2, 7, 80, False, 1]),
            patch.object(self.query, "reciprocal_rank_fusion", return_value=[3, 7]) as fusion,
        ):
            result = self.query.retrieve("query")
        fusion.assert_called_once_with([[7, 1], [3, 2]], limit=self.query.TOP_K)
        self.assertIs(result[0], self.chunks[3])
        self.assertIs(result[1], self.chunks[7])

    def test_vector_only_uses_shared_search_and_no_hybrid_components(self):
        with (
            patch.object(self.query, "_vector_search_ids", return_value=[4, 2]) as vector,
            patch.object(self.query, "reciprocal_rank_fusion") as fusion,
        ):
            for limit in (None, 2, 0):
                with self.subTest(limit=limit):
                    result = self.query.retrieve_vector("original query", limit=limit)
                    vector.assert_called_with("original query", self.query.TOP_K if limit is None else limit)
                    self.assertIs(result[0], self.chunks[4])
                    self.assertIs(result[1], self.chunks[2])
        self.expand.assert_not_called()
        self.fts.assert_not_called()
        fusion.assert_not_called()

    def test_vector_search_caps_depth_and_preserves_normalization_and_order(self):
        for requested, index_count, expected_count in ((20, 3, 3), (2, 8, 2), (20, 10, 8)):
            with self.subTest(requested=requested, index_count=index_count):
                self.query.index = Mock(ntotal=index_count)
                self.query.index.search.return_value = (None, np.array([[2, 0, 1, 3, 4, 5, 6, 7]]))
                self.query.model.encode.return_value = np.array([[3, 4]], dtype="float32")
                result = self.query._vector_search_ids("  original natural-language query  ", requested)
                self.query.model.encode.assert_called_with(["  original natural-language query  "])
                embedding, count = self.query.index.search.call_args.args
                np.testing.assert_allclose(embedding, [[0.6, 0.8]])
                self.assertEqual(count, expected_count)
                self.assertEqual(result, [2, 0, 1, 3, 4, 5, 6, 7][:expected_count])

    def test_vector_search_discards_padding_and_invalid_positions(self):
        self.query.index.search.return_value = (None, np.array([[4, -1, 2, 8, -5, 4, 0, 1]]))
        self.query.model.encode.return_value = np.array([[3, 4]], dtype="float32")
        self.assertEqual(self.query._vector_search_ids("query", 20), [4, 2, 0, 1])

    def test_vector_search_skips_empty_index_and_nonpositive_limits(self):
        self.query.index.ntotal = 0
        for limit in (20, 0, -1):
            self.assertEqual(self.query._vector_search_ids("query", limit), [])
        self.query.model.encode.assert_not_called()
        self.query.index.search.assert_not_called()

    def test_vector_search_reuses_existing_loader(self):
        self.query.index = None
        with patch.object(self.query, "_ensure_index_exists", return_value=False) as loader:
            self.assertEqual(self.query._vector_search_ids("query", 20), [])
        loader.assert_called_once_with()
        self.query.model.encode.assert_not_called()


if __name__ == "__main__":
    unittest.main()
