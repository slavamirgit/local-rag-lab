"""Hybrid integration tests with real threads and no external model calls."""

from contextlib import ExitStack, chdir
from concurrent.futures import ThreadPoolExecutor
import importlib.util
from pathlib import Path
import pickle
import sqlite3
import sys
from tempfile import TemporaryDirectory
from threading import Barrier, Event, current_thread, main_thread
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
        from rag import store

        self.directory = directory = Path(stack.enter_context(TemporaryDirectory()))
        self.index_path = index_path = directory / "index.faiss"
        index_path.touch()
        self.chunks_path = chunks_path = directory / "chunks.pkl"
        self.rag_path = directory / "rag.db"
        self.chunks = [
            {"text": f"chunk {i}", "source": f"source-{i}.txt", "chunk_id": 0}
            for i in range(8)
        ]
        chunks_path.write_bytes(pickle.dumps(self.chunks))
        store.build_rag_db(self.chunks, self.rag_path)
        embedding = ModuleType("sentence_transformers")
        embedding.SentenceTransformer = Mock()
        self.constructor = embedding.SentenceTransformer
        self.model = self.constructor.return_value
        rebuild = ModuleType("rag.build_index")
        rebuild.build_index = Mock(side_effect=RuntimeError("injected rebuild failure"))
        self.rebuild = rebuild.build_index
        stack.enter_context(patch.dict(sys.modules, {"rag.build_index": rebuild}))
        with (
            patch.dict(sys.modules, {"sentence_transformers": embedding}),
            patch.object(config, "FAISS_INDEX_PATH", str(index_path)),
            patch.object(config, "CHUNKS_PATH", str(chunks_path)),
            patch.object(config, "RAG_DB_PATH", str(self.rag_path)),
            patch.object(faiss, "read_index", return_value=Mock(ntotal=8)) as read_index,
        ):
            spec = importlib.util.spec_from_file_location(
                "hybrid_query_under_test", root / "src/rag/query.py"
            )
            self.query = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(self.query)
            self.constructor.assert_not_called()
            read_index.assert_not_called()
            self.rebuild.assert_not_called()
            self.assertIsNone(self.query.model)
            self.assertIsNone(self.query.index)
            self.assertEqual(self.query.chunks, [])
            # Existing algorithm tests start with explicitly prepared artifacts.
            self.assertTrue(self.query._ensure_index_exists())
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
                self.model.encode.return_value = np.array([[3, 4]], dtype="float32")
                result = self.query._vector_search_ids("  original natural-language query  ", requested)
                self.query.model.encode.assert_called_with(["  original natural-language query  "])
                embedding, count = self.query.index.search.call_args.args
                np.testing.assert_allclose(embedding, [[0.6, 0.8]])
                self.assertEqual(count, expected_count)
                self.assertEqual(result, [2, 0, 1, 3, 4, 5, 6, 7][:expected_count])

    def test_vector_search_discards_padding_and_invalid_positions(self):
        self.query.index.search.return_value = (None, np.array([[4, -1, 2, 8, -5, 4, 0, 1]]))
        self.model.encode.return_value = np.array([[3, 4]], dtype="float32")
        self.assertEqual(self.query._vector_search_ids("query", 20), [4, 2, 0, 1])

    def test_vector_search_skips_empty_index_and_nonpositive_limits(self):
        self.query.index.ntotal = 0
        for limit in (20, 0, -1):
            self.assertEqual(self.query._vector_search_ids("query", limit), [])
        self.model.encode.assert_not_called()
        self.constructor.assert_not_called()
        self.query.index.search.assert_not_called()

    def test_vector_only_prepares_artifacts_before_shared_search(self):
        self.query.index = None
        with patch.object(self.query, "_ensure_index_exists", return_value=False) as loader:
            self.assertEqual(self.query.retrieve_vector("query", 20), [])
        loader.assert_called_once_with()
        self.model.encode.assert_not_called()
        self.constructor.assert_not_called()

    def test_vector_only_nonpositive_limits_do_not_initialize_artifacts(self):
        self.query.index = None
        self.query.chunks = []
        with patch.object(self.query, "_ensure_index_exists") as loader:
            for limit in (0, -1):
                self.assertEqual(self.query.retrieve_vector("query", limit), [])
        loader.assert_not_called()
        self.constructor.assert_not_called()

    def prepare_real_artifacts(self, *, corrupt_faiss=False):
        from rag import fts, store

        self.stored_chunks = [
            {"text": "unrelated background", "source": "unrelated.txt", "chunk_id": 7},
            {"text": "oldneedle recovery instructions", "source": "old.txt", "chunk_id": 8},
        ]
        self.chunks_path.write_bytes(pickle.dumps(self.stored_chunks))
        store.build_rag_db(self.stored_chunks, self.rag_path)
        if corrupt_faiss:
            self.index_path.write_bytes(b"corrupt FAISS data")
        else:
            index = faiss.IndexFlatIP(2)
            index.add(np.array([[1, 0], [0, 1]], dtype="float32"))
            faiss.write_index(index, str(self.index_path))
        self.query.chunks = []
        self.query.index = None
        self.query.model = None
        self.model.encode.side_effect = lambda inputs: np.array([[0, 4]], dtype="float32")
        self.expand.return_value = ["oldneedle"]
        self.fts.side_effect = lambda inputs, limit: fts.search_fts(
            inputs, limit=limit, db_path=self.rag_path
        )
        self.assertEqual(fts.search_fts("oldneedle", db_path=self.rag_path), [1])

    def assert_fts_chunk(self, result):
        self.assertEqual(self.query.chunks, self.stored_chunks)
        self.assertEqual(result, [self.stored_chunks[1]])
        self.assertIs(result[0], self.query.chunks[1])

    def test_initial_loading_ignores_conflicting_missing_and_corrupt_pickle(self):
        conflicting_chunks = [
            {"text": "pickle must be ignored", "source": "wrong.txt", "chunk_id": 999},
        ]
        for retrieve in (self.query.retrieve, self.query.retrieve_vector):
            for contents in (pickle.dumps(conflicting_chunks), None, b"invalid pickle"):
                with self.subTest(retrieve=retrieve.__name__, contents=contents):
                    self.prepare_real_artifacts()
                    if contents is None:
                        self.chunks_path.unlink()
                    else:
                        self.chunks_path.write_bytes(contents)
                    with patch.object(self.query, "load_chunks",
                                      wraps=self.query.load_chunks) as loader:
                        result = retrieve("oldneedle")
                    loader.assert_called_once_with(str(self.rag_path))
                    self.assertEqual(self.query.chunks, self.stored_chunks)
                    self.assertEqual(result, [self.stored_chunks[1], self.stored_chunks[0]])
                    self.assertIs(result[0], self.query.chunks[1])
                    self.assertIs(result[1], self.query.chunks[0])
        self.rebuild.assert_not_called()

    def test_populated_cache_is_reused_without_reading_database(self):
        self.prepare_real_artifacts()
        self.assertTrue(self.query._ensure_chunks_loaded())
        loaded_chunks = self.query.chunks
        with patch.object(self.query, "load_chunks") as loader:
            self.assertTrue(self.query._ensure_chunks_loaded())
            self.assertEqual(self.query.retrieve("oldneedle"),
                             [self.stored_chunks[1], self.stored_chunks[0]])
        loader.assert_not_called()
        self.assertIs(self.query.chunks, loaded_chunks)
        self.rebuild.assert_not_called()

    def test_relative_rag_path_resolves_from_working_directory(self):
        self.prepare_real_artifacts()
        with (
            chdir(self.directory),
            patch.object(self.query, "RAG_DB_PATH", "rag.db"),
            patch.object(self.query, "load_chunks", wraps=self.query.load_chunks) as loader,
        ):
            result = self.query.retrieve("oldneedle")
        loader.assert_called_once_with("rag.db")
        self.assertEqual(self.query.chunks, self.stored_chunks)
        self.assertIs(result[0], self.query.chunks[1])
        self.rebuild.assert_not_called()

    def test_invalid_database_identity_prevents_chunk_mapping(self):
        self.prepare_real_artifacts()
        with sqlite3.connect(self.rag_path) as connection:
            connection.execute("UPDATE chunks SET id = 2 WHERE id = 0")
        with self.assertLogs(self.query.logger, level="WARNING") as logs:
            self.assertEqual(self.query.retrieve("oldneedle"), [])
        self.assertIn("Invalid chunk identity", " ".join(logs.output))
        self.assertEqual(self.query.chunks, [])
        self.rebuild.assert_called_once_with()
        self.fts.assert_not_called()
        self.constructor.assert_not_called()

    def test_healthy_retrievals_search_concurrently(self):
        self.prepare_real_artifacts()
        self.query.model = self.model
        queries = ("oldneedle first", "oldneedle second")
        rendezvous = Barrier(4, timeout=5)
        finished = {(leg, query): Event() for leg in ("vector", "fts") for query in queries}
        real_encode, real_fts = self.model.encode.side_effect, self.fts.side_effect

        def encode(inputs):
            rendezvous.wait()
            finished["vector", inputs[0]].set()
            return real_encode(inputs)

        def search(inputs, limit):
            rendezvous.wait()
            finished["fts", inputs[0]].set()
            return real_fts(inputs, limit=limit)

        self.expand.side_effect = lambda query: [query]
        self.model.encode.side_effect = encode
        self.fts.side_effect = search
        with ThreadPoolExecutor(max_workers=2) as executor:
            requests = [executor.submit(self.query.retrieve, query) for query in queries]
            results = [request.result(timeout=10) for request in requests]
        # All four branches must reach the barrier; swallowed branch failures
        # cannot turn a globally serialized implementation into a passing test.
        self.assertTrue(all(event.is_set() for event in finished.values()))
        for result in results:
            self.assertEqual(result, [self.stored_chunks[1], self.stored_chunks[0]])
        self.rebuild.assert_not_called()
        self.constructor.assert_not_called()

    def test_concurrent_vector_searches_construct_model_once(self):
        self.prepare_real_artifacts()
        initialization_attempts = Barrier(2, timeout=5)
        encoding = Barrier(2, timeout=5)
        model_lock = self.query._model_lock
        real_encode = self.model.encode.side_effect

        class SynchronizedInitializationLock:
            def __enter__(self):
                # Both callers have observed model=None before either can
                # construct it. The second must re-check after taking the lock.
                initialization_attempts.wait()
                return model_lock.__enter__()

            def __exit__(self, *args):
                return model_lock.__exit__(*args)

        def encode(inputs):
            encoding.wait()
            return real_encode(inputs)

        self.model.encode.side_effect = encode
        with (
            patch.object(self.query, "_model_lock", SynchronizedInitializationLock()),
            ThreadPoolExecutor(max_workers=2) as executor,
        ):
            requests = [executor.submit(self.query.retrieve_vector, query)
                        for query in ("first", "second")]
            results = [request.result(timeout=10) for request in requests]
        self.constructor.assert_called_once_with(self.query.EMBEDDING_MODEL)
        self.assertIs(self.query.model, self.model)
        self.assertEqual(self.model.encode.call_count, 2)
        self.assertEqual({tuple(call.args[0]) for call in self.model.encode.call_args_list},
                         {("first",), ("second",)})
        for result in results:
            self.assertEqual(result, [self.stored_chunks[1], self.stored_chunks[0]])
        self.rebuild.assert_not_called()

    def test_corrupt_faiss_and_failed_rebuild_preserve_real_fts_mapping(self):
        self.prepare_real_artifacts(corrupt_faiss=True)
        with self.assertLogs(self.query.logger, level="WARNING") as logs:
            result = self.query.retrieve("oldneedle")
        self.assert_fts_chunk(result)
        self.rebuild.assert_called_once_with()
        self.constructor.assert_not_called()
        self.assertIsNone(self.query.index)
        self.assertIn("Vector index loading failed", " ".join(logs.output))
        self.assertIn("injected rebuild failure", " ".join(logs.output))

    def test_missing_faiss_and_failed_rebuild_preserve_real_fts_mapping(self):
        self.prepare_real_artifacts()
        self.index_path.unlink()
        with self.assertLogs(self.query.logger, level="WARNING"):
            self.assert_fts_chunk(self.query.retrieve("oldneedle"))
        self.rebuild.assert_called_once_with()
        self.constructor.assert_not_called()

    def test_read_index_exception_does_not_clear_loaded_chunks(self):
        self.prepare_real_artifacts()
        self.assertTrue(self.query._ensure_chunks_loaded())
        loaded_chunks = self.query.chunks
        with (
            patch.object(faiss, "read_index", side_effect=OSError("FAISS unreadable")),
            self.assertLogs(self.query.logger, level="WARNING") as logs,
        ):
            self.assert_fts_chunk(self.query.retrieve("oldneedle"))
        self.assertIs(self.query.chunks, loaded_chunks)
        self.assertIn("FAISS unreadable", " ".join(logs.output))
        self.rebuild.assert_called_once_with()

    def test_lazy_model_failure_is_vector_work_and_preserves_fts(self):
        self.prepare_real_artifacts()
        rendezvous = Barrier(2, timeout=5)
        searched = Event()
        real_fts = self.fts.side_effect

        def fail_model(name):
            self.assertIsNot(current_thread(), main_thread())
            self.assertEqual(name, self.query.EMBEDDING_MODEL)
            self.assertEqual(self.query.chunks, self.stored_chunks)
            self.assertIsNotNone(self.query.index)
            rendezvous.wait()
            raise RuntimeError("injected model initialization failure")

        def search(*args, **kwargs):
            rendezvous.wait()
            searched.set()
            return real_fts(*args, **kwargs)

        self.constructor.side_effect = fail_model
        self.fts.side_effect = search
        for attempt in range(2):
            with (
                self.assertLogs(self.query.logger, level="WARNING") as logs,
                patch.object(self.query, "reciprocal_rank_fusion",
                             wraps=self.query.reciprocal_rank_fusion) as fusion,
            ):
                self.assert_fts_chunk(self.query.retrieve("oldneedle"))
            fusion.assert_called_once_with([[], [1]], limit=self.query.TOP_K)
            self.assertEqual(self.constructor.call_count, attempt + 1)
            self.assertIsNone(self.query.model)
            self.assertIn("injected model initialization failure", " ".join(logs.output))
        self.assertTrue(searched.is_set())
        self.rebuild.assert_not_called()

        # A later successful construction is published and reused normally.
        self.constructor.side_effect = None
        self.fts.side_effect = real_fts
        for _ in range(2):
            self.assertEqual(self.query.retrieve("oldneedle"),
                             [self.stored_chunks[1], self.stored_chunks[0]])
        self.assertEqual(self.constructor.call_count, 3)
        self.assertIs(self.query.model, self.model)
        self.rebuild.assert_not_called()

    def test_healthy_lazy_model_is_reused_and_vector_algorithm_is_unchanged(self):
        self.prepare_real_artifacts()
        original = "  oldneedle recovery instructions?  "
        for retrieve in (self.query.retrieve, self.query.retrieve_vector):
            result = retrieve(original)
            self.assertEqual(result, [self.stored_chunks[1], self.stored_chunks[0]])
            self.assertIs(result[0], self.query.chunks[1])
            self.assertIs(result[1], self.query.chunks[0])
            self.model.encode.assert_called_with([original])
        self.constructor.assert_called_once_with(self.query.EMBEDDING_MODEL)
        self.assertIs(self.query.model, self.model)
        self.assertEqual(self.query.chunks, self.stored_chunks)
        self.rebuild.assert_not_called()
        self.expand.assert_called_once_with(original)
        self.fts.assert_called_once()

    def test_real_vector_encode_and_search_failures_preserve_fts(self):
        for stage in ("encode", "search"):
            with self.subTest(stage=stage):
                self.prepare_real_artifacts()
                self.assertTrue(self.query._ensure_index_exists())
                target = self.model if stage == "encode" else self.query.index
                with (
                    patch.object(target, stage, side_effect=RuntimeError(f"injected {stage} failure")),
                    self.assertLogs(self.query.logger, level="WARNING") as logs,
                ):
                    self.assert_fts_chunk(self.query.retrieve("oldneedle"))
                self.assertIn(f"injected {stage} failure", " ".join(logs.output))

    def test_unrecoverable_chunks_never_map_fts_ids(self):
        self.prepare_real_artifacts()
        for contents in (None, b"corrupt SQLite database"):
            with self.subTest(contents=contents):
                self.query.chunks = []
                self.query.index = None
                if contents is None:
                    self.rag_path.unlink()
                else:
                    self.rag_path.write_bytes(contents)
                with self.assertLogs(self.query.logger, level="WARNING"):
                    self.assertEqual(self.query.retrieve("oldneedle"), [])
                self.assertEqual(self.query.chunks, [])
                self.assertFalse(self.query._retrieval_lock.locked())
        self.assertEqual(pickle.loads(self.chunks_path.read_bytes()), self.stored_chunks)
        self.assertEqual(self.rebuild.call_count, 2)
        self.fts.assert_not_called()
        self.constructor.assert_not_called()

    def test_successful_repair_reloads_generation_before_overlapping_searches(self):
        from rag import fts, store

        self.prepare_real_artifacts(corrupt_faiss=True)
        self.assertTrue(self.query._ensure_chunks_loaded())
        old_chunks = self.query.chunks
        new_chunks = [
            {"text": "oldneedle rebuilt instructions", "source": "rebuilt.txt", "chunk_id": 91},
            {"text": "new background", "source": "new.txt", "chunk_id": 42},
        ]
        expanded, repaired = Event(), Event()
        rendezvous = Barrier(2, timeout=5)
        vector_finished, fts_finished = Event(), Event()

        def expand(query):
            expanded.set()
            return [query]

        def rebuild():
            self.assertTrue(expanded.is_set())
            self.fts.assert_not_called()
            self.constructor.assert_not_called()
            index = faiss.IndexFlatIP(2)
            index.add(np.array([[0, 1], [1, 0]], dtype="float32"))
            faiss.write_index(index, str(self.index_path))
            store.build_rag_db(new_chunks, self.rag_path)
            repaired.set()

        def encode(inputs):
            self.assertEqual(inputs, ["oldneedle"])
            self.assertTrue(repaired.is_set())
            rendezvous.wait()
            vector_finished.set()
            return np.array([[0, 4]], dtype="float32")

        def search(inputs, limit):
            self.assertTrue(repaired.is_set())
            self.assertEqual(self.query.chunks, new_chunks)
            rendezvous.wait()
            fts_finished.set()
            return fts.search_fts(inputs, limit=limit, db_path=self.rag_path)

        self.expand.side_effect = expand
        self.rebuild.side_effect = rebuild
        self.model.encode.side_effect = encode
        self.fts.side_effect = search
        with self.assertLogs(self.query.logger, level="WARNING"):
            result = self.query.retrieve("oldneedle")
        self.rebuild.assert_called_once_with()
        self.assertTrue(vector_finished.is_set())
        self.assertTrue(fts_finished.is_set())
        self.assertEqual(result, new_chunks)
        self.assertIs(result[0], self.query.chunks[0])
        self.assertIsNot(self.query.chunks, old_chunks)
        np.testing.assert_allclose(self.query.index.reconstruct_n(0, 2), [[0, 1], [1, 0]])

    def test_missing_chunks_are_recovered_by_existing_rebuild(self):
        from rag import store

        self.prepare_real_artifacts()
        self.rag_path.unlink()
        def rebuild():
            store.build_rag_db(self.stored_chunks, self.rag_path)

        self.rebuild.side_effect = rebuild
        with self.assertLogs(self.query.logger, level="WARNING"):
            result = self.query.retrieve("oldneedle")
        self.rebuild.assert_called_once_with()
        self.assertEqual(self.query.chunks, self.stored_chunks)
        self.assertIs(result[0], self.query.chunks[1])
        self.assertEqual(result[0], self.stored_chunks[1])

    def test_post_rebuild_faiss_failure_uses_reloaded_chunks(self):
        from rag import store

        self.prepare_real_artifacts(corrupt_faiss=True)
        self.assertTrue(self.query._ensure_chunks_loaded())
        new_chunks = [self.stored_chunks[1], self.stored_chunks[0]]

        def rebuild():
            store.build_rag_db(new_chunks, self.rag_path)

        self.rebuild.side_effect = rebuild
        with self.assertLogs(self.query.logger, level="WARNING") as logs:
            result = self.query.retrieve("oldneedle")
        self.assertEqual(self.query.chunks, new_chunks)
        self.assertEqual(result, [new_chunks[0]])
        self.assertIs(result[0], self.query.chunks[0])
        self.assertIn("Vector index loading after rebuild failed", " ".join(logs.output))
        self.constructor.assert_not_called()

    def test_unreadable_rebuilt_chunks_do_not_use_cached_mapping(self):
        for contents in (None, b"unreadable new database"):
            with self.subTest(contents=contents):
                self.prepare_real_artifacts(corrupt_faiss=True)
                self.assertTrue(self.query._ensure_chunks_loaded())

                def rebuild():
                    if contents is None:
                        self.rag_path.unlink()
                    else:
                        self.rag_path.write_bytes(contents)

                self.rebuild.side_effect = rebuild
                with self.assertLogs(self.query.logger, level="WARNING") as logs:
                    self.assertEqual(self.query.retrieve("oldneedle"), [])
                self.assertEqual(self.query.chunks, [])
                self.assertIsNone(self.query.index)
                self.assertIn("Chunk loading failed", " ".join(logs.output))
        self.assertEqual(self.rebuild.call_count, 2)
        self.fts.assert_not_called()
        self.constructor.assert_not_called()

    def test_other_retrieval_cannot_rebuild_during_fts_search(self):
        self.prepare_real_artifacts(corrupt_faiss=True)
        fts_active, release_fts, second_expanded = Event(), Event(), Event()
        mapping_active, release_mapping = Event(), Event()
        rebuild_overlapped = Event()
        real_fts = self.fts.side_effect
        test = self

        class BlockingChunks(list):
            def __getitem__(self, position):
                mapping_active.set()
                try:
                    test.assertTrue(release_mapping.wait(timeout=5))
                    return super().__getitem__(position)
                finally:
                    mapping_active.clear()

        self.assertTrue(self.query._ensure_chunks_loaded())
        self.query.chunks = BlockingChunks(self.query.chunks)

        def expand(query):
            if query == "second":
                second_expanded.set()
            return ["oldneedle"]

        def search(*args, **kwargs):
            fts_active.set()
            try:
                self.assertTrue(release_fts.wait(timeout=5))
                return real_fts(*args, **kwargs)
            finally:
                fts_active.clear()

        def rebuild():
            if fts_active.is_set() or mapping_active.is_set():
                rebuild_overlapped.set()
            raise RuntimeError("injected rebuild failure")

        self.expand.side_effect = expand
        self.fts.side_effect = search
        self.rebuild.side_effect = rebuild
        with self.assertLogs(self.query.logger, level="WARNING"):
            with ThreadPoolExecutor(max_workers=2) as executor:
                first = executor.submit(self.query.retrieve, "first")
                try:
                    self.assertTrue(fts_active.wait(timeout=5))
                    second = executor.submit(self.query.retrieve, "second")
                    self.assertTrue(second_expanded.wait(timeout=5))
                    # The active retrieval holds artifact readiness through mapping.
                    acquired = self.query._retrieval_lock.acquire(blocking=False)
                    if acquired:
                        self.query._retrieval_lock.release()
                    self.assertFalse(acquired)
                    self.assertEqual(self.rebuild.call_count, 1)
                    release_fts.set()
                    self.assertTrue(mapping_active.wait(timeout=5))
                    self.assertFalse(fts_active.is_set())
                    acquired = self.query._retrieval_lock.acquire(blocking=False)
                    if acquired:
                        self.query._retrieval_lock.release()
                    self.assertFalse(acquired)
                    self.assertEqual(self.rebuild.call_count, 1)
                finally:
                    release_fts.set()
                    release_mapping.set()
                self.assert_fts_chunk(first.result(timeout=5))
                self.assert_fts_chunk(second.result(timeout=5))
        self.assertEqual(self.rebuild.call_count, 2)
        self.assertFalse(rebuild_overlapped.is_set())


if __name__ == "__main__":
    unittest.main()
