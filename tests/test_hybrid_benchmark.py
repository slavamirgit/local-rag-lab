"""Check evaluation integrity and observation without models or network."""

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


with patch.object(sys, "path", [str(Path(__file__).resolve().parents[1] / "bench"), *sys.path]):
    spec = importlib.util.spec_from_file_location(
        "hybrid_benchmark_under_test",
        Path(__file__).resolve().parents[1] / "bench/hybrid_benchmark.py",
    )
    benchmark = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(benchmark)


class HybridBenchmarkTests(unittest.TestCase):
    def test_observation_delegates_once_and_preserves_output_identity(self):
        inputs = ["original", "term"]
        contexts = [{"source": "document"}]
        query = SimpleNamespace(expand_query=Mock(return_value=inputs))
        real_expand = query.expand_query

        def retrieve(text):
            self.assertIs(query.expand_query(text), inputs)
            return contexts

        query.retrieve = retrieve
        result, latency, observation = benchmark.measure_query(query, "original")
        real_expand.assert_called_once_with("original")
        self.assertIs(query.expand_query, real_expand)
        self.assertIs(result, contexts)
        self.assertIs(observation["expansion_inputs"], inputs)
        self.assertFalse(observation["expansion_raised_unexpectedly"])
        self.assertGreaterEqual(latency, 0)

    def test_unexpected_expansion_error_reaches_production_fallback(self):
        error = RuntimeError("unexpected")
        query = SimpleNamespace(expand_query=Mock(side_effect=error))
        fallback_exercised = []

        def retrieve(text):
            try:
                query.expand_query(text)
            except RuntimeError as raised:
                self.assertIs(raised, error)
                fallback_exercised.append(text)
            return []

        query.retrieve = retrieve
        _, _, observation = benchmark.measure_query(query, "original")
        query.expand_query.assert_called_once_with("original")
        self.assertEqual(fallback_exercised, ["original"])
        self.assertEqual(observation["expansion_inputs"], ["original"])
        self.assertTrue(observation["expansion_raised_unexpectedly"])
        self.assertEqual(observation["expansion_exception_type"], "RuntimeError")

    def test_missing_or_duplicate_expansion_calls_abort(self):
        for count in (0, 2):
            query = SimpleNamespace(expand_query=Mock(return_value=["original"]))

            def retrieve(text):
                for _ in range(count):
                    query.expand_query(text)
                return []

            query.retrieve = retrieve
            with self.assertRaises(ValueError):
                benchmark.measure_query(query, "original")

    def test_comparison_rejects_changed_identity_or_labels(self):
        original = {"id": "q1", "category": "semantic", "query": "text", "expected_source": "a"}
        for key in original:
            changed = {**original, key: "changed"}
            with self.subTest(key=key), self.assertRaises(ValueError):
                benchmark.verify_query_identity([original], [changed])
        with self.assertRaises(ValueError):
            benchmark.verify_query_identity([original, original], [original, original])

    def test_missing_ranks_and_improvement_classification(self):
        pairs = [(None, 5), (1, None), (5, 1), (1, 2), (None, None), (2, 2)]
        old_rows, new_rows = [], []
        for i, (old, new) in enumerate(pairs):
            identity = dict(id=str(i), category="semantic", query=str(i), expected_source="a")
            for rows, rank in ((old_rows, old), (new_rows, new)):
                rows.append({**identity, "expected_source_rank": rank,
                             "hit_at_5": int(rank is not None),
                             "reciprocal_rank": 1 / rank if rank is not None else 0,
                             "latency_ms": 10})
        old_metrics = benchmark.aggregate(old_rows)
        new_metrics = benchmark.aggregate(new_rows)
        result = benchmark.compare_to_vector(
            {"queries": old_rows, "aggregate": {"all": old_metrics, "by_category": {"semantic": old_metrics}}},
            new_rows, {"all": new_metrics, "by_category": {"semantic": new_metrics}},
        )
        self.assertEqual([row["rank_outcome"] for row in result["queries"]],
                         ["improved", "worsened", "improved", "worsened", "unchanged", "unchanged"])
        self.assertEqual([row["id"] for row in result["queries"] if row["recovered_miss"]], ["0"])
        self.assertEqual([row["id"] for row in result["queries"] if row["new_miss"]], ["1"])

    def test_input_hash_mismatch_aborts(self):
        with patch.object(benchmark, "fingerprint", return_value={"corpus": "changed"}):
            with self.assertRaisesRegex(ValueError, "Corpus/query hashes"):
                benchmark.verify_inputs("sanity", {"metadata": {"input_sha256": {"corpus": "frozen"}}})


if __name__ == "__main__":
    unittest.main()
