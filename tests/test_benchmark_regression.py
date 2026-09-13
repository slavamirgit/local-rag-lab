"""Offline storage regression contract checks using small in-memory results."""

from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
from io import StringIO
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from bench import storage_regression as regression
from bench import vector_baseline
from bench.vector_baseline import validate_output_path

with patch.dict(sys.modules, {"vector_baseline": vector_baseline}):
    from bench import hybrid_benchmark


class BenchmarkOutputGuardTests(unittest.TestCase):
    def setUp(self):
        # A guard regression must never reach model loading or network work.
        self.enterContext(patch.dict(sys.modules, {
            name: None for name in ("config", "numpy", "torch", "rag", "requests")
        }))

    def assert_runner_rejects(self, run, output, message):
        mode, dataset = run.split("-")
        runner = vector_baseline if mode == "vector" else hybrid_benchmark
        output_args = [] if output is None else ["--output", str(output)]
        with patch.object(sys, "argv", [runner.__file__, "--dataset", dataset, *output_args]), \
                patch.object(Path, "read_text", side_effect=AssertionError("Input read")) as read:
            with self.assertRaises(ValueError) as raised:
                runner.main()
            self.assertIn(message, str(raised.exception))
            read.assert_not_called()

    def test_guard_rejects_all_four_historical_destinations(self):
        expected = tuple(regression.ROOT / "bench/results" / name
                         for name in regression.REFERENCES.values())
        self.assertEqual(set(vector_baseline.PROTECTED_RESULT_PATHS), set(expected))
        self.assertEqual(len(vector_baseline.PROTECTED_RESULT_PATHS), 4)
        self.assertIs(hybrid_benchmark.validate_output_path, validate_output_path)
        for historical in expected:
            with self.subTest(historical=historical):
                with self.assertRaisesRegex(ValueError, "Refusing historical result destination"):
                    validate_output_path(historical)

    def test_guard_requires_explicit_output(self):
        with self.assertRaisesRegex(ValueError, "use --output"):
            validate_output_path(None)
        for run in regression.REFERENCES:
            with self.subTest(run=run):
                self.assert_runner_rejects(run, None, "use --output")

    def test_all_runner_datasets_reject_all_historical_destinations_and_dotdot_aliases(self):
        # Includes vector sanity -> vector challenge and both hybrid references,
        # and both hybrid datasets -> both vector references.
        for run in regression.REFERENCES:
            for name in regression.REFERENCES.values():
                historical = regression.ROOT / "bench/results" / name
                for output in (historical, Path("bench/results") / name,
                               regression.ROOT / "bench/results/../results" / name):
                    with self.subTest(run=run, output=output):
                        self.assert_runner_rejects(
                            run, output, f"Refusing historical result destination {historical}"
                        )

    def test_guard_rejects_dotdot_aliases(self):
        for historical in vector_baseline.PROTECTED_RESULT_PATHS:
            alias = historical.parent / ".." / "results" / historical.name
            with self.subTest(alias=alias):
                with self.assertRaisesRegex(ValueError, "Refusing historical result destination"):
                    validate_output_path(alias)

    def test_guard_and_runners_reject_symlink_aliases(self):
        with TemporaryDirectory() as directory:
            for historical in vector_baseline.PROTECTED_RESULT_PATHS:
                alias = Path(directory) / historical.name
                alias.symlink_to(historical)
                with self.subTest(historical=historical):
                    with self.assertRaisesRegex(ValueError, "Refusing historical result destination"):
                        validate_output_path(alias)
                    for run in regression.REFERENCES:
                        with self.subTest(run=run):
                            self.assert_runner_rejects(run, alias, "Refusing historical result destination")

    def test_guard_and_runners_reject_absent_historical_files_and_dangling_symlinks(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            historical_paths = tuple(root / path.relative_to(vector_baseline.ROOT)
                                     for path in vector_baseline.PROTECTED_RESULT_PATHS)
            with patch.object(vector_baseline, "PROTECTED_RESULT_PATHS", historical_paths):
                for historical in historical_paths:
                    with self.subTest(historical=historical):
                        self.assertFalse(historical.exists())
                        alias = root / f"alias-{historical.name}"
                        alias.symlink_to(historical)
                        for output in (historical, alias):
                            with self.assertRaisesRegex(ValueError, "Refusing historical result destination"):
                                validate_output_path(output)
                            for run in regression.REFERENCES:
                                with self.subTest(run=run, output=output):
                                    self.assert_runner_rejects(
                                        run, output, "Refusing historical result destination"
                                    )
                        self.assertFalse(historical.exists())

    def test_guard_leaves_candidate_existence_and_contents_unchanged(self):
        with TemporaryDirectory() as directory:
            candidate = Path(directory) / "candidate.json"
            self.assertIs(validate_output_path(candidate), candidate)
            self.assertFalse(candidate.exists())
            candidate.write_text("existing candidate", encoding="utf-8")
            self.assertIs(validate_output_path(candidate), candidate)
            self.assertEqual(candidate.read_text(encoding="utf-8"), "existing candidate")
        relative_candidate = Path("bench/results/../results/candidate.json")
        self.assertIs(validate_output_path(relative_candidate), relative_candidate)

    def test_vector_allows_existing_candidate_before_input_loading(self):
        with TemporaryDirectory() as directory:
            candidate = Path(directory) / "candidate.json"
            candidate.write_text("existing candidate", encoding="utf-8")
            for dataset in vector_baseline.DATASETS:
                with self.subTest(dataset=dataset), \
                        patch.object(sys, "argv", [vector_baseline.__file__, "--dataset", dataset,
                                                   "--output", str(candidate)]), \
                        patch.object(Path, "read_text", side_effect=RuntimeError("Input loading")) as read:
                    with self.assertRaisesRegex(RuntimeError, "Input loading"):
                        vector_baseline.main()
                    read.assert_called_once_with(encoding="utf-8")
            self.assertEqual(candidate.read_text(encoding="utf-8"), "existing candidate")

    def test_hybrid_refuses_existing_candidate_before_input_loading(self):
        with TemporaryDirectory() as directory:
            candidate = Path(directory) / "candidate.json"
            candidate.write_text("existing candidate", encoding="utf-8")
            for dataset in vector_baseline.DATASETS:
                with self.subTest(dataset=dataset), \
                        patch.object(sys, "argv", [hybrid_benchmark.__file__, "--dataset", dataset,
                                                   "--output", str(candidate)]), \
                        patch.object(Path, "read_text", side_effect=AssertionError("Input read")) as read:
                    with self.assertRaisesRegex(FileExistsError, "Refusing to overwrite"):
                        hybrid_benchmark.main()
                    read.assert_not_called()
            self.assertEqual(candidate.read_text(encoding="utf-8"), "existing candidate")


def result():
    metrics = {"hit_at_5": 1, "mrr_at_5": 1.0, "latency_ms": {"mean": 10}}
    return {
        "metadata": {"measured_at_utc": "before", "git_revision": "old"},
        "queries": [
            {"id": query_id, "ranked_sources": ["a", "b"], "expected_source_rank": 1,
             "hit_at_5": 1, "reciprocal_rank": 1.0, "latency_ms": 10,
             "expansion_inputs": ["original", "term"]}
            for query_id in ("q1", "q2")
        ],
        "aggregate": {"all": deepcopy(metrics),
                      "by_category": {"semantic": deepcopy(metrics)}},
    }


class BenchmarkRegressionTests(unittest.TestCase):
    def setUp(self):
        self.historical = result()
        self.candidate = deepcopy(self.historical)

    def compare(self, hybrid=False):
        regression.compare_run(self.historical, self.candidate,
                               run="hybrid-sanity" if hybrid else "vector-sanity",
                               hybrid=hybrid)

    def test_ignores_latency_and_measurement_metadata_without_mutation(self):
        self.candidate["metadata"] = {
            "measured_at_utc": "after", "git_revision": "new",
            "implementation_sha256": {"production": "new"},
            "benchmark_runner_sha256": {"runner": "new"},
            "packages": {"torch": "different"}, "platform": "different",
        }
        for row in self.candidate["queries"]:
            row["latency_ms"] = 999
        for metrics in (self.candidate["aggregate"]["all"],
                        self.candidate["aggregate"]["by_category"]["semantic"]):
            metrics["latency_ms"] = {"mean": 999}
        before = deepcopy((self.historical, self.candidate))
        self.compare()
        self.compare(hybrid=True)
        self.assertEqual((self.historical, self.candidate), before)

    def test_changed_query_fields_fail_with_useful_values(self):
        for field, changed in (("ranked_sources", ["b", "a"]),
                               ("expected_source_rank", None), ("hit_at_5", 0),
                               ("reciprocal_rank", 0.5)):
            with self.subTest(field=field):
                self.candidate = deepcopy(self.historical)
                self.candidate["queries"][0][field] = changed
                with self.assertRaises(regression.RegressionMismatch) as raised:
                    self.compare()
                message = str(raised.exception)
                for expected in ("vector-sanity", "query='q1'", f"field={field}",
                                 f"historical={self.historical['queries'][0][field]!r}",
                                 f"candidate={changed!r}"):
                    self.assertIn(expected, message)

    def test_changed_overall_metrics_fail(self):
        for field in ("hit_at_5", "mrr_at_5"):
            with self.subTest(field=field):
                self.candidate = deepcopy(self.historical)
                self.candidate["aggregate"]["all"][field] = 0.5
                with self.assertRaisesRegex(regression.RegressionMismatch,
                                            f"aggregate.all.{field}"):
                    self.compare()

    def test_changed_category_metrics_fail(self):
        for field in ("hit_at_5", "mrr_at_5"):
            with self.subTest(field=field):
                self.candidate = deepcopy(self.historical)
                self.candidate["aggregate"]["by_category"]["semantic"][field] = 0.5
                with self.assertRaisesRegex(regression.RegressionMismatch,
                                            f"aggregate.by_category.semantic.{field}"):
                    self.compare()

    def test_changed_category_set_fails(self):
        categories = self.candidate["aggregate"]["by_category"]
        categories["mixed"] = categories.pop("semantic")
        with self.assertRaisesRegex(regression.RegressionMismatch, "by_category.categories"):
            self.compare()

    def test_changed_ordered_query_ids_fail(self):
        for change in ("reorder", "rename", "remove", "duplicate"):
            with self.subTest(change=change):
                self.candidate = deepcopy(self.historical)
                rows = self.candidate["queries"]
                if change == "reorder":
                    rows.reverse()
                elif change == "rename":
                    rows[0]["id"] = "changed"
                elif change == "remove":
                    rows.pop()
                else:
                    rows[1]["id"] = rows[0]["id"]
                with self.assertRaisesRegex(regression.RegressionMismatch, "ordered_query_ids"):
                    self.compare()

    def test_hybrid_requires_exact_expansion_inputs(self):
        self.candidate["queries"][0]["expansion_inputs"].reverse()
        with self.assertRaisesRegex(regression.RegressionMismatch,
                                    "hybrid-sanity query='q1' field=expansion_inputs"):
            self.compare(hybrid=True)

    def test_vector_does_not_require_expansion_inputs(self):
        for row in self.historical["queries"] + self.candidate["queries"]:
            del row["expansion_inputs"]
        self.compare()
        self.candidate["queries"][0]["expansion_inputs"] = ["ignored"]
        self.compare()

    def test_missing_required_fields_fail_even_when_both_missing(self):
        for field in (*regression.QUERY_FIELDS, "expansion_inputs", "id"):
            with self.subTest(field=field):
                self.historical, self.candidate = result(), result()
                del self.historical["queries"][0][field]
                del self.candidate["queries"][0][field]
                with self.assertRaisesRegex(regression.RegressionMismatch, "<missing>"):
                    self.compare(hybrid=True)

    def test_cli_all_four_pass_and_mismatch_returns_nonzero(self):
        args = [item for run in regression.REFERENCES
                for item in (f"--{run}", f"/candidate/{run}.json")]
        stdout, stderr = StringIO(), StringIO()
        with patch.object(Path, "read_text", return_value=json.dumps(self.historical)), \
                redirect_stdout(stdout), redirect_stderr(stderr):
            self.assertEqual(regression.main(args), 0)
        for run in regression.REFERENCES:
            self.assertIn(f"PASS: {run}", stdout.getvalue())
        self.assertIn("All four", stdout.getvalue())
        self.candidate["queries"][0]["expected_source_rank"] = 2
        with patch.object(Path, "read_text", side_effect=[json.dumps(self.historical),
                                                         json.dumps(self.candidate)]), \
                redirect_stdout(StringIO()), redirect_stderr(stderr):
            self.assertEqual(regression.main(args), 1)
        self.assertIn("vector-sanity query='q1' field=expected_source_rank", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
