"""Offline storage regression contract checks using small in-memory results."""

from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
from io import StringIO
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from bench import storage_regression as regression
from bench.vector_baseline import validate_output_path


class BenchmarkOutputGuardTests(unittest.TestCase):
    def test_runners_reject_default_and_explicit_historical_destinations(self):
        for run, name in regression.REFERENCES.items():
            mode, dataset = run.split("-")
            runner = "vector_baseline.py" if mode == "vector" else "hybrid_benchmark.py"
            historical = regression.ROOT / "bench/results" / name
            for output_args in ([], ["--output", str(historical)],
                                ["--output", f"bench/results/../results/{name}"]):
                with self.subTest(run=run, output_args=output_args):
                    process = subprocess.run(
                        [sys.executable, str(regression.ROOT / "bench" / runner),
                         "--dataset", dataset, *output_args],
                        cwd=regression.ROOT, capture_output=True, text=True, timeout=5,
                    )
                    self.assertNotEqual(process.returncode, 0)
                    self.assertIn(f"Refusing historical result destination {historical}",
                                  process.stderr)
                    self.assertIn("use --output with a different path", process.stderr)

    def test_guard_rejects_absent_historical_files_and_symlink_aliases(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            for name in regression.REFERENCES.values():
                with self.subTest(name=name):
                    historical = root / name
                    self.assertFalse(historical.exists())
                    alias = root / f"alias-{name}"
                    alias.symlink_to(historical)
                    for output in (None, historical, alias):
                        with self.assertRaisesRegex(ValueError, "use --output"):
                            validate_output_path(output, historical)
                    self.assertFalse(historical.exists())

    def test_guard_leaves_candidate_existence_and_contents_unchanged(self):
        with TemporaryDirectory() as directory:
            historical = Path(directory) / "historical.json"
            candidate = Path(directory) / "candidate.json"
            self.assertIs(validate_output_path(candidate, historical), candidate)
            self.assertFalse(candidate.exists())
            candidate.write_text("existing candidate", encoding="utf-8")
            self.assertIs(validate_output_path(candidate, historical), candidate)
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
