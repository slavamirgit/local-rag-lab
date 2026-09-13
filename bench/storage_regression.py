"""Compare fresh storage-migration measurements with four frozen retrieval results."""

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
REFERENCES = {
    "vector-sanity": "vector-baseline.json",
    "vector-challenge": "vector-challenge-baseline.json",
    "hybrid-sanity": "hybrid-sanity.json",
    "hybrid-challenge": "hybrid-challenge.json",
}
QUERY_FIELDS = ("ranked_sources", "expected_source_rank", "hit_at_5", "reciprocal_rank")
METRIC_FIELDS = ("hit_at_5", "mrr_at_5")
MISSING = object()


class RegressionMismatch(ValueError):
    """The first differing or missing required retrieval field."""


def compare_run(historical, candidate, *, run, hybrid=False):
    """Require the retrieval contract, ignoring latency and measurement metadata."""
    def check(field, old, new, query_id=None):
        if old is MISSING or new is MISSING or old != new:
            location = run if query_id is None else f"{run} query={query_id!r}"
            old_value = "<missing>" if old is MISSING else repr(old)
            new_value = "<missing>" if new is MISSING else repr(new)
            raise RegressionMismatch(
                f"{location} field={field}: historical={old_value}; candidate={new_value}"
            )

    old_rows, new_rows = historical["queries"], candidate["queries"]
    old_ids = [row.get("id", MISSING) for row in old_rows]
    new_ids = [row.get("id", MISSING) for row in new_rows]
    check("ordered_query_ids", old_ids, new_ids)
    fields = QUERY_FIELDS + (("expansion_inputs",) if hybrid else ())
    for old, new in zip(old_rows, new_rows):
        check("id", old.get("id", MISSING), new.get("id", MISSING))
        for field in fields:
            check(field, old.get(field, MISSING), new.get(field, MISSING), old["id"])

    old_metrics, new_metrics = historical["aggregate"], candidate["aggregate"]
    for field in METRIC_FIELDS:
        check(f"aggregate.all.{field}", old_metrics["all"].get(field, MISSING),
              new_metrics["all"].get(field, MISSING))
    old_categories = old_metrics["by_category"]
    new_categories = new_metrics["by_category"]
    check("aggregate.by_category.categories", sorted(old_categories), sorted(new_categories))
    for category in sorted(old_categories):
        for field in METRIC_FIELDS:
            check(f"aggregate.by_category.{category}.{field}",
                  old_categories[category].get(field, MISSING),
                  new_categories[category].get(field, MISSING))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for run in REFERENCES:
        parser.add_argument(f"--{run}", type=Path, required=True)
    args = parser.parse_args(argv)
    for run, reference_name in REFERENCES.items():
        reference = ROOT / "bench/results" / reference_name
        candidate_path = getattr(args, run.replace("-", "_"))
        try:
            historical = json.loads(reference.read_text(encoding="utf-8"))
            candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
            compare_run(historical, candidate, run=run, hybrid=run.startswith("hybrid-"))
        except RegressionMismatch as exc:
            print(f"FAIL: {exc}", file=sys.stderr)
            return 1
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
            print(f"FAIL: {run}: cannot compare {reference} with {candidate_path}: {exc}",
                  file=sys.stderr)
            return 1
        print(f"PASS: {run}")
    print("All four storage regression comparisons passed (latency ignored).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
