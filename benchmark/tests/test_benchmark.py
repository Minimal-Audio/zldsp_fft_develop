"""Regression checks for failures that used to produce misleading benchmark reports."""

import copy
import json
import math
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run import case_keys, parse_accuracy, parse_throughput, summarize_pairs
from build_config import build_benchmark


def timing(direction, api, order=5, duration=1000, **extra):
    return dict(name=f"{direction}/{api}/{order}", run_type="iteration", cpu_time=duration,
                time_unit="ns", transforms_per_iteration=1, **extra)


class ThroughputParsingTests(unittest.TestCase):
    def setUp(self):
        self.rows = [timing(direction, "aos_aos") for direction in ("forward", "backward")]

    def parse(self, rows=None):
        return parse_throughput({"benchmarks": self.rows if rows is None else rows}, "cfft", 5, 5)

    def test_directions_and_one_transform_units(self):
        rows = self.parse()
        self.assertEqual({row["direction"] for row in rows}, {"forward", "backward"})
        self.assertTrue(all(row["cpu_time_us"] == 1 for row in rows))
        self.assertTrue(all(row["mflops"] == 800 for row in rows))

    def test_repetitions_use_median_and_ignore_aggregate_percentages(self):
        rows = self.rows + [dict(row, cpu_time=3000) for row in self.rows]
        rows += [dict(self.rows[0], run_type="aggregate", aggregate_name="cv", cpu_time=0.1)]
        self.assertEqual(self.parse(rows)[0]["cpu_time_us"], 2)

    def test_missing_backward_is_an_error(self):
        with self.assertRaisesRegex(ValueError, "Missing"):
            self.parse(self.rows[:1])

    def test_invalid_times_and_iteration_counts_fail(self):
        for field, value in (("cpu_time", 0), ("cpu_time", -1), ("cpu_time", math.nan),
                             ("cpu_time", math.inf), ("time_unit", "minutes"),
                             ("transforms_per_iteration", 2)):
            with self.subTest(field=field, value=value):
                rows = copy.deepcopy(self.rows)
                rows[0][field] = value
                with self.assertRaises(ValueError):
                    self.parse(rows)

    def test_benchmark_execution_error_fails(self):
        rows = copy.deepcopy(self.rows)
        rows[0].update(error_occurred=True, error_message="non-finite output")
        with self.assertRaises(RuntimeError):
            self.parse(rows)

    def test_unsupported_size_is_explicit_and_not_zero_throughput(self):
        rows = copy.deepcopy(self.rows)
        rows[0].update(error_occurred=True, error_message="unsupported size: test")
        result = self.parse(rows)
        skipped = next(row for row in result if row["status"] == "skipped")
        self.assertNotIn("mflops", skipped)

    def test_entire_unsupported_range_stays_explicit(self):
        rows = [dict(row, error_occurred=True, error_message="unsupported size: test") for row in self.rows]
        results = self.parse(rows)
        self.assertEqual(len(results), 2)
        self.assertTrue(all(row["status"] == "skipped" and "mflops" not in row for row in results))

    def test_every_api_and_direction_is_required(self):
        for kind, count in (("cfft", 8), ("rfft", 5)):
            rows = [timing(direction, api) for direction, api in case_keys(kind, True)]
            parsed = parse_throughput({"benchmarks": rows}, kind, 5, 5, True)
            self.assertEqual(len(parsed), count)
            with self.assertRaises(ValueError):
                parse_throughput({"benchmarks": rows[:-1]}, kind, 5, 5, True)


class AccuracyParsingTests(unittest.TestCase):
    def setUp(self):
        self.rows = [dict(library=library, direction=direction, api="aos" if direction == "roundtrip" else "median",
                          order=5, status="passed", relative_rmse=1e-8, max_abs=1e-7)
                     for library in ("fftw3_estimate", "kfr", "zldsp")
                     for direction in ("forward", "backward", "roundtrip")]

    def parse(self, rows):
        return parse_accuracy("\n".join(json.dumps(row) for row in rows), "rfft", 5, False, 1e-5)

    def test_all_three_libraries_and_directions(self):
        self.assertEqual(len(self.parse(self.rows)), 9)

    def test_any_library_failure_is_an_error(self):
        for value in (math.nan, math.inf, 1e-2):
            rows = copy.deepcopy(self.rows)
            rows[0]["relative_rmse"] = value  # FFTW, not only zldsp, must pass.
            with self.assertRaises(ValueError):
                self.parse(rows)

    def test_missing_duplicate_and_excluded_library(self):
        for rows in (self.rows[:-1], self.rows + self.rows[:1],
                     [dict(row, library="pffft") if row["library"] == "kfr" else row for row in self.rows]):
            with self.assertRaises(ValueError):
                self.parse(rows)


class ComparisonTests(unittest.TestCase):
    def rows(self, duration):
        return [dict(order=5, direction=direction, api="aos_aos", status="measured", cpu_time_us=duration)
                for direction in ("forward", "backward")]

    def test_speedup_direction_and_separate_directions(self):
        rows = summarize_pairs([self.rows(2)], [self.rows(4)])
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(row["throughput_ratio"] == 2 and row["change_percent"] == 100 for row in rows))

    def test_case_mismatch_fails(self):
        with self.assertRaises(ValueError):
            summarize_pairs([self.rows(2)], [self.rows(4)[:1]])

    def test_build_validates_before_running_tools(self):
        with patch("build_config.run_command") as command:
            for algorithm, target in (("typo", "throughput_cfft"), ("zldsp", "typo")):
                with self.assertRaises(ValueError):
                    build_benchmark(algorithm, target)
            command.assert_not_called()


if __name__ == "__main__":
    unittest.main()
