"""Regression checks for failures that used to produce misleading benchmark reports."""

import copy
from contextlib import redirect_stdout
from io import StringIO
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run import case_keys, parse_accuracy, parse_args, parse_throughput, print_tables, save_results, summarize_pairs
from build_config import build_benchmark, get_algo_list


def timing(direction, api, order=5, duration=1000, **extra):
    if direction == "convolve":
        return dict(name=f"{direction}/{api}/{order}", run_type="iteration", cpu_time=duration,
                    time_unit="ns", transforms_per_iteration=2, convolutions_per_iteration=1, **extra)
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
        for kind, count in (("cfft", 8), ("rfft", 11)):
            rows = [timing(direction, api) for direction, api in case_keys(kind, True, convolve_baselines=True)]
            parsed = parse_throughput({"benchmarks": rows}, kind, 5, 5, True)
            self.assertEqual(len(parsed), count)
            with self.assertRaises(ValueError):
                parse_throughput({"benchmarks": rows[:-1]}, kind, 5, 5, True)

    def test_convolution_units_and_separate_baselines(self):
        rows = [timing(direction, api) for direction, api in case_keys("rfft", True, convolve_baselines=True)]
        parsed = parse_throughput({"benchmarks": rows}, "rfft", 5, 5, True)
        conv = {row["api"]: row for row in parsed if row["direction"] == "convolve"}
        self.assertEqual(len(conv), 6)
        for api, operations in (("complex_aos", 896), ("complex_soa", 896), ("real", 832)):
            self.assertEqual(conv[api]["mflops"], operations)
            self.assertEqual(conv[api + "_separate"]["mflops"], operations)
        bad = copy.deepcopy(rows)
        row = next(row for row in bad if row["name"].startswith("convolve/"))
        row["convolutions_per_iteration"] = 2
        with self.assertRaisesRegex(ValueError, "one convolution"):
            parse_throughput({"benchmarks": bad}, "rfft", 5, 5, True)
        with self.assertRaisesRegex(ValueError, "Missing"):
            parse_throughput({"benchmarks": [row for row in rows if "_separate/" not in row["name"]]},
                             "rfft", 5, 5, True)


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

    def test_rfft_api_accuracy_requires_all_convolution_layouts(self):
        cases = case_keys("rfft", True) | {("roundtrip", "aos")}
        rows = [dict(library="zldsp", direction=direction, api=api, order=5, status="passed",
                     relative_rmse=1e-8, max_abs=1e-7) for direction, api in cases]
        output = "\n".join(json.dumps(row) for row in rows)
        self.assertEqual(len(parse_accuracy(output, "rfft", 5, True, 1e-5)), 9)
        missing = "\n".join(json.dumps(row) for row in rows if row["api"] != "complex_soa")
        with self.assertRaisesRegex(ValueError, "coverage mismatch"):
            parse_accuracy(missing, "rfft", 5, True, 1e-5)


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


class ReportingTests(unittest.TestCase):
    def summary(self, output):
        return json.loads(output.getvalue().split("JSON summary:\n", 1)[1])

    def test_default_accuracy_summary_retains_directions_libraries_and_precision(self):
        args = parse_args(["accuracy", "cfft", "5", "10", "--double"])
        rows = [dict(order=order, library=library, direction=direction, api="median", status="passed",
                     relative_rmse=(order + library_index + direction_index) * 1.234567890123456e-17,
                     max_abs=1.234567890123456e-15)
                for order in range(5, 11)
                for library_index, library in enumerate(("zldsp", "kfr"))
                for direction_index, direction in enumerate(("forward", "backward"))]
        output = StringIO()
        with redirect_stdout(output):
            print_tables(args, rows)
            save_results(args, args.kind, "accuracy", rows, threshold=args.threshold)
        summary = self.summary(output)
        self.assertEqual(summary["orders"], list(range(5, 11)))
        self.assertEqual(summary["precision"], "double")
        for metric in ("relative_rmse", "max_abs"):
            for library in ("zldsp", "kfr"):
                for direction in ("forward", "backward"):
                    expected = [row[metric] for row in rows if row["library"] == library and row["direction"] == direction]
                    self.assertEqual(summary["metrics"][metric][f"{direction}/median"][library], expected)

    def test_throughput_skips_are_null_and_detailed_json_is_preserved(self):
        rows = [dict(order=order, library=library, direction=direction, api="aos_aos", status="measured",
                     cpu_time_us=0.1234567890123456, mflops=1410 if order == 5 else 800.1234567890123, samples=3)
                for order in (5, 6) for library in ("zldsp", "pffft") for direction in ("forward", "backward")]
        rows[2] = dict(order=5, library="pffft", direction="forward", api="aos_aos", status="skipped",
                       reason="unsupported size: test")
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "results.json"
            args = parse_args(["throughput", "cfft", "5", "6", "--json", str(path)])
            output = StringIO()
            with redirect_stdout(output):
                print_tables(args, rows)
                save_results(args, args.kind, "throughput", rows)
            summary = self.summary(output)
            self.assertEqual(json.loads(path.read_text())["results"], rows)
        text = output.getvalue()
        self.assertIn("SKIP", text)
        self.assertNotIn("CPU time", text)
        self.assertIn("1410.0", text)
        self.assertIn("[1410.0, 800.1]", text)
        self.assertIn("[null, 800.1]", text)
        self.assertEqual(set(summary["metrics"]), {"mflops"})
        self.assertEqual(summary["metrics"]["mflops"]["forward/aos_aos"]["pffft"], [None, 800.1])
        self.assertEqual(summary["skipped"], [rows[2]])

    def test_commit_summary_omits_raw_runs_but_keeps_revisions_and_ratios(self):
        args = parse_args(["compare", "rfft", "5", "5"])
        row = dict(order=5, direction="backward", api="aos_real", current_us=1.5, previous_us=3.0,
                   throughput_ratio=2.0, change_percent=100.0, median_ratio=2.0, mad_percent=0.0)
        revisions = dict(current="new", previous="old")
        output = StringIO()
        with redirect_stdout(output):
            save_results(args, args.kind, "commit_comparison", [row], revisions=revisions, runs={})
        summary = self.summary(output)
        self.assertEqual(summary["revisions"], revisions)
        self.assertNotIn("runs", summary)
        self.assertEqual(summary["metrics"]["throughput_ratio"]["backward/aos_real"]["zldsp"], [2.0])

    def test_standard_library_set_excludes_fftw_and_full_set_adds_it(self):
        with patch("build_config.platform.system", return_value="Darwin"), \
             patch("build_config.platform.machine", return_value="arm64"), \
             patch.dict("build_config.os.environ", {}, clear=True):
            standard = get_algo_list(full=False)
            full = get_algo_list(full=True)
        self.assertEqual(set(standard), {"vdsp", "kfr", "pffft", "zldsp"})
        self.assertEqual(set(full) - set(standard), {"fftw3", "fftw3_estimate"})
        self.assertTrue(set(standard) <= set(full))


if __name__ == "__main__":
    unittest.main()
