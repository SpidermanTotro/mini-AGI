import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from minagi.config import load

from minagi.benchmark import activate_config, compare_reports, summarize_results


class BenchmarkTests(unittest.TestCase):
    def test_activate_config_sets_absolute_environment_path(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {
                "GREENLIGHT_CONFIG": "/old/config.yaml",
                "MINI_AGI_CONFIG": "/legacy/config.yaml"}):
            path = Path(tmp) / "config.yaml"
            path.write_text("model: {}\n", encoding="utf-8")
            activated = activate_config(path)
            self.assertEqual(activated, str(path.resolve()))
            self.assertEqual(os.environ["MINI_AGI_CONFIG"], activated)
            self.assertEqual(os.environ["GREENLIGHT_CONFIG"], activated)
            self.assertEqual(load(), {"model": {}})

    def test_environment_config_precedence_without_explicit_override(self):
        with patch.dict(os.environ, {
                "GREENLIGHT_CONFIG": "/greenlight.yaml",
                "MINI_AGI_CONFIG": "/legacy.yaml"}):
            self.assertEqual(activate_config(None), "/greenlight.yaml")

    def test_summary_counts_scored_cases(self):
        results = [
            {"name": "a", "exact_match": True},
            {"name": "b", "exact_match": False},
            {"name": "qualitative"},
        ]
        summary = summarize_results(results)
        self.assertEqual(summary["cases"], 3)
        self.assertEqual(summary["scored_cases"], 2)
        self.assertEqual(summary["exact_matches"], 1)
        self.assertEqual(summary["exact_match_accuracy"], 0.5)

    def test_compare_reports(self):
        base = {
            "provenance": {"commit": "old"},
            "summary": {"exact_match_accuracy": 0.5},
            "timing": {"elapsed_seconds": 10.0},
        }
        candidate = {
            "provenance": {"commit": "new"},
            "summary": {"exact_match_accuracy": 0.75},
            "timing": {"elapsed_seconds": 8.0},
        }
        comparison = compare_reports(base, candidate)
        self.assertEqual(comparison["base_commit"], "old")
        self.assertEqual(comparison["candidate_commit"], "new")
        self.assertEqual(comparison["exact_match_accuracy_delta"], 0.25)
        self.assertEqual(comparison["elapsed_seconds_delta"], -2.0)


if __name__ == "__main__":
    unittest.main()
