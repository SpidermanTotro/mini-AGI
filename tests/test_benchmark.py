import unittest

from minagi.benchmark import compare_reports, summarize_results


class BenchmarkTests(unittest.TestCase):
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
