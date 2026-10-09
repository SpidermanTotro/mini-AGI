"""Pure-Python tests for report-only expert balance A/B promotion checks.

Synthetic fixture numbers are schema checks, never evidence of GPU improvement.
"""
import unittest
from tools.compare_balance_ab import compare, validate_result


def sample(**changes):
    values = {
        "heldout_loss": 0.7,
        "expert_utilization": 0.5,
        "capacity_drop": 0.01,
        "n_experts": 16,
        "resume_ok": True,
        "balance_strength": 0.0,
        "initial_checkpoint_sha256": "a" * 64,
        "training_corpus_sha256": "b" * 64,
        "heldout_corpus_sha256": "c" * 64,
        "seed": 42,
        "training_steps": 100,
    }
    values.update(changes)
    return values


class BalanceReportGateTests(unittest.TestCase):
    def test_synthetic_valid_pair_only_clears_schema_gate(self):
        ok, _ = compare(sample(), sample(balance_strength=0.001,
                                       heldout_loss=0.69,
                                       expert_utilization=0.6))
        self.assertTrue(ok)

    def test_report_provenance_must_match(self):
        baseline = sample()
        for field, value in (
            ("seed", 43),
            ("training_steps", 200),
            ("initial_checkpoint_sha256", "d" * 64),
            ("training_corpus_sha256", "e" * 64),
            ("heldout_corpus_sha256", "f" * 64),
        ):
            with self.subTest(field=field):
                ok, reason = compare(
                    baseline,
                    sample(balance_strength=0.001,
                           heldout_loss=0.69,
                           expert_utilization=0.6, **{field: value}),
                )
                self.assertFalse(ok)
                self.assertIn(field, reason)

    def test_resume_is_hard_gate(self):
        ok, reason = compare(sample(), sample(balance_strength=0.001,
                                             heldout_loss=0.5,
                                             expert_utilization=0.8,
                                             resume_ok=False))
        self.assertFalse(ok)
        self.assertIn("restart/resume", reason)

    def test_loss_capacity_and_growth_regressions_block(self):
        baseline = sample()
        good = dict(balance_strength=0.001, heldout_loss=0.69,
                    expert_utilization=0.6)
        for change in ({"heldout_loss": 0.71},
                       {"capacity_drop": 0.02},
                       {"n_experts": 17}):
            with self.subTest(change=change):
                ok, reason = compare(baseline, sample(**(good | change)))
                self.assertFalse(ok)
                self.assertIn("candidate does not clear", reason)

    def test_invalid_and_incomplete_reports_fail_closed(self):
        bad_reports = (
            {"heldout_loss": float("nan")},
            {"balance_strength": float("inf")},
            {"initial_checkpoint_sha256": "bad-hash"},
            {"resume_ok": "true"},
            {"expert_utilization": 1.5},
        )
        for change in bad_reports:
            with self.subTest(change=change):
                candidate_values = {
                    "balance_strength": 0.001,
                    "heldout_loss": 0.69,
                    "expert_utilization": 0.6,
                    **change,
                }
                ok, _ = compare(sample(), sample(**candidate_values))
                self.assertFalse(ok)
        incomplete = sample()
        incomplete.pop("heldout_corpus_sha256")
        with self.assertRaisesRegex(ValueError, "heldout_corpus_sha256"):
            validate_result(incomplete)


if __name__ == "__main__":
    unittest.main()
