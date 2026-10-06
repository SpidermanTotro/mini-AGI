import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.build_balance_ab_report import build_report


class BuildBalanceABReportTests(unittest.TestCase):
    def test_builds_contract_report_from_doctor_evidence(self):
        history = [{"heldout_loss": 0.75, "pool_dropped": 2,
                    "pool_requested": 100}]
        experts = [{"use": [4, 0, 2, 1], "uid": [1, 2, 3, 4]}]
        reads = [history, experts, experts]
        with patch("tools.build_balance_ab_report.read_history",
                   side_effect=reads), \
             patch("tools.build_balance_ab_report.diagnose_experts",
                   return_value={"health": "healthy",
                                 "observations": {"experts": 4,
                                                  "total_use": 7.0}}), \
             patch("tools.build_balance_ab_report.resume_preflight",
                   return_value={"health": "healthy"}):
            out = build_report("history", "experts", "weights")
        self.assertEqual(out["heldout_loss"], 0.75)
        self.assertEqual(out["expert_utilization"], 0.75)
        self.assertEqual(out["capacity_drop"], 0.02)
        self.assertEqual(out["n_experts"], 4)
        self.assertTrue(out["resume_ok"])

    def test_missing_heldout_evidence_is_rejected(self):
        with patch("tools.build_balance_ab_report.read_history",
                   return_value=[{"loss": 1.0}]):
            with self.assertRaisesRegex(ValueError, "heldout"):
                build_report("history", "experts", "weights")


if __name__ == "__main__":
    unittest.main()
