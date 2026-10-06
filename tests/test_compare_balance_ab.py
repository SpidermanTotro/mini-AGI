import json
import tempfile
import unittest
from pathlib import Path

from tools.compare_balance_ab import compare, load_report


class BalanceABToolTests(unittest.TestCase):
    def report(self, **changes):
        data = {
            "heldout_loss": 1.0,
            "expert_utilization": 0.5,
            "capacity_drop": 0.1,
            "n_experts": 64,
            "resume_ok": True,
        }
        data.update(changes)
        return data

    def test_compare_reports_signed_deltas(self):
        out = compare(
            self.report(),
            self.report(heldout_loss=0.9, expert_utilization=0.7,
                        capacity_drop=0.05, n_experts=72),
        )
        self.assertAlmostEqual(out["heldout_loss_delta"], -0.1)
        self.assertAlmostEqual(out["expert_utilization_delta"], 0.2)
        self.assertAlmostEqual(out["capacity_drop_delta"], -0.05)
        self.assertEqual(out["expert_count_delta"], 8)

    def test_loader_rejects_failed_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "candidate.json"
            path.write_text(json.dumps(self.report(resume_ok=False)))
            with self.assertRaisesRegex(ValueError, "restart/resume"):
                load_report(path)

    def test_loader_rejects_incomplete_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "candidate.json"
            path.write_text(json.dumps({"heldout_loss": 1.0}))
            with self.assertRaisesRegex(ValueError, "expert_utilization"):
                load_report(path)


if __name__ == "__main__":
    unittest.main()
