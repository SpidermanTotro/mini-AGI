"""Contract tests for upstream router-balance experiments.

These tests deliberately do not enable upstream behavior in stable training.
They define the measurements an A/B candidate must report before promotion.
"""

import unittest


REQUIRED_METRICS = (
    "heldout_loss",
    "expert_utilization",
    "capacity_drop",
    "n_experts",
    "resume_ok",
)


def validate_balance_ab_result(result):
    missing = [name for name in REQUIRED_METRICS if name not in result]
    if missing:
        raise ValueError("missing A/B metrics: " + ", ".join(missing))
    if result["heldout_loss"] < 0:
        raise ValueError("heldout_loss must be nonnegative")
    if not 0 <= result["expert_utilization"] <= 1:
        raise ValueError("expert_utilization must be in [0, 1]")
    if not 0 <= result["capacity_drop"] <= 1:
        raise ValueError("capacity_drop must be in [0, 1]")
    if int(result["n_experts"]) <= 0:
        raise ValueError("n_experts must be positive")
    if result["resume_ok"] is not True:
        raise ValueError("candidate must pass restart/resume before promotion")
    return True


class UpstreamBalanceABContractTests(unittest.TestCase):
    def test_complete_candidate_can_be_compared(self):
        self.assertTrue(validate_balance_ab_result({
            "heldout_loss": 0.65,
            "expert_utilization": 0.8,
            "capacity_drop": 0.01,
            "n_experts": 64,
            "resume_ok": True,
        }))

    def test_resume_failure_blocks_promotion(self):
        with self.assertRaisesRegex(ValueError, "restart/resume"):
            validate_balance_ab_result({
                "heldout_loss": 0.64,
                "expert_utilization": 0.9,
                "capacity_drop": 0.0,
                "n_experts": 80,
                "resume_ok": False,
            })

    def test_missing_behavior_metric_blocks_promotion(self):
        with self.assertRaisesRegex(ValueError, "expert_utilization"):
            validate_balance_ab_result({
                "heldout_loss": 0.64,
                "capacity_drop": 0.0,
                "n_experts": 80,
                "resume_ok": True,
            })


if __name__ == "__main__":
    unittest.main()
