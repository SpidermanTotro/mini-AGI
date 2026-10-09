"""Streaming A/B telemetry is measured, reset per interval, never invented."""
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import torch

from train import _balance_capacity_sample, _balance_expert_use


class BalanceStreamTelemetryTests(unittest.TestCase):
    def test_capacity_counts_from_real_pool_hook(self):
        model = SimpleNamespace(pool_dropped=Mock(return_value=(0.125, 3, 24)))
        self.assertEqual(_balance_capacity_sample(model),
                         {"pool_dropped": 3, "pool_requested": 24})
        model.pool_dropped.assert_called_once_with()

    def test_missing_capacity_hook_is_absent_not_zero(self):
        self.assertEqual(_balance_capacity_sample(SimpleNamespace()), {})

    def test_expert_use_snapshot_copies_tensor_without_mutation(self):
        original = torch.tensor([7.0, 0.0, 2.0])
        model = SimpleNamespace(pool=SimpleNamespace(use=original))
        self.assertEqual(_balance_expert_use(model), {"use": [7.0, 0.0, 2.0]})
        self.assertTrue(torch.equal(original, torch.tensor([7.0, 0.0, 2.0])))

    def test_missing_usage_does_not_fabricate_experts(self):
        self.assertEqual(_balance_expert_use(SimpleNamespace(pool=object())), {})


if __name__ == "__main__":
    unittest.main()
