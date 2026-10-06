"""Contract tests for upstream router-balance experiments.

These tests deliberately do not enable upstream behavior in stable training.
They define the measurements an A/B candidate must report before promotion.
"""

import unittest
import tempfile

import torch

from minagi.recur import RecurConfig
from minagi.recur import RecurCoder
from train import build_paged
from minagi import store
from tools.compare_balance_ab import compare


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

    def test_comparator_requires_utilization_gain_without_loss_regression(self):
        base = {"heldout_loss": 0.65, "expert_utilization": 0.60,
                "capacity_drop": 0.01, "n_experts": 64, "resume_ok": True}
        better = dict(base, expert_utilization=0.75, heldout_loss=0.64)
        ok, _ = compare(base, better, 0.0)
        self.assertTrue(ok)
        worse_loss = dict(better, heldout_loss=0.66)
        ok, _ = compare(base, worse_loss, 0.0)
        self.assertFalse(ok)

    def test_comparator_keeps_restart_as_hard_gate(self):
        base = {"heldout_loss": 0.65, "expert_utilization": 0.60,
                "capacity_drop": 0.01, "n_experts": 64, "resume_ok": True}
        candidate = dict(base, expert_utilization=0.90, heldout_loss=0.60,
                         resume_ok=False)
        ok, reason = compare(base, candidate, 0.0)
        self.assertFalse(ok)
        self.assertIn("restart/resume", reason)

    def test_missing_behavior_metric_blocks_promotion(self):
        with self.assertRaisesRegex(ValueError, "expert_utilization"):
            validate_balance_ab_result({
                "heldout_loss": 0.64,
                "capacity_drop": 0.0,
                "n_experts": 80,
                "resume_ok": True,
            })



class UpstreamBalanceImplementationTests(unittest.TestCase):
    def _paged_model(self):
        cfg = RecurConfig(
            vocab_size=265, d_model=8, n_head=1, d_ff=16, block=8,
            n_prelude=1, n_recur=1, n_coda=0, max_steps=1,
            use_pool=True, pool_experts=4, pool_d_ff=8, pool_top_k=1,
            pool_max=4,
        )
        tmp = tempfile.TemporaryDirectory()
        seed = RecurCoder(cfg)
        store.save(seed, tmp.name, step=0, val=1.0, cfg=cfg.__dict__)
        model, _, pool, _ = build_paged(
            tmp.name, torch.device("cpu"), resident=2, ram_capacity=4)
        model.train()
        return tmp, model, pool

    def test_balance_zero_is_exactly_off(self):
        tmp, model, pool = self._paged_model()
        try:
            pool.balance = 0.0
            x = torch.tensor([[1, 2, 3, 4]])
            y = torch.tensor([[2, 3, 4, 5]])
            _, loss = model(x, y)
            self.assertEqual(float(model.pool_balance()), 0.0)
            self.assertIsNone(pool.balance_term())
            self.assertTrue(torch.isfinite(loss))
        finally:
            tmp.cleanup()

    def test_positive_balance_is_finite_and_reaches_router_gradient(self):
        tmp, model, pool = self._paged_model()
        try:
            pool.balance = 0.001
            # Make recent use deliberately uneven so the balance term is
            # non-zero and therefore has a meaningful router gradient.
            pool.recent.zero_()
            pool.recent[0] = 1.0
            x = torch.tensor([[1, 2, 3, 4]])
            y = torch.tensor([[2, 3, 4, 5]])
            _, lm_loss = model(x, y)
            term = model.pool_balance()
            self.assertTrue(torch.isfinite(term))
            self.assertNotEqual(float(term.detach()), 0.0)
            (lm_loss + term).backward()
            grads = [
                m.router.weight.grad
                for m in model.modules()
                if hasattr(m, "router") and getattr(m.router, "weight", None) is not None
            ]
            self.assertTrue(any(g is not None and torch.isfinite(g).all()
                                and float(g.abs().sum()) > 0 for g in grads))
        finally:
            tmp.cleanup()


if __name__ == "__main__":
    unittest.main()
