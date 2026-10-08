"""Contract and implementation tests for experimental expert balancing."""

import json
import tempfile
import unittest
from pathlib import Path

import torch

from minagi.recur import RecurConfig, RecurCoder
from train import build_paged
from minagi import store
from tools.compare_balance_ab import compare, load, validate_result


def report(**changes):
    """Small comparable synthetic A/B report; not a training benchmark."""
    data = {
        "heldout_loss": 0.65,
        "expert_utilization": 0.60,
        "capacity_drop": 0.01,
        "n_experts": 64,
        "resume_ok": True,
        "balance_strength": 0.0,
        "initial_checkpoint_sha256": "a" * 64,
        "training_corpus_sha256": "b" * 64,
        "heldout_corpus_sha256": "c" * 64,
        "seed": 42,
        "training_steps": 100,
    }
    data.update(changes)
    return data


def validate_balance_ab_result(result):
    validate_result(result)
    if not result["resume_ok"]:
        raise ValueError("candidate must pass restart/resume before promotion")
    return True


class UpstreamBalanceABContractTests(unittest.TestCase):
    def test_complete_candidate_can_be_compared(self):
        self.assertTrue(validate_balance_ab_result(report()))

    def test_resume_failure_blocks_promotion(self):
        with self.assertRaisesRegex(ValueError, "restart/resume"):
            validate_balance_ab_result(report(resume_ok=False))

    def test_comparator_requires_utilization_gain_without_loss_regression(self):
        base = report()
        better = report(balance_strength=0.001, expert_utilization=0.75,
                        heldout_loss=0.64)
        ok, _ = compare(base, better, 0.0)
        self.assertTrue(ok)
        worse_loss = dict(better, heldout_loss=0.66)
        ok, _ = compare(base, worse_loss, 0.0)
        self.assertFalse(ok)

    def test_comparator_keeps_restart_as_hard_gate(self):
        base = report()
        candidate = report(balance_strength=0.001, expert_utilization=0.90,
                           heldout_loss=0.60, resume_ok=False)
        ok, reason = compare(base, candidate, 0.0)
        self.assertFalse(ok)
        self.assertIn("restart/resume", reason)

    def test_missing_behavior_metric_blocks_promotion(self):
        candidate = report()
        del candidate["expert_utilization"]
        with self.assertRaisesRegex(ValueError, "expert_utilization"):
            validate_balance_ab_result(candidate)

    def test_rejects_cross_checkpoint_corpus_seed_and_step_comparisons(self):
        base = report()
        for key, value in (
            ("initial_checkpoint_sha256", "e" * 64),
            ("training_corpus_sha256", "f" * 64),
            ("heldout_corpus_sha256", "d" * 64),
            ("seed", 43),
            ("training_steps", 101),
        ):
            with self.subTest(field=key):
                candidate = report(balance_strength=0.001,
                                   expert_utilization=0.75, **{key: value})
                ok, reason = compare(base, candidate)
                self.assertFalse(ok)
                self.assertIn(key, reason)

    def test_rejects_comparison_with_balance_enabled_on_baseline(self):
        base = report(balance_strength=0.001)
        candidate = report(balance_strength=0.001, expert_utilization=0.75)
        ok, reason = compare(base, candidate)
        self.assertFalse(ok)
        self.assertIn("baseline", reason)

    def test_capacity_drop_regression_blocks_promotion(self):
        base = report()
        candidate = report(balance_strength=0.001, expert_utilization=0.75,
                           capacity_drop=0.04)
        ok, reason = compare(base, candidate)
        self.assertFalse(ok)
        self.assertIn("capacity_drop", reason)
        self.assertTrue(compare(base, candidate, max_capacity_drop_regression=0.03)[0])

    def test_unapproved_expert_growth_blocks_promotion(self):
        base = report()
        candidate = report(balance_strength=0.001, expert_utilization=0.75,
                           n_experts=80)
        ok, reason = compare(base, candidate)
        self.assertFalse(ok)
        self.assertIn("expert_growth", reason)
        self.assertTrue(compare(base, candidate, max_expert_growth=16)[0])

    def test_invalid_and_nonfinite_values_fail_closed(self):
        for field, value in (
            ("heldout_loss", float("nan")),
            ("expert_utilization", 1.1),
            ("capacity_drop", -0.01),
            ("n_experts", True),
            ("resume_ok", "true"),
            ("initial_checkpoint_sha256", "not-a-hash"),
            ("balance_strength", float("inf")),
        ):
            with self.subTest(field=field):
                candidate = report(balance_strength=0.001, expert_utilization=0.75)
                candidate[field] = value
                ok, _ = compare(report(), candidate)
                self.assertFalse(ok)

    def test_missing_metadata_is_not_silently_accepted(self):
        candidate = report(balance_strength=0.001, expert_utilization=0.75)
        del candidate["training_corpus_sha256"]
        ok, reason = compare(report(), candidate)
        self.assertFalse(ok)
        self.assertIn("training_corpus_sha256", reason)

    def test_load_requires_valid_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            file = Path(tmp) / "baseline.json"
            file.write_text(json.dumps(report()), encoding="utf-8")
            self.assertEqual(load(file)["seed"], 42)
            file.write_text(json.dumps({"heldout_loss": 0.5}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "missing"):
                load(file)


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
