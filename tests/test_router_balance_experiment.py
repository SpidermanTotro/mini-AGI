import tempfile
import unittest

import torch

from minagi.paged import PagedPool
from minagi.pool import PooledMLP


class RouterBalanceExperimentTests(unittest.TestCase):
    def make_pool(self, root):
        # A tiny paged pool is enough to exercise selection/balance state.
        return PagedPool(str(root), d_model=4, d_ff=8, n_experts=4,
                         resident=2, ram_capacity=4, device="cpu")

    def test_default_balance_is_off(self):
        with tempfile.TemporaryDirectory() as tmp:
            pool = self.make_pool(tmp)
            self.assertEqual(pool.balance, 0.0)
            pool.begin_text(explore=True)
            self.assertIsNone(pool.balance_term())

    def test_balance_and_legacy_exploration_cannot_mix(self):
        with tempfile.TemporaryDirectory() as tmp:
            pool = self.make_pool(tmp)
            pool.balance = 3.5e-4
            pool.explore_bias = 0.65
            with self.assertRaisesRegex(ValueError, "mutually exclusive"):
                pool.begin_text(explore=True)

    def test_balance_term_has_router_gradient(self):
        with tempfile.TemporaryDirectory() as tmp:
            pool = self.make_pool(tmp)
            pool.balance = 3.5e-4
            pool.explore_bias = 0.0
            pool.recent[:] = torch.tensor([8.0, 1.0, 1.0, 0.0])
            # Back the experts with tensors, then clear the resident card so
            # this forward must execute the first-admission path where the
            # balance term is intentionally computed.
            for i in range(4):
                pool.tiers.put(i, {
                    "w1": torch.zeros(8, 4), "w3": torch.zeros(8, 4),
                    "w2": torch.zeros(4, 8),
                }, dirty=False)
            pool.slots[:] = [-1] * len(pool.slots)
            route = PooledMLP(pool, d_model=4, top_k=1,
                              grad_checkpoint=False)
            route.train()
            pool.begin_text(explore=True)
            x = torch.randn(1, 3, 4, requires_grad=True)
            route(x)
            term = pool.balance_term()
            self.assertIsNotNone(term)
            self.assertTrue(term.requires_grad)
            term.backward()
            self.assertIsNotNone(route.router.weight.grad)
            self.assertGreater(float(route.router.weight.grad.abs().sum()), 0.0)
            # Detached trunk states mean this auxiliary term alone cannot bend x.
            self.assertIsNone(x.grad)

    def test_even_usage_and_uniform_router_is_zero_balance(self):
        with tempfile.TemporaryDirectory() as tmp:
            pool = self.make_pool(tmp)
            pool.balance = 3.5e-4
            pool.explore_bias = 0.0
            pool.recent.fill_(1.0)
            for i in range(4):
                pool.tiers.put(i, {
                    "w1": torch.zeros(8, 4), "w3": torch.zeros(8, 4),
                    "w2": torch.zeros(4, 8),
                }, dirty=False)
            pool.slots[:] = [-1] * len(pool.slots)
            route = PooledMLP(pool, d_model=4, top_k=1,
                              grad_checkpoint=False)
            route.train()
            with torch.no_grad():
                route.router.weight.zero_()
            pool.begin_text(explore=True)
            route(torch.zeros(1, 2, 4))
            self.assertAlmostEqual(float(pool.balance_term().detach()), 0.0,
                                   places=7)

    def test_balance_term_resets_for_next_forward(self):
        with tempfile.TemporaryDirectory() as tmp:
            pool = self.make_pool(tmp)
            pool.balance = 3.5e-4
            pool.explore_bias = 0.0
            pool.note_balance(torch.tensor(1.0, requires_grad=True))
            self.assertIsNotNone(pool.balance_term())
            pool.begin_text(explore=True)
            self.assertIsNone(pool.balance_term())

    def test_eval_forward_does_not_create_balance_term(self):
        with tempfile.TemporaryDirectory() as tmp:
            pool = self.make_pool(tmp)
            pool.balance = 3.5e-4
            pool.explore_bias = 0.0
            for i in range(4):
                pool.tiers.put(i, {
                    "w1": torch.zeros(8, 4), "w3": torch.zeros(8, 4),
                    "w2": torch.zeros(4, 8),
                }, dirty=False)
            pool.slots[:] = [-1] * len(pool.slots)
            route = PooledMLP(pool, d_model=4, top_k=1,
                              grad_checkpoint=False)
            route.eval()
            pool.begin_text(explore=False)
            with torch.no_grad():
                route(torch.zeros(1, 2, 4))
            self.assertIsNone(pool.balance_term())


if __name__ == "__main__":
    unittest.main()
