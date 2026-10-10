"""Regression tests for safe AdamW checkpoint recovery (stdlib unittest)."""
import unittest

import torch
from minagi.store import _stamp_missing_steps


def make_optimizer():
    p = torch.nn.Parameter(torch.tensor([1.0]))
    q = torch.nn.Parameter(torch.tensor([2.0]))
    return torch.optim.AdamW([p, q]), p, q


def moments(opt, p, step=None):
    st = opt.state[p]
    st["exp_avg"] = torch.ones_like(p)
    st["exp_avg_sq"] = torch.ones_like(p)
    if step is not None:
        st["step"] = torch.tensor(float(step))


class AdamWOrphanStepsTests(unittest.TestCase):
    def test_no_step_provenance_fails_without_mutating_moments(self):
        opt, p, _ = make_optimizer()
        moments(opt, p)
        before = opt.state[p]["exp_avg"].clone()
        with self.assertRaisesRegex(RuntimeError, "no step counters"):
            _stamp_missing_steps(opt)
        self.assertNotIn("step", opt.state[p])
        self.assertTrue(torch.equal(before, opt.state[p]["exp_avg"]))

    def test_missing_step_inherits_known_counter(self):
        opt, p, q = make_optimizer()
        moments(opt, p, step=12)
        moments(opt, q)
        _stamp_missing_steps(opt)
        self.assertEqual(float(opt.state[q]["step"]), 12.0)

    def test_empty_optimizer_state_is_valid(self):
        opt, p, q = make_optimizer()
        _stamp_missing_steps(opt)
        self.assertFalse(opt.state[p])
        self.assertFalse(opt.state[q])

    def test_repaired_optimizer_can_step(self):
        opt, p, q = make_optimizer()
        moments(opt, p, step=4)
        moments(opt, q)
        _stamp_missing_steps(opt)
        p.grad = torch.ones_like(p)
        q.grad = torch.ones_like(q)
        opt.step()
        self.assertEqual(float(opt.state[q]["step"]), 5.0)


if __name__ == "__main__":
    unittest.main()
