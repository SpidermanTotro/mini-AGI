"""Regression tests for GradSNR with conditional gradients (stdlib unittest)."""
import unittest

import torch
from minagi.optim import GradSNR


def observe(meter, params, grads):
    for p, g in zip(params, grads):
        p.grad = None if g is None else torch.tensor(g, dtype=p.dtype)
    return meter.observe(params)


class GradSNRRegressionTests(unittest.TestCase):
    def test_missing_gradient_resets_coordinates(self):
        ps = [torch.nn.Parameter(torch.zeros(2)), torch.nn.Parameter(torch.zeros(3))]
        meter = GradSNR(beta=0.9)
        for _ in range(8):
            observe(meter, ps, [[1, 2], [3, 4, 5]])
        self.assertAlmostEqual(meter.ratio(), 1.0, places=5)
        self.assertIsNone(observe(meter, ps, [[1, 2], None]))
        self.assertEqual(meter.n, 1)
        self.assertIsNone(observe(meter, ps, [[1, 2], [3, 4, 5]]))
        self.assertEqual(meter.n, 1)

    def test_equal_size_parameter_swap_resets(self):
        ps = [torch.nn.Parameter(torch.zeros(2)), torch.nn.Parameter(torch.zeros(2))]
        meter = GradSNR()
        for _ in range(8):
            observe(meter, ps, [[1, 1], None])
        self.assertIsNotNone(meter.ratio())
        self.assertIsNone(observe(meter, ps, [None, [1, 1]]))
        self.assertEqual(meter.n, 1)

    def test_all_missing_gradients_do_not_change_state(self):
        ps = [torch.nn.Parameter(torch.zeros(1))]
        meter = GradSNR()
        observe(meter, ps, [[2]])
        self.assertIsNone(observe(meter, ps, [None]))
        self.assertEqual(meter.n, 1)
        self.assertIsNone(observe(meter, ps, [[2]]))
        self.assertEqual(meter.n, 2)

    def test_constant_gradient_ratio(self):
        ps = [torch.nn.Parameter(torch.zeros(3))]
        meter = GradSNR()
        for _ in range(16):
            observe(meter, ps, [[1, 2, 3]])
        self.assertAlmostEqual(meter.ratio(), 1.0, places=5)

    def test_zero_gradients(self):
        ps = [torch.nn.Parameter(torch.zeros(2))]
        meter = GradSNR()
        for _ in range(10):
            self.assertIsNone(observe(meter, ps, [[0, 0]]))


if __name__ == "__main__":
    unittest.main()
