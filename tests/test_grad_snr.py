import unittest

import torch

from minagi.optim import GradSNR


class GradSNRTests(unittest.TestCase):
    def test_changing_active_gradient_set_resets_meter_without_crashing(self):
        a = torch.nn.Parameter(torch.zeros(3))
        b = torch.nn.Parameter(torch.zeros(2))
        meter = GradSNR(beta=0.9)

        a.grad = torch.ones_like(a)
        b.grad = torch.ones_like(b)
        self.assertIsNone(meter.observe([a, b]))
        self.assertEqual(meter.m.numel(), 5)

        b.grad = None
        self.assertIsNone(meter.observe([a, b]))
        self.assertEqual(meter.m.numel(), 3)
        self.assertEqual(meter.n, 1)

    def test_equal_length_but_different_parameter_set_resets_meter(self):
        a = torch.nn.Parameter(torch.zeros(2))
        b = torch.nn.Parameter(torch.zeros(2))
        meter = GradSNR(beta=0.9)

        a.grad = torch.ones_like(a)
        self.assertIsNone(meter.observe([a, b]))

        a.grad = None
        b.grad = torch.full_like(b, 2.0)
        self.assertIsNone(meter.observe([a, b]))
        self.assertEqual(meter.n, 1)
        self.assertTrue(torch.equal(meter.m, torch.full((2,), 2.0)))


if __name__ == "__main__":
    unittest.main()
