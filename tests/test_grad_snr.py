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

    def test_constant_gradient_reads_one_from_the_first_reading(self):
        # Both averages start at the first reading, so neither is biased
        # toward zero and the ratio needs no bias correction: dividing by
        # 1 - beta^n inflated it 6.7x at the eighth reading and let a
        # constant gradient read above one.  (integrated from PR #28)
        a = torch.nn.Parameter(torch.zeros(4))
        meter = GradSNR()
        for n in range(1, 401):
            a.grad = torch.full((4,), 0.5)
            r = meter.observe([a])
            if n >= 8:
                self.assertAlmostEqual(r, 1.0, places=5, msg=(n, r))

    def test_noise_reads_below_one_early(self):
        # Pure noise must never read as more signal than possible - before
        # the fix the eighth reading of pure noise read 4.87.  (PR #28)
        torch.manual_seed(0)
        a = torch.nn.Parameter(torch.zeros(256))
        meter = GradSNR()
        for _ in range(8):
            a.grad = torch.randn(256)
            r = meter.observe([a])
        self.assertIsNotNone(r)
        self.assertLess(r, 1.0)


if __name__ == "__main__":
    unittest.main()
