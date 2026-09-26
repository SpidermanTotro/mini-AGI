import tempfile
import unittest

import numpy as np
import torch

from minagi.paged import PagedPool


class AuditionTests(unittest.TestCase):
    def make_pool(self, path):
        for i in range(4):
            a = np.full((2, 2), i, dtype=np.float32)
            np.savez(f"{path}/e{i:05d}.npz", w1=a, w3=a, w2=a)
        return PagedPool(path, 2, 2, 4, resident=1, ram_capacity=1)

    def test_audition_does_not_reset_pruning_clock(self):
        with tempfile.TemporaryDirectory() as path:
            pool = self.make_pool(path)
            pool.segments = 10
            pool.last_seen[2] = 3
            pool._auditioned = (2,)
            pool.swap_to([2])
            self.assertEqual(pool.last_seen[2].item(), 3)
            self.assertTrue(pool.ever[2])
            pool._auditioned = ()
            pool.swap_to([2])
            self.assertEqual(pool.last_seen[2].item(), pool.segments)

    def test_pruning_preserves_audition_identity_and_growth_invalidates_scores(self):
        with tempfile.TemporaryDirectory() as path:
            pool = self.make_pool(path)
            pool.swap_to([2])
            pool.segments = 100
            pool.last_seen[:] = torch.tensor([0., 100., 100., 0.])
            pool.last_try[:] = torch.tensor([10., 20., 30., 40.])
            pool._auditioned = (2,)
            pool._fit_raw = torch.arange(4.)
            pool._suppressed = torch.ones(4, dtype=torch.bool)
            self.assertEqual(pool.prune(step=100, survival=10), 2)
            self.assertEqual(pool.uid.tolist(), [1, 2])
            self.assertEqual(pool.last_try.tolist(), [20., 30.])
            self.assertEqual(pool._auditioned, (1,))
            self.assertIsNone(pool._fit_raw)
            self.assertIsNone(pool._suppressed)
            pool._fit_raw = torch.arange(2.)
            pool._suppressed = torch.ones(2, dtype=torch.bool)
            pool.add_experts(1, step=101)
            self.assertEqual(pool.last_try.tolist(), [20., 30., 0.])
            self.assertIsNone(pool._fit_raw)
            self.assertIsNone(pool._suppressed)


if __name__ == "__main__":
    unittest.main()
