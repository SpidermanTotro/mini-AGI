import os
import tempfile
import unittest

import numpy as np
import torch

from minagi.paged import Tiers


class ReadOnlyPagedStorageTests(unittest.TestCase):
    def _write_expert(self, root, i, value):
        a = np.full((2, 2), value, dtype=np.float32)
        np.savez(os.path.join(root, f"e{i:05d}.npz"), w1=a, w3=a, w2=a)

    def test_read_only_eviction_never_writes_source_expert(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._write_expert(tmp, 0, 1.0)
            self._write_expert(tmp, 1, 2.0)
            path = os.path.join(tmp, "e00000.npz")
            with open(path, "rb") as source:
                before = source.read()

            tiers = Tiers(tmp, 2, 2, ram_capacity=1,
                          device="cpu", read_only=True)
            ent = tiers.fetch(0)
            ent["w1"].fill_(99.0)
            tiers.put(0, ent, dirty=True)

            # Fetching another expert evicts expert 0. A dry read must discard
            # its transient in-memory mutation rather than writing it back.
            tiers.fetch(1)
            tiers.flush()

            with open(path, "rb") as source:
                after = source.read()
            self.assertEqual(after, before)
            self.assertEqual(tiers.writebacks, 0)
            self.assertFalse(tiers.dirty)


if __name__ == "__main__":
    unittest.main()
