import unittest

from minagi.stream import ramp_context


class ContextRampTests(unittest.TestCase):
    def test_disabled_ramp_returns_end_context(self):
        self.assertEqual(ramp_context(0, 100, 4096, 4096), 4096)
        self.assertEqual(ramp_context(0, 100, 8192, 4096), 4096)

    def test_ramp_starts_at_start_context_and_finishes_at_end(self):
        self.assertEqual(ramp_context(0, 100, 1024, 4096), 1024)
        self.assertEqual(ramp_context(35, 100, 1024, 4096), 4096)
        self.assertEqual(ramp_context(100, 100, 1024, 4096), 4096)

    def test_ramp_is_monotonic_and_bounded(self):
        values = [
            ramp_context(step, 1000, 1024, 8192, warm=0.35, granularity=256)
            for step in range(0, 351, 5)
        ]
        self.assertEqual(values[0], 1024)
        self.assertEqual(values[-1], 8192)
        self.assertTrue(all(1024 <= value <= 8192 for value in values))
        self.assertTrue(all(a <= b for a, b in zip(values, values[1:])))

    def test_ramp_uses_requested_granularity_before_endpoint(self):
        values = [
            ramp_context(step, 1000, 1024, 8192, warm=0.35, granularity=256)
            for step in range(0, 350, 7)
        ]
        self.assertTrue(all(value % 256 == 0 for value in values))

    def test_zero_total_is_safe(self):
        value = ramp_context(0, 0, 1024, 4096)
        self.assertEqual(value, 1024)


if __name__ == "__main__":
    unittest.main()
