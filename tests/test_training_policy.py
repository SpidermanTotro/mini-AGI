import unittest

from minagi.training_policy import chars_to_steps, growth_held_due, lr_at


class LearningRatePolicyTests(unittest.TestCase):
    def test_warmup_is_linear(self):
        self.assertAlmostEqual(lr_at(0, 100, 1.0, 10), 0.1)
        self.assertAlmostEqual(lr_at(9, 100, 1.0, 10), 1.0)

    def test_cosine_decay_reaches_floor(self):
        self.assertAlmostEqual(lr_at(100, 100, 1.0, 10), 0.1)

    def test_zero_warmup_is_safe(self):
        self.assertAlmostEqual(lr_at(0, 100, 1.0, 0), 1.0)

    def test_steps_past_total_stay_at_floor(self):
        self.assertAlmostEqual(lr_at(150, 100, 1.0, 10), 0.1)


class CadencePolicyTests(unittest.TestCase):
    def test_chars_to_steps_tracks_chunk_size(self):
        self.assertEqual(chars_to_steps(2048, 512), 4)
        self.assertEqual(chars_to_steps(2048, 1024), 2)

    def test_chars_to_steps_never_returns_zero(self):
        self.assertEqual(chars_to_steps(0, 512), 1)
        self.assertEqual(chars_to_steps(100, 0), 100)

    def test_growth_held_cadence(self):
        self.assertTrue(growth_held_due(0, 4))
        self.assertTrue(growth_held_due(40, 4))
        self.assertFalse(growth_held_due(39, 4))

    def test_growth_held_handles_zero_cadence(self):
        self.assertTrue(growth_held_due(0, 0))
        self.assertFalse(growth_held_due(1, 0))


if __name__ == "__main__":
    unittest.main()
