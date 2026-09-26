import unittest
from types import SimpleNamespace

from minagi.training.cli import dispatch


class TrainingCliDispatchTests(unittest.TestCase):
    def test_dispatch_returns_handler_status(self):
        args = SimpleNamespace(fn=lambda _args: 7)
        self.assertEqual(dispatch(args), 7)

    def test_dispatch_normalizes_none_to_success(self):
        args = SimpleNamespace(fn=lambda _args: None)
        self.assertEqual(dispatch(args), 0)

    def test_dispatch_rejects_missing_handler(self):
        with self.assertRaisesRegex(ValueError, "no handler"):
            dispatch(SimpleNamespace())


if __name__ == "__main__":
    unittest.main()
