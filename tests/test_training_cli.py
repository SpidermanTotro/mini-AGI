import argparse
import unittest
from types import SimpleNamespace

from minagi.training.cli import add_ponder_probe_command, dispatch


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


class PonderProbeParserTests(unittest.TestCase):
    def test_defaults_and_handler_are_owned_by_greenlight_cli(self):
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="cmd", required=True)
        handler = lambda _args: 0
        add_ponder_probe_command(subparsers, handler)

        args = parser.parse_args(["ponder-probe"])

        self.assertEqual(args.cmd, "ponder-probe")
        self.assertEqual(args.ckpt, "weights")
        self.assertEqual(args.data, "data_math_char")
        self.assertEqual(args.task, "add")
        self.assertEqual(args.n, 20)
        self.assertEqual(args.max_digits, 8)
        self.assertIs(args.fn, handler)

    def test_explicit_options_keep_their_types(self):
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="cmd", required=True)
        add_ponder_probe_command(subparsers, lambda _args: 0)

        args = parser.parse_args([
            "ponder-probe", "--ckpt", "run.pt", "--data", "math",
            "--task", "mul", "--n", "5", "--max-digits", "12",
        ])

        self.assertEqual(
            (args.ckpt, args.data, args.task, args.n, args.max_digits),
            ("run.pt", "math", "mul", 5, 12),
        )


if __name__ == "__main__":
    unittest.main()
