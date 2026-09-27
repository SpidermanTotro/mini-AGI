import io
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from minagi.training.ponder_probe import run_ponder_probe


class _Encoding:
    ids = [1, 2]


class _Tokenizer:
    def encode(self, _prompt):
        return _Encoding()


class _Model:
    def __init__(self):
        self.calls = 0

    def __call__(self, _ids, collect=False):
        self.calls += 1
        step = 1.0 if self.calls <= 2 else 1.5
        return None, {"steps": torch.tensor([[step]])}


class PonderProbeRunnerTests(unittest.TestCase):
    def test_reports_adaptive_depth_and_returns_success(self):
        args = SimpleNamespace(
            device="cpu", ckpt="weights", data="math",
            task="add", n=2, max_digits=2,
        )
        task = lambda _rng, digits, _flag: f"{digits}+{digits}={digits * 2}"
        model = _Model()

        with (
            patch("minagi.training.ponder_probe.load_recur",
                  return_value=(model, {})),
            patch("minagi.training.ponder_probe.load_tokenizer",
                  return_value=_Tokenizer()),
            patch("corpora.arithmetic.TASKS",
                  {"add": (task, 9, None)}),
            patch("sys.stdout", new_callable=io.StringIO) as stdout,
        ):
            status = run_ponder_probe(args)

        report = stdout.getvalue()
        self.assertEqual(status, 0)
        self.assertEqual(model.calls, 4)
        self.assertIn("1-digit 1.00 steps -> 2-digit 1.50 steps (+0.50)", report)
        self.assertIn("adaptive compute is working", report)


if __name__ == "__main__":
    unittest.main()
