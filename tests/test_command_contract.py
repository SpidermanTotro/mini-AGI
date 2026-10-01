import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def _vopt():
    """Prefer the project venv; fall back to whatever runs the tests."""
    venv = Path.home() / "src/mini-AGI/.venv/bin/python"
    return str(venv) if venv.exists() else sys.executable


class CommandLineContractTests(unittest.TestCase):
    """
    Every argument train.py reads must be one it defines.

    Upstream deleted options Greenlight had added - `--margin`,
    `--dwell-chars`, `--segment-chars` - and each deletion merged cleanly,
    because deleting an option is not a conflict with a file that still reads
    it. The result was a `read` that raised AttributeError on its first line
    of real work, behind a fully green test suite.

    Nothing catches that except comparing what the code reads against what it
    defines, so this does, on every run, in milliseconds.
    """

    def test_every_argument_read_is_defined(self):
        src = (REPO / "train.py").read_text(encoding="utf-8")
        defined = set(re.findall(r'add_argument\(\s*"--([a-z0-9-]+)"', src))
        defined |= set(re.findall(r'dest="([a-z_0-9]+)"', src))
        defined = {d.replace("-", "_") for d in defined}
        used = set(re.findall(r"\bargs\.([a-z_][a-z_0-9]*)", src))
        # `paths` is the positional, not an option
        self.assertEqual(used - defined - {"paths"}, set(),
                         "train.py reads arguments it never defines")


class TrainingLifecycleTests(unittest.TestCase):
    """
    train -> evaluate -> save -> restart -> keep learning, as a subprocess.

    Both of this project's commands, `read` and `stream`, have no unit test
    that constructs them. That is not a gap in the tests; it is the gap the
    bugs lived in: a green suite while `read` could not start and a resumed
    `stream` died at its first optimiser step.

    A real run takes minutes even on a toy corpus, so this is opt-in. Run it
    with GREENLIGHT_LIFECYCLE=1; it is the check that says the thing actually
    works, rather than that its parts do.
    """

    RUN = os.environ.get("GREENLIGHT_LIFECYCLE") == "1"

    @unittest.skipUnless(RUN, "set GREENLIGHT_LIFECYCLE=1 to run the real "
                              "train/resume lifecycle (~minutes)")
    def test_stream_trains_evaluates_saves_and_resumes(self):
        import numpy as np

        py = _vopt()
        with tempfile.TemporaryDirectory() as root:
            data = Path(root) / "data_char"
            data.mkdir()
            rng = np.random.default_rng(0)
            lines = [f"{int(rng.integers(0, 9))} plus {int(rng.integers(0, 9))} "
                     f"is {sum((int(rng.integers(0, 9)),))}" for _ in range(10)]
            text = "".join(lines)
            ids = np.zeros(len(text) + 1, dtype=np.uint16)
            ids[0] = 1
            for i, ch in enumerate(text):
                ids[i + 1] = min(1000, ord(ch))
            for name in ("train.bin", "val.bin"):
                ids.tofile(data / name)

            weights = Path(root) / "weights"
            common = ["--device", "cpu", "stream", "--mix",
                      f"{data}:1.0", "--weights-dir", str(weights),
                      "--context", "256", "--ctx-start", "128",
                      "--chunk", "64", "--lr", "1e-3", "--warmup", "1",
                      "--log-every", "4", "--eval-every", "8",
                      "--eval-chunks", "2", "--grow-k", "0"]
            first = subprocess.run([py, str(REPO / "train.py")] + common
                                   + ["--steps", "8"],
                                   cwd=REPO, capture_output=True, text=True,
                                   timeout=1800)
            self.assertEqual(first.returncode, 0, first.stdout + first.stderr)
            self.assertIn("val ", first.stdout)
            self.assertTrue((weights / "manifest.json").exists())

            # resume from that checkpoint and keep learning
            second = subprocess.run([py, str(REPO / "train.py")] + common
                                    + ["--steps", "16"],
                                    cwd=REPO, capture_output=True, text=True,
                                    timeout=1800)
            self.assertEqual(second.returncode, 0,
                             second.stdout + second.stderr)
            self.assertNotIn("KeyError", second.stderr)


if __name__ == "__main__":
    unittest.main()
