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



    def corpus(self, root, lines=400):
        """A tiny text corpus and a disjoint held-out folder."""
        train, held = Path(root) / "train", Path(root) / "held"
        train.mkdir(exist_ok=True)
        held.mkdir(exist_ok=True)
        import random
        rng = random.Random(7)
        body = [f"{rng.randint(2, 99)} plus {rng.randint(2, 99)} is "
                f"{0}." for _ in range(lines)]
        body = [b.replace(" is 0.", f" is {int(b.split()[0]) + int(b.split()[2])}.")
                for b in body]
        (train / "train.txt").write_text("\n".join(body) + "\n", encoding="utf-8")
        other = [f"{rng.randint(2, 99)} plus {rng.randint(2, 99)} is 0." for _ in range(80)]
        other = [b.replace(" is 0.", f" is {int(b.split()[0]) + int(b.split()[2])}.")
                 for b in other]
        (held / "val.txt").write_text("\n".join(other) + "\n", encoding="utf-8")
        return train, held

    @unittest.skipUnless(RUN, "set GREENLIGHT_LIFECYCLE=1 to run the real "
                              "train/resume lifecycle (~minutes)")
    def test_read_trains_evaluates_saves_and_resumes(self):
        """
        The same lifecycle through `read`, which is the command the README
        calls the one the project runs.

        The stream lifecycle test cannot catch a broken `read`: two separate
        defects made `read` unrunnable - a deleted `--segment-chars` and a
        `FolderEvaluator` signature that no longer took the argument its caller
        passed - and both passed 138 green tests, because nothing constructed
        this command at all.
        """
        py = _vopt()
        with tempfile.TemporaryDirectory() as root:
            train, held = self.corpus(root)
            weights = Path(root) / "weights"
            common = ["--device", "cpu", "read", str(train),
                      "--weights-dir", str(weights), "--save",
                      "--chunk", "64", "--context", "128",
                      "--context-start", "64", "--passes", "1",
                      "--eval-chars", "2", "--grow-k", "0",
                      "--resident", "2", "--ram-capacity", "4",
                      "--sample-every", "0", "--no-plots",
                      "--held-out", str(held)]
            first = subprocess.run([py, str(REPO / "train.py")] + common,
                                   cwd=REPO, capture_output=True, text=True,
                                   timeout=3600)
            self.assertEqual(first.returncode, 0,
                             first.stdout[-3000:] + first.stderr[-3000:])
            self.assertTrue((weights / "manifest.json").exists())
            self.assertIn("held-out", (first.stdout + first.stderr).lower())

            second = subprocess.run([py, str(REPO / "train.py")] + common,
                                    cwd=REPO, capture_output=True, text=True,
                                    timeout=3600)
            self.assertEqual(second.returncode, 0,
                             second.stdout[-3000:] + second.stderr[-3000:])


if __name__ == "__main__":
    unittest.main()
