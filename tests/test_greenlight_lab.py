import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from greenlight_lab import read_feedback, run_lab, write_feedback


class GreenlightLabTests(unittest.TestCase):
    def test_feedback_jsonl_requires_valid_reviewed_pairs(self):
        with tempfile.TemporaryDirectory() as root:
            feedback = Path(root) / "feedback.jsonl"
            feedback.write_text(json.dumps({
                "prompt": "<user>Q</user>\n<bot>",
                "target": "Reviewed answer",
            }) + "\n", encoding="utf-8")
            self.assertEqual(read_feedback(feedback), [
                ("<user>Q</user>\n<bot>", "Reviewed answer")])

            feedback.write_text("[]\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "must be an object"):
                read_feedback(feedback)

    def test_feedback_is_written_as_trainable_text(self):
        with tempfile.TemporaryDirectory() as root:
            target = Path(root) / "approved.txt"
            write_feedback(target, [
                ("<user>Question</user>\n<bot>", "Human correction"),
                ("Continue this sentence", "with a reviewed ending."),
            ])
            self.assertEqual(target.read_text(encoding="utf-8"),
                             "<user>Question</user>\n<bot>\n"
                             "Human correction\n</bot>\n"
                             "Continue this sentence\n"
                             "with a reviewed ending.\n")

    def test_rounds_evaluate_after_training_and_feed_reviewed_examples_forward(self):
        with tempfile.TemporaryDirectory() as root:
            root_path = Path(root)
            feedback = root_path / "feedback.jsonl"
            feedback.write_text(json.dumps({
                "prompt": "<user>Q</user>\n<bot>",
                "target": "Reviewed answer",
            }) + "\n", encoding="utf-8")
            weights_dir = root_path / "weights"
            train_corpus = root_path / "data.txt"
            train_corpus.write_text("training data\n", encoding="utf-8")
            held_out = root_path / "val.txt"
            held_out.write_text("held out\n", encoding="utf-8")
            config = Path("config-16gb.yaml")
            calls = []

            def fake_runner(command, cwd, env, log_path=None):
                calls.append((command, env.copy(), log_path))
                if "-m" in command:
                    output = Path(command[command.index("--output") + 1])
                    output.parent.mkdir(parents=True, exist_ok=True)
                    output.write_text(json.dumps({
                        "step": len(calls), "params": 100,
                        "results": [{"exact_match": True}],
                    }), encoding="utf-8")
                return 0

            self.assertEqual(run_lab(
                train_corpus=train_corpus,
                held_out=held_out,
                weights_dir=weights_dir,
                config=config,
                rounds=2,
                minutes_per_round=1,
                feedback_path=feedback,
                interactive=False,
                device="cpu",
                seed=5,
                runner=fake_runner,
                out_dir=root_path / "lab-runs",
            ), 0)
            self.assertEqual(len(calls), 4)
            self.assertEqual(calls[0][0][1:5], [
                "train.py", "--device", "cpu", "read"])
            self.assertIn("--save", calls[0][0])
            command = calls[0][0]
            self.assertEqual(command[command.index("--history") + 1],
                             str(root_path / "lab-runs" / "expert_history.jsonl"))
            self.assertEqual(command[command.index("--sample-log") + 1],
                             str(root_path / "lab-runs" / "samples.txt"))
            self.assertIn("--no-plots", command)
            self.assertIn("--output", calls[1][0])
            self.assertEqual(calls[0][1]["GREENLIGHT_CONFIG"],
                             str(config.resolve()))

    def test_lab_refuses_to_modify_r1_checkpoint(self):
        with tempfile.TemporaryDirectory() as root:
            root_path = Path(root)
            train = root_path / "train.txt"
            held_out = root_path / "val.txt"
            config = root_path / "config.yaml"
            train.write_text("train", encoding="utf-8")
            held_out.write_text("val", encoding="utf-8")
            config.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "refusing to train the R1"):
                run_lab(
                    train_corpus=train,
                    held_out=held_out,
                    weights_dir=Path("greenlight-16g-r1"),
                    config=config,
                    runner=lambda *args, **kwargs: self.fail("trainer must not run"),
                    out_dir=root_path / "runs",
                )


if __name__ == "__main__":
    unittest.main()