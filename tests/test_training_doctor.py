import json
import os
import tempfile
import unittest

from minagi.training_doctor import diagnose, diagnose_file


class TrainingDoctorTests(unittest.TestCase):
    def test_learning_run_is_recognized(self):
        rows = [
            {"event": "step", "step": i, "loss": 5.0 - i * 0.4,
             "grad_norm": 1.0, "chars_per_s": 1000.0}
            for i in range(8)
        ]
        report = diagnose(rows)
        codes = {f["code"] for f in report["findings"]}
        self.assertIn("LOSS_IMPROVING", codes)
        self.assertNotIn("ZERO_GRADIENTS", codes)

    def test_zero_gradients_are_critical(self):
        rows = [
            {"event": "step", "step": i, "loss": 5.0,
             "grad_norm": 0.0, "chars_per_s": 1000.0}
            for i in range(8)
        ]
        report = diagnose(rows)
        self.assertEqual(report["health"], "critical")
        self.assertIn("ZERO_GRADIENTS",
                      {f["code"] for f in report["findings"]})

    def test_nonfinite_loss_is_critical(self):
        rows = [{"event": "step", "loss": float("nan"), "grad_norm": 1.0}]
        report = diagnose(rows)
        self.assertEqual(report["health"], "critical")
        self.assertIn("NONFINITE_LOSS",
                      {f["code"] for f in report["findings"]})

    def test_reads_real_jsonl_shape(self):
        fd, path = tempfile.mkstemp(suffix=".jsonl")
        os.close(fd)
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(json.dumps({"event": "step", "loss": 2.0,
                                    "grad_norm": 1.0, "chars_per_s": 20}) + "\n")
            report = diagnose_file(path)
            self.assertEqual(report["observations"]["step_samples"], 1)
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main()
