import json
import os
import tempfile
import unittest

from minagi.training_doctor import diagnose, diagnose_file


def real_rows():
    """A history shaped exactly as train.py's recorder writes one."""
    return [
        {"kind": "start", "step": None, "weights_dir": "weights", "steps": 24},
        {"kind": "step", "step": 0, "loss": 5.67, "grad_norm": 41.98,
         "chars_per_s": 135.0},
        {"kind": "step", "step": 1, "loss": 3.63, "grad_norm": 13.62,
         "chars_per_s": 140.0},
        {"kind": "val", "step": 1, "val": 2.15, "stderr": 0.02},
        {"kind": "step", "step": 2, "loss": 2.75, "grad_norm": 5.67,
         "chars_per_s": 145.0},
        {"kind": "val", "step": 2, "val": 1.39, "stderr": 0.01},
        {"kind": "saved", "step": 2, "val": 1.39},
        {"kind": "done", "step": None, "best": 1.39},
    ]


class RealHistoryTests(unittest.TestCase):
    """
    The doctor's own fixtures used "event"; the recorder writes "kind".

    Reading only "event" left the doctor finding zero samples in every real
    history and reporting HEALTHY - on good runs and on runs that had crashed.
    These tests use the recorder's own spelling so that cannot come back.
    """

    def test_recorder_rows_are_actually_read(self):
        report = diagnose(real_rows())
        self.assertEqual(report["observations"]["step_samples"], 3)
        self.assertEqual(report["observations"]["val_samples"], 2)

    def test_a_healthy_real_run_is_clean(self):
        report = diagnose(real_rows())
        self.assertEqual(report["health"], "healthy")
        codes = {f["code"] for f in report["findings"]}
        self.assertIn("LOSS_IMPROVING", codes)
        self.assertIn("VAL_IMPROVING", codes)

    def test_a_run_that_died_partway_is_critical(self):
        # This is the shape of the resume crash: a start record, then nothing.
        # No loss to be bad, so every loss-based check passes vacuously.
        rows = [r for r in real_rows() if r["kind"] == "start"]
        report = diagnose(rows)
        self.assertEqual(report["health"], "critical")
        codes = {f["code"] for f in report["findings"]}
        self.assertIn("RUN_TRUNCATED", codes)
        self.assertIn("NO_TELEMETRY", codes)

    def test_a_run_cut_off_before_its_done_record_is_flagged(self):
        rows = [r for r in real_rows() if r["kind"] != "done"]
        report = diagnose(rows)
        codes = {f["code"] for f in report["findings"]}
        self.assertIn("RUN_TRUNCATED", codes)

    def test_steps_without_any_evaluation_are_flagged(self):
        rows = [r for r in real_rows()
                if r["kind"] in ("start", "step", "done")]
        report = diagnose(rows)
        codes = {f["code"] for f in report["findings"]}
        self.assertIn("NO_EVALUATION", codes)

    def test_both_spellings_are_accepted(self):
        rows = [{"event": "step", "loss": 5.0 - i, "grad_norm": 1.0,
                 "chars_per_s": 10.0} for i in range(8)]
        report = diagnose(rows)
        self.assertEqual(report["observations"]["step_samples"], 8)


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


class ResumePreflightTests(unittest.TestCase):
    """
    The restart check, and proof that it can fail.

    The bug it exists for took a resumed run down at its first optimiser step,
    before it wrote a single telemetry row - so the history doctor had nothing
    to look at and every loss check passed vacuously. This asks the only
    question that mattered, by replaying the restore and stepping.
    """

    def build_checkpoint(self, root):
        import torch
        from unittest.mock import patch

        import train
        from minagi.create import create
        from minagi.store import save

        weights = os.path.join(root, "weights")
        with patch.dict("os.environ", {"MINI_AGI_CONFIG": "",
                                       "GREENLIGHT_CONFIG": ""}):
            cfg = create(weights, seed=5, verbose=False, d_model=8, n_head=2,
                         trunk_d_ff=12, block=8, max_steps=2, experts=4,
                         resident=2, d_ff=16, depth=1, top_k=2)
            model, _, pool, _ = train.build_paged(
                weights, torch.device("cpu"), resident=2, ram_capacity=4)
            pool.swap_to([0, 1])
            opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
            pool.attach_optimiser(opt)
            loss = sum(p.square().mean() for p in model.parameters())
            loss.backward()
            opt.step()
            save(model, weights, step=1, opt=opt, cfg=cfg)
        return weights

    def test_a_restorable_checkpoint_passes(self):
        from minagi.training_doctor import resume_preflight

        with tempfile.TemporaryDirectory() as root:
            weights = self.build_checkpoint(root)
            report = resume_preflight(weights)
            self.assertEqual(report["health"], "healthy", report["findings"])
            self.assertIsNone(report["observations"]["first_step_raised"])
            self.assertEqual(report["observations"]["restored_without_step"], 0)

    def test_it_catches_moments_restored_without_a_step_counter(self):
        """
        The check itself is tested by removing the repair and watching the
        doctor notice. A preflight that cannot fail is an opinion.
        """
        from minagi import store
        from minagi.training_doctor import resume_preflight

        with tempfile.TemporaryDirectory() as root:
            weights = self.build_checkpoint(root)
            original = store._stamp_missing_steps
            store._stamp_missing_steps = lambda opt: None
            try:
                report = resume_preflight(weights)
            finally:
                store._stamp_missing_steps = original
            self.assertEqual(report["health"], "critical", report["findings"])
            codes = {f["code"] for f in report["findings"]}
            self.assertIn("MISSING_OPTIM_STEP", codes)

    def test_a_directory_that_is_not_a_checkpoint_is_critical(self):
        from minagi.training_doctor import resume_preflight

        with tempfile.TemporaryDirectory() as root:
            report = resume_preflight(root)
            self.assertEqual(report["health"], "critical")
            self.assertIn("NO_CHECKPOINT",
                          {f["code"] for f in report["findings"]})


if __name__ == "__main__":
    unittest.main()
