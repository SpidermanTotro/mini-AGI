"""A/B reports must never invent provenance, held-out data or cold resume."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.build_balance_ab_report import build_report, fingerprint


class BuildBalanceABReportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.start = self.root / "start"
        self.weights = self.root / "weights"
        self.start.mkdir()
        self.weights.mkdir()
        (self.start / "weights.bin").write_bytes(b"start checkpoint")
        (self.weights / "weights.bin").write_bytes(b"resume checkpoint")
        self.train = self.root / "train.txt"
        self.test = self.root / "heldout.txt"
        self.train.write_text("training text")
        self.test.write_text("separate held out text")
        self.log = self.root / "cold-restart.log"
        self.log.write_text("fresh process restarted optimizer and evaluated")
        self.proof = self.root / "resume.json"
        self.proof_data = {
            "new_process": True,
            "exit_code": 0,
            "optimizer_restored": True,
            "heldout_eval_after_resume": True,
            "step_before": 100,
            "step_after": 101,
            "source_weights_sha256": fingerprint(self.weights),
            "log_path": str(self.log),
            "log_sha256": hashlib.sha256(self.log.read_bytes()).hexdigest(),
        }
        self.write_evidence()
        self.history = self.root / "history.jsonl"
        self.experts = self.root / "experts.jsonl"
        self.write_history(
            {"kind": "val", "step": 100, "val": 0.75},
            {"kind": "step", "step": 100, "pool_dropped": 2, "pool_requested": 100},
        )
        self.experts.write_text(json.dumps({"step": 100, "use": [4, 0, 2, 1]}) + "\n")

    def write_evidence(self):
        self.proof.write_text(json.dumps(self.proof_data))

    def write_history(self, *rows):
        self.history.write_text("".join(json.dumps(x) + "\n" for x in rows))

    def build(self, **changed):
        kw = dict(
            initial_checkpoint=self.start,
            training_corpus=self.train,
            heldout_corpus=self.test,
            seed=42,
            training_steps=100,
            balance_strength=0.001,
            cold_resume_evidence=self.proof,
        )
        kw.update(changed)
        with patch("tools.build_balance_ab_report.resume_preflight",
                   return_value={"health": "healthy"}), \
             patch("tools.build_balance_ab_report.diagnose_experts",
                   return_value={"health": "healthy",
                                 "observations": {"experts": 4}}):
            return build_report(self.history, self.experts, self.weights, **kw)

    def test_valid_report_contains_real_schema_and_cold_log(self):
        report = self.build()
        self.assertEqual(report["heldout_loss"], 0.75)
        self.assertEqual(report["expert_utilization"], 0.75)
        self.assertEqual(report["capacity_drop"], 0.02)
        self.assertEqual(report["n_experts"], 4)
        self.assertEqual(report["initial_checkpoint_sha256"], fingerprint(self.start))
        self.assertEqual(report["training_corpus_sha256"], fingerprint(self.train))
        self.assertEqual(report["heldout_corpus_sha256"], fingerprint(self.test))
        self.assertTrue(report["resume_ok"])
        self.assertEqual(report["cold_resume_evidence"]["step_after"], 101)

    def test_heldout_must_be_actual_validation_event(self):
        self.write_history({"kind": "step", "loss": 0.8, "pool_dropped": 0,
                            "pool_requested": 100})
        with self.assertRaisesRegex(ValueError, "held-out evaluation"):
            self.build()

    def test_capacity_drop_without_denominator_fails_closed(self):
        self.write_history({"kind": "val", "val": 0.75},
                           {"kind": "step", "pool_dropped": 2})
        with self.assertRaisesRegex(ValueError, "pool_requested"):
            self.build()

    def test_preflight_is_not_substitute_for_cold_restart(self):
        self.proof.unlink()
        with self.assertRaises((OSError, ValueError)):
            self.build()

    def test_optimizer_restart_must_advance_step(self):
        self.proof_data["step_after"] = 100
        self.write_evidence()
        with self.assertRaisesRegex(ValueError, "advanced optimizer steps"):
            self.build()

    def test_proof_log_cannot_be_silently_swapped(self):
        self.log.write_text("tampered log")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            self.build()

    def test_nonfinite_metrics_and_overlapping_corpus_fail_closed(self):
        self.write_history({"kind": "val", "val": float("nan")},
                           {"kind": "step", "pool_dropped": 2, "pool_requested": 100})
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            self.build()
        with self.assertRaisesRegex(ValueError, "identical fingerprints"):
            self.build(heldout_corpus=self.train)

    def test_checkpoint_fingerprint_rejects_symlinks(self):
        link = self.weights / "outside"
        try:
            link.symlink_to(self.train)
        except (OSError, NotImplementedError):
            self.skipTest("symlinks unavailable")
        with self.assertRaisesRegex(ValueError, "symlink"):
            fingerprint(self.weights)


if __name__ == "__main__":
    unittest.main()
