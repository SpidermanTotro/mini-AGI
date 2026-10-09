"""Pure schema/provenance regression tests; synthetic data are not A/B evidence."""
import json
import tempfile
import unittest
from pathlib import Path

from tools.compare_balance_ab import compare, load


class BalanceABToolTests(unittest.TestCase):
    def report(self, **changes):
        record = {
            "heldout_loss": 1.0,
            "expert_utilization": 0.5,
            "capacity_drop": 0.1,
            "n_experts": 64,
            "resume_ok": True,
            "balance_strength": 0.0,
            "initial_checkpoint_sha256": "a" * 64,
            "training_corpus_sha256": "b" * 64,
            "heldout_corpus_sha256": "c" * 64,
            "seed": 42,
            "training_steps": 100,
        }
        record.update(changes)
        return record

    def test_controlled_candidate_can_clear_narrow_schema_gate(self):
        ok, reason = compare(
            self.report(),
            self.report(balance_strength=0.001, heldout_loss=0.9,
                        expert_utilization=0.7, capacity_drop=0.05),
        )
        self.assertTrue(ok, reason)

    def test_expert_growth_blocks_promotion_by_default(self):
        ok, reason = compare(
            self.report(),
            self.report(balance_strength=0.001, heldout_loss=0.9,
                        expert_utilization=0.7, n_experts=72),
        )
        self.assertFalse(ok)
        self.assertIn("expert_growth", reason)

    def test_loader_requires_checkpoint_and_corpus_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "candidate.json"
            raw = self.report()
            raw.pop("initial_checkpoint_sha256")
            path.write_text(json.dumps(raw))
            with self.assertRaisesRegex(ValueError, "initial_checkpoint_sha256"):
                load(path)

    def test_resume_failure_cannot_clear_gate(self):
        ok, reason = compare(
            self.report(),
            self.report(balance_strength=0.001, heldout_loss=0.8,
                        expert_utilization=0.7, resume_ok=False),
        )
        self.assertFalse(ok)
        self.assertIn("restart/resume", reason)

    def test_different_corpus_seed_or_steps_do_not_compare(self):
        for change in (
            {"heldout_corpus_sha256": "f" * 64},
            {"seed": 43},
            {"training_steps": 99},
        ):
            with self.subTest(change=change):
                ok, reason = compare(
                    self.report(),
                    self.report(balance_strength=0.001, heldout_loss=0.9,
                                expert_utilization=0.7, **change),
                )
                self.assertFalse(ok)
                self.assertIn("provenance mismatch", reason)


if __name__ == "__main__":
    unittest.main()
