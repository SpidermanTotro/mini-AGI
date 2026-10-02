import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import training_doctor as training_doctor_cli
from minagi.training_doctor import (diagnose, diagnose_experts, diagnose_file,
                                      diagnose_generation, parse_samples, text_quality,
                                      diagnose_retention)


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


def sample_block(step, val, adapted, raw, domain="stories", text=""):
    block = {"step": step, "val": val, "chars_m": 1.0, "minutes": step,
             "experts": 64,
             "samples": [{"domain": domain, "mode": "raw", "repeated": raw},
                         {"domain": domain, "mode": "adapted",
                          "repeated": adapted}]}
    if text:
        block["samples"].append({"domain": domain, "mode": "raw_text",
                                 "text": text})
    return block


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


class RoutingTests(unittest.TestCase):
    """
    Routing checks, built against the schema PagedPool.telemetry writes and
    validated against a real 128-expert run's expert_history.jsonl.
    """

    def record(self, use, gate=None, admits=None, uid=None, **extra):
        n = len(use)
        out = {"use": list(use), "experts": n, "chars": 1_000_000,
               "segments": 100, "uid": uid or list(range(n))}
        out["gate"] = list(gate) if gate is not None else [1.0] * n
        out["admits"] = list(admits) if admits is not None else [1] * n
        out.update(extra)
        return out

    def test_a_collapsed_router_is_critical(self):
        # 16 experts, half of them take everything. Loss would be fine.
        use = [100.0] * 8 + [1.0] * 8
        report = diagnose_experts([self.record(use)])
        self.assertEqual(report["health"], "critical")
        self.assertIn("ROUTING_COLLAPSED",
                      {f["code"] for f in report["findings"]})

    def test_a_router_with_a_preference_is_not_flagged(self):
        # Six experts clearly preferred, the rest clearly not - but the top
        # ten still hold well under 80% and the pool is far from even, so this
        # is a working router rather than a collapsed or an indifferent one.
        use = [30.0] * 6 + [3.0] * 26
        gate = [1.0 + (i % 8) * 0.05 for i in range(32)]
        report = diagnose_experts([self.record(use, gate=gate)])
        codes = {f["code"] for f in report["findings"]}
        self.assertNotIn("ROUTING_COLLAPSED", codes)
        self.assertNotIn("ROUTING_HAS_NO_PREFERENCE", codes)
        self.assertGreaterEqual(
            report["observations"]["preference"], 0.05)

    def test_perfectly_even_routing_is_flagged(self):
        report = diagnose_experts([self.record([7.0] * 64)])
        codes = {f["code"] for f in report["findings"]}
        self.assertIn("ROUTING_HAS_NO_PREFERENCE", codes)

    def test_flat_gates_are_flagged(self):
        report = diagnose_experts([self.record([7.0] * 16, gate=[1.0] * 16)])
        self.assertIn("GATES_NOT_SEPARATING",
                      {f["code"] for f in report["findings"]})

    def test_separated_gates_are_not_flagged(self):
        gate = [0.2 + i * 0.3 for i in range(16)]
        report = diagnose_experts([self.record([7.0] * 16, gate=gate)])
        self.assertNotIn("GATES_NOT_SEPARATING",
                         {f["code"] for f in report["findings"]})

    def test_a_quarter_of_the_pool_never_used_is_flagged(self):
        use = [10.0] * 12 + [0.0] * 4
        report = diagnose_experts([self.record(use)])
        self.assertIn("DEAD_EXPERTS", {f["code"] for f in report["findings"]})

    def test_experts_removed_mid_run_are_reported(self):
        first = self.record([5.0] * 8, uid=list(range(8)))
        second = self.record([5.0] * 5, uid=[3, 4, 5, 6, 7], chars=2_000_000)
        report = diagnose_experts([first, second])
        codes = {f["code"] for f in report["findings"]}
        self.assertIn("EXPERTS_PRUNED", codes)
        evidence = next(f["evidence"] for f in report["findings"]
                        if f["code"] == "EXPERTS_PRUNED")
        self.assertEqual(evidence["removed"], 3)

    def test_no_records_is_not_a_crash(self):
        report = diagnose_experts([])
        self.assertEqual(report["observations"]["records"], 0)


class RetentionTests(unittest.TestCase):
    """
    Per-domain forgetting. The recorder writes each domain's loss separately,
    which is the only place a forgotten domain is visible.
    """

    def val(self, step, **domains):
        return {"kind": "val", "step": step, "val": 1.0,
                "per_domain": {k: v for k, v in domains.items()}}

    def test_a_domain_that_gets_worse_while_others_improve_is_flagged(self):
        rows = [
            self.val(0, math_=1.00, code=1.00, prose=1.00),
            self.val(10, math_=1.80, code=0.70, prose=0.70),
            self.val(20, math_=2.50, code=0.50, prose=0.50),
        ]
        report = diagnose_retention(rows)
        codes = {f["code"] for f in report["findings"]}
        self.assertIn("DOMAIN_FORGOTTEN", codes)
        evidence = next(f["evidence"] for f in report["findings"]
                        if f["code"] == "DOMAIN_FORGOTTEN")
        self.assertEqual(evidence["domain"], "math_")

    def test_every_domain_improving_is_clean(self):
        rows = [
            self.val(0, math_=2.00, code=2.00, prose=2.00),
            self.val(10, math_=1.40, code=1.30, prose=1.50),
            self.val(20, math_=1.10, code=1.05, prose=1.20),
        ]
        report = diagnose_retention(rows)
        self.assertEqual(report["health"], "healthy", report["findings"])

    def test_one_domain_far_worse_than_another_is_flagged(self):
        rows = [
            self.val(0, math_=1.00, code=1.00),
            self.val(10, math_=0.50, code=5.00),
        ]
        report = diagnose_retention(rows)
        self.assertIn("DOMAIN_IMBALANCE",
                      {f["code"] for f in report["findings"]})

    def test_a_single_evaluation_cannot_forget_anything(self):
        report = diagnose_retention([self.val(0, math_=1.0)])
        self.assertEqual(report["health"], "healthy")
        self.assertEqual(report["observations"]["tracked"], 0)

    def test_history_without_domains_is_not_a_crash(self):
        report = diagnose_retention([{"kind": "val", "step": 1, "val": 1.0}])
        self.assertEqual(report["observations"]["evaluations"], 0)


class GenerationTests(unittest.TestCase):
    """
    Generation quality, measured against the loss falling beside it.

    These metrics are shape-based on purpose. The corpus is code, chess,
    arithmetic and prose in one run, so a word list flags valid output as
    misspelled; what actually breaks is repetition, entropy and word-shape.
    """

    block = staticmethod(sample_block)

    def test_text_metrics_see_a_repeat_loop(self):
        loop = "the " * 60
        q = text_quality(loop)
        self.assertGreater(q["repeated_ngrams"], 0.5)
        run = text_quality("a" * 200)
        self.assertGreaterEqual(run["longest_run"], 200)

    def test_text_metrics_see_prose(self):
        prose = ("Once upon a time there was a small cat that lived near the "
                 "river bank and every morning it walked along the water. ")
        q = text_quality(prose)
        self.assertLess(q["repeated_ngrams"], 0.05)
        self.assertGreater(q["char_entropy"], 3.5)
        self.assertGreater(q["word_like_rate"], 0.8)
        self.assertLess(q["max_consonant_run"], 6)

    def test_alphabet_soup_is_caught(self):
        # The soup is entirely alphabetic, so a "is this a word" ratio cannot
        # see it - this is why the consonant run is the metric.
        soup = "bfjjk hkdkgm tkkitk yigj gibvbrish " * 12
        q = text_quality(soup)
        self.assertGreaterEqual(q["max_consonant_run"], 6)
        self.assertGreater(q["repeated_ngrams"], 0.5)

    def test_loss_falling_while_output_collapses_is_critical(self):
        blocks = []
        for i in range(9):
            early = i < 3
            blocks.append(self.block(
                i, 2.0 if early else 1.0,
                adapted=0.02 if early else 0.60,
                raw=0.10 if early else 0.98))
        report = diagnose_generation(blocks)
        self.assertEqual(report["health"], "critical")
        self.assertIn("LOSS_IMPROVING_OUTPUT_COLLAPSING",
                      {f["code"] for f in report["findings"]})

    def test_loss_and_output_improving_together_is_clean(self):
        blocks = [self.block(i, 2.0 if i < 3 else 1.0,
                             adapted=0.60 if i < 3 else 0.02,
                             raw=0.95 if i < 3 else 0.05)
                  for i in range(9)]
        report = diagnose_generation(blocks)
        codes = {f["code"] for f in report["findings"]}
        self.assertNotIn("LOSS_IMPROVING_OUTPUT_COLLAPSING", codes)
        self.assertNotIn("LOSS_AND_OUTPUT_DIVERGING", codes)

    def test_mild_opposite_movement_warns_without_calling_it_collapse(self):
        blocks = [self.block(i, 2.0 if i < 3 else 1.0,
                             adapted=0.090 if i < 3 else 0.116,
                             raw=0.10)
                  for i in range(9)]
        report = diagnose_generation(blocks)
        codes = {f["code"] for f in report["findings"]}
        self.assertIn("LOSS_AND_OUTPUT_DIVERGING", codes)
        self.assertNotIn("LOSS_IMPROVING_OUTPUT_COLLAPSING", codes)

    def test_unimproved_loss_masks_no_collapse(self):
        # loss flat, output collapsing: reported as repetitive, not as the
        # divergence case, because there was no improvement to diverge from
        blocks = [self.block(i, 1.0, adapted=0.02 if i < 3 else 0.60,
                             raw=0.10) for i in range(9)]
        report = diagnose_generation(blocks)
        codes = {f["code"] for f in report["findings"]}
        self.assertNotIn("LOSS_IMPROVING_OUTPUT_COLLAPSING", codes)
        self.assertIn("GENERATION_STILL_REPETITIVE", codes)

    def test_domains_are_measured_separately_not_averaged(self):
        """
        The aggregate can hide a split: half the domains degenerating while the
        other half improve reads as a small overall rise. Real run: repetition
        up +0.220 on wikipedia while arithmetic fell -0.032 and stories -0.060,
        and the average moved +0.025 and looked like noise.

        A selector that filters blocks rather than samples reports the same
        number for all nine domains and still looks plausible, so this asserts
        the domains actually differ.
        """
        blocks = []
        for i in range(9):
            early = i < 3
            blocks.append(self.block(
                i, 2.0 if early else 1.0,
                adapted=0.05 if early else 0.06,
                raw=0.10, domain="arithmetic", text="the cat sat on the mat"))
            blocks.append(self.block(
                i, 2.0 if early else 1.0,
                adapted=0.20 if early else 0.80,
                raw=0.10, domain="wikipedia", text="the cat sat on the mat"))
        report = diagnose_generation(blocks)
        per = report["observations"]["per_domain"]
        self.assertNotAlmostEqual(per["arithmetic"]["change"],
                                  per["wikipedia"]["change"], places=3)
        self.assertIn("DOMAIN_DEGENERATING",
                      {f["code"] for f in report["findings"]})
        evidence = {f["evidence"]["domain"]: f["evidence"]
                    for f in report["findings"] if f["code"] == "DOMAIN_DEGENERATING"}
        self.assertIn("wikipedia", evidence)
        self.assertNotIn("arithmetic", evidence)

    def test_a_domain_that_improves_is_not_flagged(self):
        blocks = []
        for i in range(9):
            early = i < 3
            blocks.append(self.block(
                i, 2.0 if early else 1.0,
                adapted=0.60 if early else 0.05,
                raw=0.10, domain="stories", text="the cat sat on the mat"))
        report = diagnose_generation(blocks)
        self.assertNotIn("DOMAIN_DEGENERATING",
                         {f["code"] for f in report["findings"]})
        self.assertNotIn("DOMAIN_SLIPPING_TEXT",
                         {f["code"] for f in report["findings"]})

    def test_no_blocks_is_not_a_crash(self):
        self.assertEqual(diagnose_generation([])["observations"]["blocks"], 0)


class SampleParsingTests(unittest.TestCase):
    def test_it_reads_the_shape_the_sampler_writes(self):
        body = """# session started
==============================================================

step 2,037   4.2M of 7,874M characters (0.05%)   25 min   85 experts
grad norm 3.42 against a clip of 1   clipping
held-out loss 2.3895 +/-0.0443 nats   3.4474 bits/char   perplexity 10.91
  arithmetic 2.023   chat 2.729
repeats 87% of 8-grams, greedy with no guard
==============================================================

--- stories ---
prompt: 'Once upon a time '
[raw]  repeated 8-grams 97%
the the the the the the the the the the
[adapted]  repeated 8-grams 2%
thing, sorely thering asted condiler.
"""
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False,
                                         encoding="utf-8") as fh:
            fh.write(body)
            path = fh.name
        try:
            blocks = parse_samples(path)
        finally:
            os.unlink(path)
        self.assertEqual(len(blocks), 1)
        block = blocks[0]
        self.assertEqual(block["step"], 2037)
        self.assertEqual(block["experts"], 85)
        self.assertAlmostEqual(block["val"], 2.3895)
        modes = [s["mode"] for s in block["samples"]]
        self.assertEqual(modes, ["raw", "raw_text", "adapted", "adapted_text"])
        self.assertIn("the the the", block["samples"][1]["text"])
        self.assertIn("sorely", block["samples"][3]["text"])


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


class FalseAlarmTests(unittest.TestCase):
    block = staticmethod(sample_block)

    """
    Reproductions of reported false verdicts, each one a check that used to
    pass when it should not have. A diagnostic that cries wolf on a healthy
    run gets ignored on an unhealthy one.
    """

    def test_a_balanced_small_pool_is_not_called_collapsed(self):
        # "Top ten" out of eight experts IS the whole pool, so every healthy
        # eight-expert run scored 100% and reported ROUTING_COLLAPSED.
        report = diagnose_experts([{"use": [5.0] * 8, "gate": [1.0] * 8,
                                    "admits": [1] * 8, "uid": list(range(8)),
                                    "experts": 8, "chars": 1000,
                                    "segments": 100}])
        self.assertNotIn("ROUTING_COLLAPSED",
                         {f["code"] for f in report["findings"]})
        self.assertNotEqual(report["health"], "critical")

    def test_no_evidence_is_never_a_clean_bill_of_health(self):
        for report in (diagnose_experts([]), diagnose_retention([]),
                       diagnose_generation([])):
            self.assertEqual(report["health"], "critical")
            self.assertIn("NO_EVIDENCE",
                          {f["code"] for f in report["findings"]})

    def test_a_history_with_no_telemetry_is_not_healthy(self):
        report = diagnose([{"kind": "start", "step": None}])
        self.assertEqual(report["health"], "critical")

    def test_a_new_worse_domain_does_not_read_as_collapse(self):
        # Story repetition unchanged, and a code domain arrives late that
        # repeats more. The pooled average rises, which used to be reported
        # as LOSS_IMPROVING_OUTPUT_COLLAPSING. Nothing actually got worse.
        blocks = []
        for i in range(9):
            early = i < 3
            blocks.append(self.block(
                i, 2.0 if early else 1.0, adapted=0.02, raw=0.10,
                domain="stories", text="the cat sat on the mat"))
            if not early:
                blocks[-1]["samples"].append(
                    {"domain": "code", "mode": "adapted", "repeated": 0.70})
                blocks[-1]["samples"].append(
                    {"domain": "code", "mode": "raw", "repeated": 0.95})
        report = diagnose_generation(blocks)
        codes = {f["code"] for f in report["findings"]}
        self.assertNotIn("LOSS_IMPROVING_OUTPUT_COLLAPSING", codes)
        self.assertNotIn("DOMAIN_DEGENERATING", codes)
        self.assertIn("DOMAIN_MIX_CHANGED", codes)

    def test_the_consonant_finding_only_claims_what_it_measured(self):
        # Raw loops, guarded output clean. The old message asserted the
        # problem survived the guard, having never read the guarded text.
        blocks = []
        for i in range(9):
            early = i < 3
            blocks.append(self.block(
                i, 2.0 if early else 1.0, adapted=0.05, raw=0.10,
                domain="stories", text="the cat sat down quietly"))
        for b in blocks[-3:]:
            b["samples"].append({
                "domain": "stories", "mode": "raw_text",
                "text": "bfjjk hkdkgm tkkitk yigj gibvbrish bfjjk hkdkgm"})
            b["samples"].append({
                "domain": "stories", "mode": "adapted_text",
                "text": "the cat sat down quietly and then went to sleep"})
        report = diagnose_generation(blocks)
        evidence = next(f["evidence"] for f in report["findings"]
                        if f["code"] == "OUTPUT_NOT_WORD_SHAPED")
        self.assertGreaterEqual(evidence["raw_max_consonant_run"], 6)
        self.assertLess(evidence["guarded_max_consonant_run"], 6)
        message = next(f["message"] for f in report["findings"]
                       if f["code"] == "OUTPUT_NOT_WORD_SHAPED")
        self.assertNotIn("survives", message)
        self.assertIn("raw output only", message)

    def test_the_guard_claim_is_made_when_it_is_true(self):
        blocks = []
        for i in range(9):
            early = i < 3
            blocks.append(self.block(
                i, 2.0 if early else 1.0, adapted=0.05, raw=0.10,
                domain="stories", text="the cat sat down quietly"))
        for b in blocks[-3:]:
            for mode in ("raw_text", "adapted_text"):
                b["samples"].append({
                    "domain": "stories", "mode": mode,
                    "text": "bfjjk hkdkgm tkkitk yigj gibvbrish bfjjk"})
        report = diagnose_generation(blocks)
        evidence = next(f["evidence"] for f in report["findings"]
                        if f["code"] == "OUTPUT_NOT_WORD_SHAPED")
        self.assertGreaterEqual(evidence["raw_max_consonant_run"], 6)
        self.assertGreaterEqual(evidence["guarded_max_consonant_run"], 6)
        message = next(f["message"] for f in report["findings"]
                       if f["code"] == "OUTPUT_NOT_WORD_SHAPED")
        self.assertIn("AND the guarded text", message)


class MissedFailureTests(unittest.TestCase):
    def test_a_domain_learned_and_then_forgotten_is_caught(self):
        # 5.0 -> 0.8 -> 3.0 is still three times better than where it started,
        # so a first-versus-last comparison called it healthy. Forgetting is
        # measured against what was achieved.
        rows = [
            {"kind": "val", "step": 0, "val": 2.5,
             "per_domain": {"math_": 5.0, "code": 2.0}},
            {"kind": "val", "step": 1, "val": 1.0,
             "per_domain": {"math_": 0.8, "code": 1.0}},
            {"kind": "val", "step": 2, "val": 2.0,
             "per_domain": {"math_": 3.0, "code": 0.9}},
        ]
        report = diagnose_retention(rows)
        codes = {f["code"] for f in report["findings"]}
        self.assertIn("DOMAIN_FORGOTTEN", codes)
        evidence = {f["evidence"]["domain"]: f["evidence"]
                    for f in report["findings"]
                    if f["code"] == "DOMAIN_FORGOTTEN"}
        self.assertIn("math_", evidence)
        self.assertAlmostEqual(evidence["math_"]["best"], 0.8)
        self.assertAlmostEqual(evidence["math_"]["best_at_evaluation"], 1)

    def test_a_domain_that_only_improves_is_not_called_forgotten(self):
        rows = [
            {"kind": "val", "step": 0, "val": 2.0,
             "per_domain": {"math_": 5.0, "code": 5.0}},
            {"kind": "val", "step": 1, "val": 1.2,
             "per_domain": {"math_": 2.0, "code": 2.2}},
            {"kind": "val", "step": 2, "val": 1.0,
             "per_domain": {"math_": 1.9, "code": 2.1}},
        ]
        report = diagnose_retention(rows)
        self.assertNotIn("DOMAIN_FORGOTTEN",
                         {f["code"] for f in report["findings"]})


class CommandLineExitTests(unittest.TestCase):
    """
    The exit code is the only part of the Doctor an automated caller reads.

    `--json` used to print its report and return, so a critical finding exited
    0 and CI recorded a dead run as healthy. Asserting the status in a unit
    test needs a subprocess or a caught SystemExit, because main() raises
    rather than returns - which is also why nothing caught it before.
    """

    fixture = Path(__file__).parent / "fixtures" / "doctor_selftest_history.jsonl"

    def run_cli(self, *argv):
        out = io.StringIO()
        with patch.object(sys, "argv", ["training_doctor.py", *argv]), \
                redirect_stdout(out):
            with self.assertRaises(SystemExit) as raised:
                training_doctor_cli.main()
        return raised.exception.code, out.getvalue()

    def test_a_critical_finding_exits_non_zero_as_json(self):
        code, printed = self.run_cli("--json", str(self.fixture))
        report = json.loads(printed)
        self.assertEqual(report["health"], "critical")
        self.assertIn("NO_EVIDENCE", {f["code"] for f in report["findings"]})
        self.assertEqual(code, 1)

    def test_a_healthy_history_exits_zero(self):
        # real_rows() alone is not healthy: it carries no per-domain losses, so
        # retention cannot be judged and the doctor says so. A run the doctor
        # must call healthy needs the evidence it asks for.
        rows = real_rows()
        for row in rows:
            if row["kind"] == "val":
                row["per_domain"] = {"stories": 1.4 - row["step"] * 0.2,
                                     "code": 1.5 - row["step"] * 0.2}
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "history.jsonl"
            path.write_text("\n".join(json.dumps(r) for r in rows),
                            encoding="utf-8")
            code, printed = self.run_cli("--json", str(path))
        report = json.loads(printed)
        self.assertEqual(report["health"], "healthy", report["findings"])
        self.assertEqual(code, 0)

    def test_the_committed_fixture_exists(self):
        # CI's Doctor step runs against this file. It used to point at a path
        # under runs/, which is gitignored: nothing committed it and nothing
        # created it, so the step could not have passed on a fresh checkout.
        self.assertTrue(self.fixture.is_file(), self.fixture)
