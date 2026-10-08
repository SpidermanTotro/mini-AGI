# Greenlight: deterministic expert-balance A/B promotion contract

This document applies to the **experimental** `experiment/upstream-expert-balance`
branch. It does not authorize enabling `pool.balance` on `main`.

## Controlled experiment

Create two training runs from **the same immutable initial checkpoint**,
training corpus, held-out corpus, seed, training-step budget, config, hardware,
and measurement protocol. Change **only** the expert-balance coefficient:

- Baseline: `balance_strength = 0.0`
- Candidate: `balance_strength = 0.001` (initial test value)

Use separate output directories and checkpoints; do not overwrite the starting
checkpoint. Measure actual held-out loss, utilization, capacity-drop rate,
expert count, and cold restart/resume success for **both** runs. Save the hash of
the *starting* checkpoint and the training and held-out corpus. For a paged
checkpoint directory, fingerprint a deterministic manifest of all relevant
files, not merely one arbitrary expert file. Hash the same canonical manifest
in both runs. Ensure training/validation data do not overlap.

Repeat the controlled comparison across multiple predeclared seeds before
arguing for promotion. The script below checks **one comparable pair**; it
cannot by itself establish statistical significance or general capabilities.

## Required JSON

Both `baseline.json` and `candidate.json` must have these fields.
The example demonstrates the **schema only**, not a real experiment:

```json
{
  "heldout_loss": 0.65,
  "expert_utilization": 0.60,
  "capacity_drop": 0.01,
  "n_experts": 64,
  "resume_ok": true,
  "balance_strength": 0.0,
  "initial_checkpoint_sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
  "training_corpus_sha256": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
  "heldout_corpus_sha256": "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
  "seed": 42,
  "training_steps": 100,
  "chars_per_second": 250.0,
  "peak_vram_gb": 8.0
}
```

Replace the example numbers/hashes with actual measured data. All three
fingerprints must be lowercase 64-character SHA-256 values. In the candidate
report, use a positive `balance_strength`. `chars_per_second` and
`peak_vram_gb` are optional diagnostics; report them on both runs if measured.

`expert_utilization` and `capacity_drop` must be fractions between 0 and 1,
measured using **identical definitions** and windows for both runs.

## Compare and inspect

```bash
python tools/compare_balance_ab.py baseline.json candidate.json
```

Defaults are deliberately conservative. Promotion is denied if:

- a required field is missing, invalid or nonfinite;
- checkpoint/corpus/seed/step provenance differs;
- baseline balance is not zero or candidate balance is not positive;
- either run fails its restart/resume test;
- held-out loss regresses, utilization fails to improve, capacity-drop rate
  increases, or the candidate grows more experts than the baseline.

A predeclared experimental tolerance can be passed explicitly:

```bash
python tools/compare_balance_ab.py baseline.json candidate.json \
  --max-loss-regression 0.0 \
  --max-capacity-drop-regression 0.0 \
  --max-expert-growth 0
```

Nonzero tolerances require scientific justification; do not adjust them
after inspecting results to manufacture a pass. The tool records optional
throughput/VRAM deltas as diagnostics, not automatic promotion criteria.
A green result means only that the provided reports clear this narrow gate;
it is **not** a substitute for benchmark provenance, independent reproducibility,
GPU stability, long-run training, or human review.
