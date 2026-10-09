# Greenlight router-balance evaluation runbook (draft)

This runbook belongs to experimental PR #25. It **does not** enable
router balancing on `main`, launch model training, or assert that either
candidate improves performance. PR #33 remains a competing design.

## Preflight and safety

1. Back up the original checkpoint and keep that directory immutable.
   Never run baseline and candidate in the same output directory.
2. Record Git commit SHA, installed Python/Torch/CUDA, NVIDIA driver,
   GPU/VRAM, corpus provenance and random seed.
3. Keep training and held-out corpora separate. The report rejects identical
   corpus fingerprints, but that alone cannot rule out partial overlap.
4. Verify the real training command reads `GREENLIGHT_CONFIG`. Preparing
   profiles does **not** launch the training command.
5. This is a *no-exploration* two-arm A/B: `pool.explore_bias=0` in **both**
   arms, `pool.balance=0` versus `0.001`. Only balance changes. It is
   **not** a comparison against R5's default exploration bonus. R5-default
   performance belongs in a third, separately declared reference run.
6. Use an identical initial checkpoint, corpus, seed, training budget,
   model/optimizer hyperparameters, hardware and evaluation definition.

## Generate profiles without launching training

```bash
python tools/prepare_balance_ab.py \
  --config config.yaml \
  --out runs/balance-ab/seed-42/profiles \
  --command 'python train.py stream --weights-dir {weights_dir} --out {run_dir}'
```

This saves separate YAML profiles and a manifest with **different** per-arm
`--weights-dir` and `--out` paths and refuses to overwrite existing outputs.
The tool **does not create or seed checkpoint directories**. Manually clone
the same immutable starting checkpoint into each displayed `weights_dir` before
launching either command; otherwise the two runs would not have equal starts.
Use the actual `train.py stream` flags and input mix supported by your checkout.
Never execute a generated command until the starting checkpoint and corpus
arguments have been independently checked.

## Independent cold restart proof

Training Doctor's preflight rebuilds an optimizer in memory and takes a
diagnostic CPU step. It is **not** a substitute for a *fresh process* restart:

1. Save a checkpoint after some training steps.
2. Exit the trainer process and start a **new process** pointing at the saved
   checkpoint; preserve the original checkpoint as an immutable input.
3. Confirm restored AdamW moments/counter, take further optimization steps,
   evaluate on the held-out set, and preserve the raw stdout/stderr log.
4. Fingerprint the exact restart-source checkpoint directory and output log,
   then record the following separately for baseline and candidate:

```json
{
  "new_process": true,
  "exit_code": 0,
  "optimizer_restored": true,
  "heldout_eval_after_resume": true,
  "step_before": 100,
  "step_after": 101,
  "source_weights_sha256": "64_lowercase_hex_from_checkpoint_fingerprint",
  "log_path": "/absolute/path/to/restart.log",
  "log_sha256": "64_lowercase_hex_of_log_contents"
}
```

Metadata does not prove the run happened; reviewers still need the original
logs. The `source_weights_sha256` here fingerprints the checkpoint **used
for the cold restart**. The report's `initial_checkpoint_sha256` fingerprints
the **shared initial checkpoint** used to start both A/B arms.

## Build auditable reports (read-only inputs)

The reporter requires actual `kind=val` rows with a `val` held-out loss,
expert routing usage history, and measured capacity-drop telemetry. It refuses
to invent missing metrics or mistake training loss for held-out loss.

```bash
python tools/build_balance_ab_report.py \
  --history runs/baseline/history.jsonl \
  --expert-history runs/baseline/expert_history.jsonl \
  --weights runs/baseline/restart-checkpoint \
  --initial-checkpoint checkpoints/frozen-start \
  --training-corpus data/train.txt \
  --heldout-corpus data/heldout.txt \
  --seed 42 --training-steps 100 --balance-strength 0.0 \
  --cold-resume-evidence runs/baseline/cold-resume.json \
  --output runs/baseline/report.json
```

Repeat with *candidate* paths and `--balance-strength 0.001`. The tool
hashes checkpoint and corpora read-only, and refuses to overwrite old reports.

Compare using the strict comparator now on Greenlight `main`:

```bash
python tools/compare_balance_ab.py \
  runs/baseline/report.json runs/candidate/report.json
```

Never put synthetic fixture numbers in real reports. Repeat predeclared seeds,
measure expert starvation, held-out loss, capacity, expert growth, throughput,
memory cost and actual restart integrity. A passing report gate is not AGI/ASI.

## Missing instrumentation is a blocker, not zero drop

If the trainer does not log `pool_dropped` and `pool_requested` (or an actually
measured `capacity_drop` rate), the report refuses promotion. Instrument
both arms with matching definitions/windows first. Never supply invented
capacity telemetry just to pass this gate.
