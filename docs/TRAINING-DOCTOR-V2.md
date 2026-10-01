# Training Doctor v2 — Continual Learning Validation

> **Status:** operational reference. Every finding listed here is exercised by
> `tests/test_training_doctor.py`, and the two modes that were silent before
> are covered by tests that fail when their repair is removed.

The Doctor answers questions a loss curve cannot. Loss is measured over the
text a model was trained on; the questions that matter most are about the text
it produces, the state it saved, the pool it routes through, and whether it can
still do any of it after a restart.

Four modes, none of which write to the model or its weights directory.

## Usage

```bash
PY="$HOME/src/mini-AGI/.venv/bin/python"   # never bare python3: see AGENTS.md

# 1. A run's telemetry
"$PY" training_doctor.py runs/agi/history.jsonl

# 2. A checkpoint, before restarting from it
"$PY" training_doctor.py --preflight weights/

# 3. Where the routing mass actually went
"$PY" training_doctor.py --experts runs/expert_history.jsonl

# 4. What the model actually wrote, against the loss falling beside it
"$PY" training_doctor.py --samples runs/samples.txt
```

`--json` emits machine-readable output, `--out PATH` also writes it. Exit code
is 1 when health is `critical`, otherwise 0.

## Mode 1 — history

Reads `history.jsonl` and reports what the run's own record shows.

| Code | Severity | Meaning |
| --- | --- | --- |
| `NONFINITE_LOSS` | critical | NaN or infinite loss appeared |
| `RUN_TRUNCATED` | critical | A `start` record with no `done`: the run died partway |
| `NO_TELEMETRY` | critical | Rows exist but no training telemetry at all |
| `ZERO_GRADIENTS` | critical | Grad norm reached zero |
| `LOSS_SPIKE` | critical | A single loss far above its neighbours |
| `THROUGHPUT_COLLAPSE` | critical | Characters/second fell far below its baseline |
| `INTERMITTENT_ZERO_GRADIENTS` | warning | Gradients reach zero and recover |
| `NO_EVALUATION` | warning | Steps taken, never evaluated |
| `LOSS_IMPROVING` | pass | Training loss materially below its early baseline |
| `LOSS_NOT_IMPROVING` | warning | Loss flat or rising against its early baseline |
| `VAL_IMPROVING` | pass | Held-out loss improved across evaluations |
| `GENERALIZATION_GAP` | warning | Training improved, held-out did not |
| `DOMAIN_FORGOTTEN` | warning | One domain's held-out loss rose materially |
| `DOMAIN_SLIPPING` | warning | One domain drifted upwards |
| `DOMAIN_IMBALANCE` | warning | Worst domain over 3× the best on one model |

**The history schema.** `train.py`'s recorder tags rows `kind`; values are
`start`, `step`, `val`, `saved`, `done`. v1 read a different key and was
therefore blind to every real run. Both spellings are accepted, and the tests
use the recorder's own.

## Mode 2 — preflight

Replays the restore the training loop performs, then asks the one question that
matters: can the optimiser take a step? This is the mode that exists because
the failure it looks for kills the run *before* a telemetry row is written, so
mode 1 has nothing to look at and every loss check passes vacuously.

| Code | Severity | Meaning |
| --- | --- | --- |
| `NO_CHECKPOINT` | critical | No manifest: nothing to resume |
| `MISSING_EXPERT_FILE` | critical | Manifest lists experts that are not on disk |
| `RESTORE_WILL_NOT_LOAD` | critical | Building the model raised, with the error |
| `MISSING_OPTIM_STEP` | critical | Restored moments carry no step counter |
| `RESTART_STEP_FAILED` | critical | Replaying the restore and stepping raised |
| `OPTIM_RESTORED` | pass | Moments and counter both came back |

**It builds the model the way `train.py stream` builds it**, not the way
`build_paged` does. Those are different models from the same directory: the
trainer's pool exposes `pool.experts.N.w1`, which the moment restore matches,
while `build_paged`'s pool holds stacked slot tensors named `pool.w1`, which it
matches none of. Checking the wrong build reports a broken restart as safe.

## Mode 3 — experts

Reads `runs/expert_history.jsonl` and asks where routing mass went.

| Code | Severity | Meaning |
| --- | --- | --- |
| `ROUTING_COLLAPSED` | critical | Ten experts took ≥80% of all routing |
| `GATES_ALL_ZERO` | critical | Every gate is zero; nothing can be routed to |
| `ROUTING_HAS_NO_PREFERENCE` | warning | Routing is almost perfectly even |
| `GATES_NOT_SEPARATING` | warning | Gates within 5% of each other |
| `DEAD_EXPERTS` | warning | ≥25% of the pool never routed to |
| `EXPERT_CHURN` | warning | Admission far exceeds learning per working set |
| `EXPERTS_PRUNED` | warning | Experts removed mid-run |

Concentration is reported as **preference**: 0 is perfectly even routing, 1 is
a single expert taking everything. It is normalised between those two rather
than using a raw Herfindahl, because uniform is `1/n` and a fixed HHI threshold
either fires on everything or nothing depending on pool size.

## Mode 4 — samples

Parses `runs/samples.txt` and compares early-run thirds against late-run
thirds, so one bad sample cannot condemn a run.

| Code | Severity | Meaning |
| --- | --- | --- |
| `LOSS_IMPROVING_OUTPUT_COLLAPSING` | critical | Held-out loss fell and repetition rose >5 points |
| `RAW_OUTPUT_DEGENERATE` | critical | Unguarded output ≥90% repeated 8-grams |
| `LOSS_AND_OUTPUT_DIVERGING` | warning | Loss improved, repetition rose slightly |
| `GENERATION_STILL_REPETITIVE` | warning | ≥35% of guarded 8-grams are repeats |
| `LOW_CHARACTER_ENTROPY` | warning | ≈2.5 bits/char: the shape of a loop |
| `REPETITION_RUNAWAY` | warning | One character repeated ≥40 times |
| `OUTPUT_NOT_WORD_SHAPED` | warning | ≥6 consonants in a row without a vowel |
| `OUTPUT_MOSTLY_NON_ALPHA` | warning | Most tokens contain no letters at all |

**Why the mild case only warns.** A threshold loose enough to catch a 27%
relative rise in repetition is loose enough to fire on noise; one tight enough
to avoid noise will miss a real slide. The mild case reports both numbers and
the direction instead of being promoted to a collapse to make a particular run
look damning.

**Why there is no spell checker.** These metrics are shape-based. The corpus is
code, chess, arithmetic and prose in one run, so any word list flags valid
output. `bfjjk` is also *entirely alphabetic*, so a "does this look like a
word" ratio scores it as perfect prose; the metric that catches it is the
longest consonant run without a vowel, measured across the whole sample
because the nonsense does not respect word boundaries. English has no six in a
row, not even in "strengths".

## What the Doctor does not cover

Stated plainly, because a diagnostic that hides its blind spots is the thing
this work exists to stop:

- **The models themselves.** Every mode reads telemetry or artifacts. Nothing
  here decides whether an architecture is good.
- **Whether held-out text is held out.** Train/eval contamination is not
  checked; it is the next unbuilt check.
- **Domain correctness.** Arithmetic accuracy and chess validity are measured
  as shape, not as right answers.
- **Causal attribution.** A finding tells you where and when, not why.

## Related

- `docs/HISTORY.md` — the integration record and how these checks were built
- `docs/CLI-TRAINING-2026-09-27.md` — training CLI reference
- `AGENTS.md` — which interpreter to use, and why bare `python3` fails here
