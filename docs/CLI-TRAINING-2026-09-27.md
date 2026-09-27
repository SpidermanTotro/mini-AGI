# CLI and comparison repairs — 2026-09-27

Starting revision: `811311e`.

- Added `greenlight.py doctor`, `train`, `compare`, `generate`, and `chat`.
- Activated explicit benchmark configuration before model loading; honored
  GREENLIGHT_CONFIG ahead of the compatibility environment variable.
- Added matched prompt and held-out comparisons with a corpus content fingerprint.
  Incompatible protocols are flagged and score deltas withheld.
- Kept baseline checkpoints separate: comparison copies weights and optimizer
  state to a new path and trains only the copy. Existing candidate/output paths
  are rejected. Training/validation overlap is rejected.
- Fixed held-out file support: the trainer previously silently skipped files;
  FolderEvaluator previously assumed a directory.
- Converted Ollama timeouts, connection failures, and non-object JSON into
  readable CLI errors.
- Added regression tests for config activation, baseline preservation, option
  forwarding, data separation, invalid duration, and protocol mismatch handling.

Validation: all 100 tests passed on CPU, including the new regression tests.
A tiny 16-wide, four-expert model completed actual training, save/reload,
generation, and a baseline/candidate training-and-evaluation cycle on CPU.
These are functional tests, not representative quality or performance benchmarks.
No user checkpoint or desktop GPU was available in this environment; the full
16 GB profile and real R1/R2 quality comparison must run on the user's hardware.

Remaining project work: representative datasets, longer training, repeated matched
runs, and review of generated answers. Ollama and Greenlight are separate model
backends. This change does not train Ollama models or compare historical code
revisions. The minute budget bounds the training loop only. Models are experimental.
