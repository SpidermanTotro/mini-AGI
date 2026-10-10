# Greenlight Next — Pre-migration v0.1

Status: **development only**. This branch starts from `main` after checkpoint safety PR #37 merged; it is not a new trained model and is not promoted for production.

## Scope

- Preserve all existing Space Bunny, Fledge, and Greenlight weights and checkpoints. No model-weight merge.
- Bring forward checkpoint recovery only after the AdamW preflight and Training Doctor regressions pass on Python 3.11 and 3.14.
- Add local-first assistant components behind opt-in interfaces: conversation storage, retrieval, permission-gated tools, and evaluation.
- Keep model inference separate from assistant orchestration. No cloud requirement or hidden external calls.

## Migration gates

1. **Baseline:** record main commit, Python/PyTorch/CUDA versions, hardware, checkpoint manifest, and hashes. Never write to the source checkpoint.
2. **Restore:** checkpoint -> fresh model and optimizer -> load -> training step. Report missing counters as a clear failure, never fabricate an unknown optimizer age.
3. **Compatibility:** run existing training, serving, GradSNR, and Training Doctor tests on Python 3.11 and 3.14.
4. **Assistant:** test opt-in local history, memory deletion, and denied-by-default tool writes; preserve existing CLI behavior.
5. **Evaluation:** measure generation, coding, reasoning, retention, restart, and VRAM. Record baselines before claiming improvement.
6. **Promotion:** require green CI and a documented manual GPU resume run before declaring the new version stable.

## Work board

- [x] Create isolated Greenlight Next branch from main.
- [x] Merge validated AdamW checkpoint recovery (PR #37; Python 3.11 and 3.14 CI passed).
- [ ] Add end-to-end checkpoint restart and negative-case tests.
- [ ] Implement optional local conversation store and tests.
- [ ] Implement explicit-permission tool runner and tests.
- [ ] Produce repeatable benchmark report.
- [ ] Tag a version only after all release gates pass.

## Rollback

Keep `main` and all checkpoint directories untouched during development. Any migration creates new paths; roll back by selecting the original branch and original checkpoint. Never overwrite an existing weights directory to test a migration.
