# Changelog

This file records Greenlight Recur changes in a compact, operational form.
For provenance and the longer engineering narrative, see `docs/HISTORY.md`.

Status labels:
- **Stable** — merged to `main` and passed the repository CI gate.
- **Experimental** — isolated from `main`; results are not model claims.
- **Superseded failure** — an older development revision that failed CI and
  was replaced by a later commit. GitHub keeps these historical runs visible.

## 2026-10-06

### Stable — checkpoint parameter accounting

Merged through **PR #22** as `1022a5d`.

- Model artifact verification now counts core, router, expert, total and
  resident parameters from checkpoint files.
- Expert manifest entries must contain valid positive parameter inventories.
- Verification uses NPZ data without pickle loading.
- Python 3.11 and 3.14 CI passed before merge.

### Stable — router-balance promotion contract

Merged through **PR #23** as `853222a`.

- Added an executable evidence contract for upstream router-balance
  experiments.
- A candidate report must contain held-out loss, expert utilization,
  capacity-drop rate, expert count and restart/resume status.
- Missing behavioral evidence or failed resume blocks promotion.
- This change added a gate only; it did not change training behavior.
- Python 3.11 and 3.14 CI passed before merge.

### Stable — unified Greenlight Doctor

Merged through **PR #24** as `b983b5d`.

- `greenlight.py doctor` can orchestrate runtime/config checks, checkpoint
  verification, real restore + AdamW resume preflight, training-history
  diagnosis, expert-routing diagnosis and generation diagnosis.
- Critical deep findings produce a nonzero status suitable for automation.
- Added CLI regression coverage for the orchestration path.
- Python 3.11 and 3.14 CI passed before merge.

### Experimental — isolated router balance candidate

**PR #25** is intentionally not part of stable `main`.

Candidate goals:
- evaluate an upstream-style differentiable router-balance loss;
- keep stochastic selection disabled for the first A/B;
- keep balance disabled by default;
- prevent legacy `explore_bias` and balance loss from running together;
- detach trunk activations so the balance auxiliary signal targets router rows;
- feed the balance term into live and streaming training paths;
- generate machine-readable A/B evidence and compare baseline vs candidate.

Promotion requires the PR #23 contract plus successful repository CI and a
controlled GPU A/B run. Unit tests alone are not sufficient evidence for a
training-math change.

#### Historical CI failures on PR #25

Early PR #25 revisions produced red Actions runs. These are retained by GitHub
and are **not separate current regressions**.

- `16f86eb`: new router-gradient test failed because its fixture began with
  experts resident and therefore bypassed the first-admission path where the
  balance term is computed.
- The fixture was corrected to force first admission rather than weakening the
  assertion.
- Later pushes added A/B tooling and therefore started additional CI runs.
  Runs belonging to older SHAs are superseded when a newer complete candidate
  is pushed.

When judging PR #25, use the CI result for its current head SHA rather than
counting historical red runs.

## Earlier history

Greenlight Recur's September work, upstream ancestry, independence transition,
October 1 upstream integration, discovered restart defect and Training Doctor
v2 are documented in `docs/HISTORY.md`.
