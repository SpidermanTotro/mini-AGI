#!/usr/bin/env python3
"""Fail-closed promotion gate for deterministic expert-balance A/B results.

Both JSON reports must identify the *same initial checkpoint*, training corpus,
held-out corpus, seed, and number of training steps. Only balance_strength should
differ (baseline zero, candidate positive). Reports without provenance are not
sufficient evidence to promote a model.

Required JSON fields:
  heldout_loss, expert_utilization, capacity_drop, n_experts, resume_ok,
  initial_checkpoint_sha256, training_corpus_sha256, heldout_corpus_sha256,
  seed, training_steps, balance_strength.

Hashes are 64 lowercase hex SHA-256 characters. Optional chars_per_second and
peak_vram_gb are reported as cost diagnostics when supplied in both files.
"""
import argparse
import json
import math
import re
import sys

MEASUREMENTS = ("heldout_loss", "expert_utilization", "capacity_drop",
                "n_experts", "resume_ok", "balance_strength")
PROVENANCE = ("initial_checkpoint_sha256", "training_corpus_sha256",
              "heldout_corpus_sha256", "seed", "training_steps")
REQUIRED = MEASUREMENTS + PROVENANCE
HASHES = PROVENANCE[:3]


def validate_result(x, label="result"):
    if not isinstance(x, dict):
        raise ValueError(f"{label}: expected JSON object")
    missing = [k for k in REQUIRED if k not in x]
    if missing:
        raise ValueError(f"{label}: missing {', '.join(missing)}")
    for key in ("heldout_loss", "expert_utilization", "capacity_drop",
                "balance_strength"):
        v = x[key]
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
            raise ValueError(f"{label}: {key} must be a finite number")
    if x["heldout_loss"] < 0:
        raise ValueError(f"{label}: heldout_loss must be nonnegative")
    for key in ("expert_utilization", "capacity_drop"):
        if not 0 <= x[key] <= 1:
            raise ValueError(f"{label}: {key} must be in [0, 1]")
    if x["balance_strength"] < 0:
        raise ValueError(f"{label}: balance_strength must be nonnegative")
    for key in ("n_experts", "seed", "training_steps"):
        v = x[key]
        if isinstance(v, bool) or not isinstance(v, int):
            raise ValueError(f"{label}: {key} must be an integer")
    if x["n_experts"] < 1 or x["seed"] < 0 or x["training_steps"] < 1:
        raise ValueError(f"{label}: invalid expert count, seed, or training steps")
    if type(x["resume_ok"]) is not bool:
        raise ValueError(f"{label}: resume_ok must be a boolean")
    for key in HASHES:
        if not isinstance(x[key], str) or re.fullmatch(r"[0-9a-f]{64}", x[key]) is None:
            raise ValueError(f"{label}: {key} must be a lowercase SHA-256 hex digest")
    for key in ("chars_per_second", "peak_vram_gb"):
        if key in x:
            v = x[key]
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0:
                raise ValueError(f"{label}: {key} must be finite and nonnegative")
    return x


def load(path):
    with open(path, encoding="utf-8") as f:
        return validate_result(json.load(f), str(path))


def compare(base, candidate, max_loss_regression=0.0,
            max_capacity_drop_regression=0.0, max_expert_growth=0):
    try:
        validate_result(base, "baseline")
        validate_result(candidate, "candidate")
        for key, value in (("max_loss_regression", max_loss_regression),
                           ("max_capacity_drop_regression", max_capacity_drop_regression)):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{key} must be finite and nonnegative")
        if isinstance(max_expert_growth, bool) or not isinstance(max_expert_growth, int) or max_expert_growth < 0:
            raise ValueError("max_expert_growth must be a nonnegative integer")
    except ValueError as exc:
        return False, str(exc)

    different = [key for key in PROVENANCE if base[key] != candidate[key]]
    if different:
        return False, "A/B provenance mismatch: " + ", ".join(different)
    if base["balance_strength"] != 0 or candidate["balance_strength"] <= 0:
        return False, "expected baseline balance_strength=0 and candidate balance_strength>0"
    if not base["resume_ok"] or not candidate["resume_ok"]:
        return False, "restart/resume gate failed"

    loss_delta = candidate["heldout_loss"] - base["heldout_loss"]
    util_delta = candidate["expert_utilization"] - base["expert_utilization"]
    drop_delta = candidate["capacity_drop"] - base["capacity_drop"]
    expert_growth = candidate["n_experts"] - base["n_experts"]

    ok = (loss_delta <= max_loss_regression
          and util_delta > 0
          and drop_delta <= max_capacity_drop_regression
          and expert_growth <= max_expert_growth)
    reason = (f"heldout_delta={loss_delta:+.6f} "
              f"utilization_delta={util_delta:+.6f} "
              f"capacity_drop_delta={drop_delta:+.6f} "
              f"expert_growth={expert_growth:+d}")
    for key in ("chars_per_second", "peak_vram_gb"):
        if key in base and key in candidate:
            reason += f" {key}_delta={candidate[key] - base[key]:+.6f}"
        else:
            reason += f" {key}=not_measured"
    if not ok:
        reason += " ; candidate does not clear promotion gate"
    return ok, reason


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("baseline")
    p.add_argument("candidate")
    p.add_argument("--max-loss-regression", type=float, default=0.0)
    p.add_argument("--max-capacity-drop-regression", type=float, default=0.0)
    p.add_argument("--max-expert-growth", type=int, default=0)
    a = p.parse_args()
    try:
        baseline, candidate = load(a.baseline), load(a.candidate)
        ok, reason = compare(
            baseline, candidate, a.max_loss_regression,
            a.max_capacity_drop_regression, a.max_expert_growth)
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        ok, reason = False, str(exc)
    print(json.dumps({"promote": ok, "reason": reason}, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
