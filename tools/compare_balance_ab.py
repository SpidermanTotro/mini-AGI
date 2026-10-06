#!/usr/bin/env python3
"""Compare baseline and deterministic expert-balance A/B result JSON files.

This is intentionally a promotion gate, not a claim that balance is better.
Both runs must come from the same checkpoint/corpus/seed protocol.
"""
import argparse, json, math, sys

REQUIRED = ("heldout_loss", "expert_utilization", "capacity_drop",
            "n_experts", "resume_ok")

def load(path):
    with open(path, encoding="utf-8") as f:
        x = json.load(f)
    missing = [k for k in REQUIRED if k not in x]
    if missing:
        raise ValueError(f"{path}: missing {', '.join(missing)}")
    for k in ("heldout_loss", "expert_utilization", "capacity_drop"):
        if not math.isfinite(float(x[k])):
            raise ValueError(f"{path}: {k} must be finite")
    return x

def compare(base, candidate, max_loss_regression):
    if base["resume_ok"] is not True or candidate["resume_ok"] is not True:
        return False, "restart/resume gate failed"
    loss_delta = float(candidate["heldout_loss"]) - float(base["heldout_loss"])
    util_delta = float(candidate["expert_utilization"]) - float(base["expert_utilization"])
    drop_delta = float(candidate["capacity_drop"]) - float(base["capacity_drop"])
    ok = loss_delta <= max_loss_regression and util_delta > 0
    reason = (
        f"heldout_delta={loss_delta:+.6f} "
        f"utilization_delta={util_delta:+.6f} "
        f"capacity_drop_delta={drop_delta:+.6f}"
    )
    if not ok:
        reason += " ; candidate does not clear promotion gate"
    return ok, reason

def main():
    p = argparse.ArgumentParser()
    p.add_argument("baseline")
    p.add_argument("candidate")
    p.add_argument("--max-loss-regression", type=float, default=0.0,
                   help="maximum allowed candidate minus baseline held-out loss")
    a = p.parse_args()
    base, cand = load(a.baseline), load(a.candidate)
    ok, reason = compare(base, cand, a.max_loss_regression)
    print(json.dumps({"promote": ok, "reason": reason}, indent=2))
    return 0 if ok else 1

if __name__ == "__main__":
    sys.exit(main())
