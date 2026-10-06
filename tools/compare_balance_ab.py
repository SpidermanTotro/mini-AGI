#!/usr/bin/env python3
"""Compare baseline and router-balance experiment reports.

The report schema is intentionally small and matches the Greenlight promotion
contract: heldout_loss, expert_utilization, capacity_drop, n_experts,
resume_ok. This tool never promotes a candidate; it prints the evidence and
returns nonzero when the candidate is incomplete or unsafe.
"""
import argparse
import json
from pathlib import Path


REQUIRED = ("heldout_loss", "expert_utilization", "capacity_drop",
            "n_experts", "resume_ok")


def load_report(path):
    data = json.loads(Path(path).read_text())
    missing = [k for k in REQUIRED if k not in data]
    if missing:
        raise ValueError(f"{path}: missing " + ", ".join(missing))
    if data["resume_ok"] is not True:
        raise ValueError(f"{path}: restart/resume gate failed")
    if float(data["heldout_loss"]) < 0:
        raise ValueError(f"{path}: heldout_loss must be nonnegative")
    if not 0 <= float(data["expert_utilization"]) <= 1:
        raise ValueError(f"{path}: expert_utilization must be in [0, 1]")
    if not 0 <= float(data["capacity_drop"]) <= 1:
        raise ValueError(f"{path}: capacity_drop must be in [0, 1]")
    if int(data["n_experts"]) <= 0:
        raise ValueError(f"{path}: n_experts must be positive")
    return data


def compare(base, candidate):
    return {
        "heldout_loss_delta": float(candidate["heldout_loss"]) -
                              float(base["heldout_loss"]),
        "expert_utilization_delta": float(candidate["expert_utilization"]) -
                                    float(base["expert_utilization"]),
        "capacity_drop_delta": float(candidate["capacity_drop"]) -
                               float(base["capacity_drop"]),
        "expert_count_delta": int(candidate["n_experts"]) -
                              int(base["n_experts"]),
        "baseline_resume_ok": base["resume_ok"],
        "candidate_resume_ok": candidate["resume_ok"],
    }


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("baseline", type=Path)
    p.add_argument("candidate", type=Path)
    p.add_argument("--json", action="store_true")
    args = p.parse_args(argv)
    try:
        result = compare(load_report(args.baseline),
                         load_report(args.candidate))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"A/B invalid: {exc}")
        return 2
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        for key, value in result.items():
            print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
