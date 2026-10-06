#!/usr/bin/env python3
"""Build the machine-readable result consumed by compare_balance_ab.py."""
import argparse
import json
from pathlib import Path

from minagi.training_doctor import diagnose_experts, read_history, resume_preflight


def build_report(history_path, expert_history_path, weights_dir):
    history = read_history(history_path)
    if not history:
        raise ValueError("training history contains no records")
    last = history[-1]
    heldout = last.get("heldout_loss", last.get("val_loss"))
    if heldout is None:
        raise ValueError("history has no heldout_loss or val_loss")

    experts = diagnose_experts(read_history(expert_history_path))
    obs = experts.get("observations", {})
    n = int(obs.get("experts") or 0)
    total_use = float(obs.get("total_use") or 0.0)

    # Utilization is the fraction of experts with nonzero routing in the last
    # expert-history record. Keep this definition explicit and identical for
    # baseline/candidate.
    erows = read_history(expert_history_path)
    use = erows[-1].get("use") or []
    utilization = (sum(float(v) > 0 for v in use) / len(use)) if use else 0.0

    dropped = last.get("pool_dropped", last.get("capacity_drop"))
    requested = last.get("pool_requested", last.get("capacity_requested"))
    if dropped is None:
        raise ValueError("history has no capacity-drop telemetry")
    if requested is not None and float(requested) > 0:
        capacity_drop = float(dropped) / float(requested)
    else:
        capacity_drop = float(dropped)
    if capacity_drop > 1:
        raise ValueError("capacity drop must be a rate or have requested count")

    resume = resume_preflight(weights_dir)
    return {
        "heldout_loss": float(heldout),
        "expert_utilization": float(utilization),
        "capacity_drop": float(capacity_drop),
        "n_experts": n,
        "resume_ok": resume.get("health") != "critical",
        "doctor_health": resume.get("health"),
        "expert_doctor_health": experts.get("health"),
        "total_use": total_use,
    }


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--history", type=Path, required=True)
    p.add_argument("--expert-history", type=Path, required=True)
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args(argv)
    try:
        report = build_report(args.history, args.expert_history, args.weights)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"report invalid: {exc}")
        return 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
