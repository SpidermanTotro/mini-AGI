#!/usr/bin/env python3
"""CLI for Greenlight Training Doctor."""
import argparse
import json
from pathlib import Path

from minagi.training_doctor import (diagnose, diagnose_experts,
                                      diagnose_file, diagnose_generation,
                                      diagnose_retention, parse_samples,
                                      read_history, resume_preflight)


def main():
    p = argparse.ArgumentParser(
        description="Diagnose a Greenlight training run without modifying the "
                    "model: a history, a checkpoint about to be resumed, or a "
                    "run's expert-routing history.")
    p.add_argument("target",
                   help="path to a run history.jsonl, an expert "
                        "history.jsonl, or a weights directory with "
                        "--preflight")
    p.add_argument("--preflight", action="store_true",
                   help="treat the target as a checkpoint and test whether a "
                        "restart from it can take a training step. Loads the "
                        "model, restores the optimiser, and steps it, on CPU, "
                        "without writing to the directory.")
    p.add_argument("--experts", action="store_true",
                   help="read the target as a per-expert routing history "
                        "(runs/expert_history.jsonl) and report where routing "
                        "mass actually went")
    p.add_argument("--samples", action="store_true",
                   help="read the target as a samples log (runs/samples.txt) "
                        "and measure what the model actually wrote against "
                        "the held-out loss falling beside it")
    p.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    p.add_argument("--out", help="also write the report to this path")
    args = p.parse_args()

    # Only the two history-shaped modes read JSONL; samples.txt and a weights
    # directory are not JSONL and must not be parsed as if they were.
    if args.preflight:
        report = resume_preflight(args.target)
    elif args.samples:
        report = diagnose_generation(parse_samples(args.target))
    elif args.experts:
        report = diagnose_experts(read_history(args.target))
    else:
        rows = read_history(args.target)
        report = diagnose(rows)
        retention = diagnose_retention(rows)
        for finding in retention["findings"]:
            report["findings"].append(finding)
        if retention["health"] == "critical":
            report["health"] = "critical"
        elif retention["health"] == "warning" and report["health"] == "healthy":
            report["health"] = "warning"
        report["observations"]["retention"] = retention["observations"]
    payload = json.dumps(report, indent=2, sort_keys=True)
    if args.out:
        Path(args.out).write_text(payload + "\n", encoding="utf-8")

    if args.json:
        # The exit code is the whole point for an automated caller, and it
        # used to be dropped here: `--json` returned before the status was set,
        # so a critical finding exited 0 and CI read a dead run as healthy.
        print(payload)
        raise SystemExit(1 if report["health"] == "critical" else 0)

    print("GREENLIGHT TRAINING DOCTOR")
    print("=" * 48)
    print("health:", report["health"].upper())
    o = report["observations"]
    if args.preflight:
        print("weights:", o.get("weights_dir"), "| step on disk:",
              o.get("step_on_disk"), "| experts:", o.get("experts"))
        print("restored moments:", o.get("restored_moments"),
              "| without a step counter:", o.get("restored_without_step"))
        print("first step after restore:", o.get("first_step_raised") or "ok")
    elif args.samples:
        print("blocks:", o.get("blocks"), "| steps:", o.get("steps"))
        print("held-out loss:", o.get("held_out_early"), "->",
              o.get("held_out_late"))
        print("repeated 8-grams (guarded):", o.get("repeated_early"), "->",
              o.get("repeated_late"),
              "| raw:", o.get("raw_repeated_late"))
        print("worst domain:", o.get("worst_domain"))
    elif args.experts:
        print("experts:", o.get("experts"), "| characters:", o.get("chars"))
        print("routing preference:", o.get("preference"),
              "(0 = perfectly even, 1 = one expert)")
        print("top-10 use share:", o.get("top10_use_share"),
              "| never used:", o.get("never_used"),
              "| admissions:", o.get("admits_total"))
    else:
        print("step samples:", o["step_samples"], "| val samples:",
              o["val_samples"])
        print("loss:", o["loss_first"], "->", o["loss_latest"])
        print("grad norm:", o["grad_norm_latest"])
        print("chars/s:", o["chars_per_s_latest"])
        ret = o.get("retention") or {}
        if ret.get("domains"):
            print("domains tracked:", ret.get("tracked"), "of",
                  ret.get("domains"), "| latest:", ret.get("per_domain_latest"))
    print()
    if not report["findings"]:
        print("No anomaly detected.")
    for f in report["findings"]:
        print(f"[{f['severity'].upper():8}] {f['code']}: {f['message']}")
    raise SystemExit(1 if report["health"] == "critical" else 0)


if __name__ == "__main__":
    main()
