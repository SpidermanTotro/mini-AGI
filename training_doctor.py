#!/usr/bin/env python3
"""CLI for Greenlight Training Doctor."""
import argparse
import json
from pathlib import Path

from minagi.training_doctor import diagnose_file, resume_preflight


def main():
    p = argparse.ArgumentParser(
        description="Diagnose a Greenlight training run without modifying the "
                    "model: a history, or a checkpoint about to be resumed.")
    p.add_argument("target",
                   help="path to a run history.jsonl, or to a weights "
                        "directory with --preflight")
    p.add_argument("--preflight", action="store_true",
                   help="treat the target as a checkpoint and test whether a "
                        "restart from it can take a training step. Loads the "
                        "model, restores the optimiser, and steps it, on CPU, "
                        "without writing to the directory.")
    p.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    p.add_argument("--out", help="also write the report to this path")
    args = p.parse_args()

    report = (resume_preflight(args.target) if args.preflight
              else diagnose_file(args.target))
    payload = json.dumps(report, indent=2, sort_keys=True)
    if args.out:
        Path(args.out).write_text(payload + "\n", encoding="utf-8")

    if args.json:
        print(payload)
        return

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
    else:
        print("step samples:", o["step_samples"], "| val samples:",
              o["val_samples"])
        print("loss:", o["loss_first"], "->", o["loss_latest"])
        print("grad norm:", o["grad_norm_latest"])
        print("chars/s:", o["chars_per_s_latest"])
    print()
    if not report["findings"]:
        print("No anomaly detected.")
    for f in report["findings"]:
        print(f"[{f['severity'].upper():8}] {f['code']}: {f['message']}")
    raise SystemExit(1 if report["health"] == "critical" else 0)


if __name__ == "__main__":
    main()
