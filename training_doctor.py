#!/usr/bin/env python3
"""CLI for Greenlight Training Doctor."""
import argparse
import json
from pathlib import Path

from minagi.training_doctor import diagnose_file


def main():
    p = argparse.ArgumentParser(
        description="Diagnose a Greenlight training history without modifying the model.")
    p.add_argument("history", help="path to a run history.jsonl")
    p.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    p.add_argument("--out", help="also write the report to this path")
    args = p.parse_args()

    report = diagnose_file(args.history)
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
    print("step samples:", o["step_samples"], "| val samples:", o["val_samples"])
    print("loss:", o["loss_first"], "->", o["loss_latest"])
    print("grad norm:", o["grad_norm_latest"])
    print("chars/s:", o["chars_per_s_latest"])
    print()
    if not report["findings"]:
        print("No anomaly detected in the available telemetry.")
    for f in report["findings"]:
        print(f"[{f['severity'].upper():8}] {f['code']}: {f['message']}")


if __name__ == "__main__":
    main()
