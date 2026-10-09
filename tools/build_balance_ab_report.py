#!/usr/bin/env python3
"""Build an auditable router A/B report, without running or modifying training.

An in-process optimizer preflight is NOT independent cold resume evidence.
Evidence is required from a separately executed training restart and evaluation.
The output follows tools/compare_balance_ab.py on current Greenlight main.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from minagi.training_doctor import diagnose_experts, read_history, resume_preflight


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint(path) -> str:
    """Read-only SHA-256 for a file or sorted directory file manifest.

    Directory hashing includes relative names and file content hashes.
    Symlinks are rejected so traversing a checkpoint cannot escape its root.
    """
    root = Path(path)
    if root.is_symlink() or not root.exists():
        raise ValueError(f"missing or symlinked source: {root}")
    if root.is_file():
        return _sha256_file(root)
    if not root.is_dir():
        raise ValueError(f"not a regular file or directory: {root}")
    entries = []
    for item in root.rglob("*"):
        if item.is_symlink():
            raise ValueError(f"symlink inside checkpoint/corpus: {item}")
        if item.is_file():
            entries.append(item)
    if not entries:
        raise ValueError(f"source directory contains no files: {root}")
    aggregate = hashlib.sha256()
    for item in sorted(entries, key=lambda p: p.relative_to(root).as_posix()):
        relative = item.relative_to(root).as_posix()
        aggregate.update(relative.encode("utf-8") + b"\x00")
        aggregate.update(_sha256_file(item).encode("ascii") + b"\n")
    return aggregate.hexdigest()


def _finite(value, key, *, lower=0, upper=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{key} must be numeric")
    number = float(value)
    if not math.isfinite(number) or number < lower or (upper is not None and number > upper):
        raise ValueError(f"{key} is out of bounds or nonfinite")
    return number


def cold_resume_proof(path, weights_fingerprint) -> dict:
    """Validate the recorded outputs of a real separate-process restart.

    This can validate a provided log and metadata, not authenticate that a
    training process occurred. Retain raw process logs for manual review.
    """
    proof_path = Path(path)
    info = json.loads(proof_path.read_text(encoding="utf-8"))
    if not isinstance(info, dict):
        raise ValueError("cold resume proof must be a JSON object")
    if info.get("new_process") is not True or type(info.get("exit_code")) is not int or info["exit_code"] != 0:
        raise ValueError("cold resume requires an exited successful fresh process")
    for field in ("optimizer_restored", "heldout_eval_after_resume"):
        if info.get(field) is not True:
            raise ValueError(f"cold resume did not verify {field}")
    before, after = info.get("step_before"), info.get("step_after")
    if (type(before) is not int or type(after) is not int or
            before < 0 or after <= before):
        raise ValueError("cold resume requires advanced optimizer steps")
    if info.get("source_weights_sha256") != weights_fingerprint:
        raise ValueError("resumed checkpoint fingerprint mismatches evidence")
    log_path = Path(info.get("log_path", ""))
    if not log_path.is_absolute():
        log_path = proof_path.parent / log_path
    if not log_path.is_file() or log_path.is_symlink():
        raise ValueError("cold resume process log is absent or unsafe")
    if info.get("log_sha256") != _sha256_file(log_path):
        raise ValueError("cold resume process log hash mismatch")
    return {"new_process": True, "exit_code": 0,
            "step_before": before, "step_after": after,
            "log_sha256": info["log_sha256"]}


def build_report(history_path, expert_history_path, weights_dir, *,
                 initial_checkpoint, training_corpus, heldout_corpus, seed,
                 training_steps, balance_strength, cold_resume_evidence):
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if type(training_steps) is not int or training_steps < 1:
        raise ValueError("training_steps must be a positive integer")
    coefficient = _finite(balance_strength, "balance_strength")

    start_hash = fingerprint(initial_checkpoint)
    train_hash = fingerprint(training_corpus)
    heldout_hash = fingerprint(heldout_corpus)
    if train_hash == heldout_hash:
        raise ValueError("training and held-out corpus have identical fingerprints")
    resumed_weights_hash = fingerprint(weights_dir)

    rows = read_history(history_path)
    heldout = [r for r in rows
               if r.get("kind", r.get("event")) == "val" and "val" in r]
    if not heldout:
        raise ValueError("no held-out evaluation record (kind=val, val=...)")
    last_val = heldout[-1]
    heldout_loss = _finite(last_val["val"], "heldout loss")

    # The streaming trainer can record validation + routing in the same
    # history JSONL; skip unrelated step/save events, never fake use data.
    expert_rows = [r for r in read_history(expert_history_path)
                   if isinstance(r.get("use"), list) and r["use"]]
    if not expert_rows:
        raise ValueError("expert history contains no real routing usage rows")
    recent = expert_rows[-1]
    use = recent["use"]
    utilization = sum(_finite(x, "expert use") > 0 for x in use) / len(use)
    doctors = diagnose_experts(expert_rows)
    n = int(doctors.get("observations", {}).get("experts") or len(use))
    if n < 1 or n != len(use):
        raise ValueError("expert count and routing telemetry disagree")

    # Capacity numbers MUST come from an actual measurement; no default 0.
    capacity_rows = [r for r in rows
                     if ("pool_dropped" in r or "capacity_drop" in r)]
    if not capacity_rows:
        raise ValueError("no measured capacity-drop telemetry")
    latest = capacity_rows[-1]
    if "pool_dropped" in latest:
        dropped = _finite(latest["pool_dropped"], "pool_dropped")
        requested = _finite(latest.get("pool_requested"), "pool_requested")
        if requested <= 0 or dropped > requested:
            raise ValueError("invalid drop/request counts")
        capacity_drop = dropped / requested
    else:
        capacity_drop = _finite(latest["capacity_drop"], "capacity_drop", upper=1)

    doctor = resume_preflight(weights_dir)
    preflight_passed = doctor.get("health") != "critical"
    if not preflight_passed:
        raise ValueError("checkpoint optimizer preflight is critical")
    proof = cold_resume_proof(cold_resume_evidence, resumed_weights_hash)

    return {
        "heldout_loss": heldout_loss,
        "expert_utilization": utilization,
        "capacity_drop": capacity_drop,
        "n_experts": n,
        "resume_ok": True,
        "balance_strength": coefficient,
        "initial_checkpoint_sha256": start_hash,
        "training_corpus_sha256": train_hash,
        "heldout_corpus_sha256": heldout_hash,
        "seed": seed,
        "training_steps": training_steps,
        "preflight_health": doctor.get("health"),
        "cold_resume_evidence": proof,
        "expert_doctor_health": doctors.get("health"),
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--history", type=Path, required=True)
    p.add_argument("--expert-history", type=Path, required=True)
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--initial-checkpoint", type=Path, required=True)
    p.add_argument("--training-corpus", type=Path, required=True)
    p.add_argument("--heldout-corpus", type=Path, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--training-steps", type=int, required=True)
    p.add_argument("--balance-strength", type=float, required=True)
    p.add_argument("--cold-resume-evidence", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args(argv)
    try:
        report = build_report(
            args.history, args.expert_history, args.weights,
            initial_checkpoint=args.initial_checkpoint,
            training_corpus=args.training_corpus,
            heldout_corpus=args.heldout_corpus, seed=args.seed,
            training_steps=args.training_steps,
            balance_strength=args.balance_strength,
            cold_resume_evidence=args.cold_resume_evidence,
        )
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"report rejected (no promotion): {exc}")
        return 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        print(f"report rejected: refusing to overwrite {args.output}")
        return 2
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
