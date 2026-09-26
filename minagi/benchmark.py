"""Reproducible benchmark metadata and comparison helpers for mini-AGI."""

import argparse
import hashlib
import json
import os
import platform
import subprocess
import time
from pathlib import Path

import torch

from .evaluate import evaluate_cases
from .precision import set_compute_dtype
from .recur import load_any


def _git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _config_info(path):
    if not path:
        return {"path": None, "sha256": None}
    p = Path(path)
    if not p.exists():
        return {"path": str(p), "sha256": None}
    return {
        "path": str(p),
        "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
    }


def activate_config(path):
    """Make an explicit benchmark config authoritative for downstream loaders."""
    if path:
        os.environ["MINI_AGI_CONFIG"] = str(Path(path).resolve())
    return os.environ.get("MINI_AGI_CONFIG")


def system_metadata(device):
    info = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": str(device),
        "cuda": torch.version.cuda,
        "gpu": None,
        "vram_total_bytes": None,
    }
    if device.type == "cuda" and torch.cuda.is_available():
        props = torch.cuda.get_device_properties(device)
        info["gpu"] = props.name
        info["vram_total_bytes"] = props.total_memory
    return info


def summarize_results(results):
    scored = [r for r in results if "exact_match" in r]
    return {
        "cases": len(results),
        "scored_cases": len(scored),
        "exact_matches": sum(bool(r["exact_match"]) for r in scored),
        "exact_match_accuracy": (
            sum(bool(r["exact_match"]) for r in scored) / len(scored)
            if scored else None
        ),
    }


def compare_reports(base, candidate):
    b = base.get("summary", {})
    c = candidate.get("summary", {})
    return {
        "base_commit": base.get("provenance", {}).get("commit"),
        "candidate_commit": candidate.get("provenance", {}).get("commit"),
        "exact_match_accuracy_delta": (
            c.get("exact_match_accuracy") - b.get("exact_match_accuracy")
            if c.get("exact_match_accuracy") is not None
            and b.get("exact_match_accuracy") is not None
            else None
        ),
        "elapsed_seconds_delta": candidate.get("timing", {}).get("elapsed_seconds", 0)
        - base.get("timing", {}).get("elapsed_seconds", 0),
    }


def main():
    parser = argparse.ArgumentParser(description="Run a reproducible mini-AGI benchmark")
    parser.add_argument("--weights", default="agi-16-large")
    parser.add_argument("--config", default=os.environ.get("MINI_AGI_CONFIG"))
    parser.add_argument("--output", default="runs/benchmark.json")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--precision", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--compare")
    args = parser.parse_args()

    if args.max_new_tokens <= 0:
        parser.error("--max-new-tokens must be positive")

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.reset_peak_memory_stats(device)

    set_compute_dtype(args.precision)
    model, metadata = load_any(args.weights, device, read_only=True)

    started = time.perf_counter()
    results = evaluate_cases(model, device, args.max_new_tokens)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started

    report = {
        "schema_version": 1,
        "provenance": {
            "commit": _git_commit(),
            "weights": args.weights,
            "checkpoint_step": metadata.get("step"),
            "config": _config_info(args.config),
            "seed": args.seed,
        },
        "system": system_metadata(device),
        "model": {
            "params": model.n_params(),
            "context": model.cfg.block,
            "precision": args.precision,
        },
        "timing": {
            "elapsed_seconds": elapsed,
            "max_new_tokens_per_case": args.max_new_tokens,
            "peak_vram_bytes": (
                torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
            ),
        },
        "summary": summarize_results(results),
        "results": results,
        "note": "Repeatable project regression benchmark; not a standardized intelligence score.",
    }

    if args.compare:
        base = json.loads(Path(args.compare).read_text(encoding="utf-8"))
        report["comparison"] = compare_reports(base, report)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({**report["summary"], **report["timing"]}, indent=2))
    print(f"Saved benchmark report to {output}")


if __name__ == "__main__":
    main()
