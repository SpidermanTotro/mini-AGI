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
        selected = str(Path(path).resolve())
        os.environ["GREENLIGHT_CONFIG"] = selected
        os.environ["MINI_AGI_CONFIG"] = selected
    return os.environ.get("GREENLIGHT_CONFIG") or os.environ.get("MINI_AGI_CONFIG")


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
    mismatches = []
    for section, fields in {
        "provenance": ("seed",), "model": ("precision", "context"),
        "system": ("device", "gpu", "torch"),
        "timing": ("max_new_tokens_per_case",),
    }.items():
        for field in fields:
            if base.get(section, {}).get(field) != candidate.get(section, {}).get(field):
                mismatches.append(f"{section}.{field}")
    if (base.get("provenance", {}).get("config", {}).get("sha256") !=
            candidate.get("provenance", {}).get("config", {}).get("sha256")):
        mismatches.append("config.sha256")
    case_key = lambda r: [(x.get("name"), x.get("prompt"), x.get("expected"))
                          for x in r.get("results", [])]
    if case_key(base) != case_key(candidate):
        mismatches.append("prompt cases")
    bh, ch = base.get("held_out") or {}, candidate.get("held_out") or {}
    if any(bh.get(k) != ch.get(k) for k in ("sha256", "chunk", "chunks")):
        mismatches.append("held-out protocol")
    bv, cv = bh.get("scores", {}).get("all"), ch.get("scores", {}).get("all")
    return {
        "protocol_matches": not mismatches,
        "mismatches": mismatches,
        "held_out_loss_delta": cv - bv if bv is not None and cv is not None and not mismatches else None,
        "base_commit": base.get("provenance", {}).get("commit"),
        "candidate_commit": candidate.get("provenance", {}).get("commit"),
        "exact_match_accuracy_delta": (
            c.get("exact_match_accuracy") - b.get("exact_match_accuracy")
            if not mismatches and c.get("exact_match_accuracy") is not None
            and b.get("exact_match_accuracy") is not None
            else None
        ),
        "elapsed_seconds_delta": candidate.get("timing", {}).get("elapsed_seconds", 0)
        - base.get("timing", {}).get("elapsed_seconds", 0),
    }


def main():
    parser = argparse.ArgumentParser(description="Run a reproducible mini-AGI benchmark")
    parser.add_argument("--weights", default="greenlight-16g-r1")
    parser.add_argument("--config", default=None)
    parser.add_argument("--output", default="runs/benchmark.json")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--precision", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--compare")
    parser.add_argument("--held-out", help="held-out text file or directory")
    parser.add_argument("--eval-chunks", type=int, default=32)
    args = parser.parse_args()

    if args.eval_chunks <= 0:
        parser.error("--eval-chunks must be positive")
    if args.max_new_tokens <= 0:
        parser.error("--max-new-tokens must be positive")

    args.config = activate_config(args.config)
    if args.config and not Path(args.config).is_file():
        parser.error(f"config does not exist: {args.config}")

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

    held_out = None
    if args.held_out:
        from .ingest import collect
        from .stream import FolderEvaluator
        files = collect([args.held_out], cache=None)
        if not files:
            parser.error("held-out path contains no readable text")
        digest = hashlib.sha256()
        for path in sorted(files):
            with open(path, 'rb') as source:
                digest.update(hashlib.file_digest(source, 'sha256').digest())
        chunk = min(512, model.cfg.block)
        scores = FolderEvaluator(model, args.held_out, chunk, model.cfg.block,
                                 device, per_domain=False).run(args.eval_chunks)
        if "all" not in scores:
            parser.error("held-out text is too short to score (need at least 8 tokens)")
        held_out = {"sha256": digest.hexdigest(), "chunk": chunk,
                    "chunks": args.eval_chunks, "scores": scores}

    report = {
        "schema_version": 1,
        "held_out": held_out,
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
