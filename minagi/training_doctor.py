"""Greenlight Training Doctor: low-overhead diagnosis from training telemetry.

V1 is intentionally observational. It never mutates a model, optimiser, or
checkpoint. That makes it safe to run against preserved experiments such as R4.
"""
from __future__ import annotations

import json
import math
import statistics
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable


@dataclass
class Finding:
    code: str
    severity: str
    message: str
    evidence: dict


def _finite(x):
    return isinstance(x, (int, float)) and math.isfinite(float(x))


def read_history(path: str | Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSONL: {exc}") from exc
            if isinstance(row, dict):
                rows.append(row)
    return rows


def diagnose(rows: Iterable[dict]) -> dict:
    rows = list(rows)
    steps = [r for r in rows if r.get("event") == "step"]
    vals = [r for r in rows if r.get("event") == "val"]
    findings: list[Finding] = []

    losses = [float(r["loss"]) for r in steps if _finite(r.get("loss"))]
    grads = [float(r["grad_norm"]) for r in steps if _finite(r.get("grad_norm"))]
    rates = [float(r["chars_per_s"]) for r in steps if _finite(r.get("chars_per_s"))]

    nonfinite = [r for r in steps if "loss" in r and not _finite(r.get("loss"))]
    if nonfinite:
        findings.append(Finding(
            "NONFINITE_LOSS", "critical",
            "Training produced a NaN or infinite loss; stop and preserve the last good checkpoint.",
            {"count": len(nonfinite)}))

    if losses:
        n = max(1, min(len(losses) // 3, 20))
        early = statistics.fmean(losses[:n])
        late = statistics.fmean(losses[-n:])
        delta = late - early
        if late < early * 0.95:
            findings.append(Finding(
                "LOSS_IMPROVING", "pass",
                "Training loss is materially lower than its early-run baseline.",
                {"early_mean": early, "late_mean": late, "delta": delta}))
        elif len(losses) >= 6:
            findings.append(Finding(
                "LOSS_NOT_IMPROVING", "warning",
                "Training loss is not materially improving across the observed run.",
                {"early_mean": early, "late_mean": late, "delta": delta}))

        if len(losses) >= 8:
            med = statistics.median(losses[:-1])
            if med > 0 and losses[-1] > med * 2.0:
                findings.append(Finding(
                    "LOSS_SPIKE", "warning",
                    "Latest loss is more than 2x the prior median.",
                    {"latest": losses[-1], "prior_median": med}))

    if grads:
        zeroish = sum(abs(g) <= 1e-12 for g in grads)
        if zeroish == len(grads):
            findings.append(Finding(
                "ZERO_GRADIENTS", "critical",
                "All observed gradient norms are zero; weights cannot learn from these steps.",
                {"samples": len(grads)}))
        elif zeroish:
            findings.append(Finding(
                "INTERMITTENT_ZERO_GRADIENTS", "warning",
                "Some observed steps have effectively zero gradient norm.",
                {"zero_samples": zeroish, "samples": len(grads)}))

    if rates and len(rates) >= 4:
        base = statistics.median(rates[: max(2, len(rates)//2)])
        recent = statistics.median(rates[-max(2, len(rates)//4):])
        if base > 0 and recent < base * 0.5:
            findings.append(Finding(
                "THROUGHPUT_COLLAPSE", "warning",
                "Recent character throughput is below half the earlier median.",
                {"baseline_chars_per_s": base, "recent_chars_per_s": recent}))

    val_losses = [float(r["val"]) for r in vals if _finite(r.get("val"))]
    if len(val_losses) >= 2:
        if val_losses[-1] < val_losses[0]:
            findings.append(Finding(
                "VAL_IMPROVING", "pass",
                "Held-out loss improved across recorded evaluations.",
                {"first": val_losses[0], "latest": val_losses[-1],
                 "delta": val_losses[-1] - val_losses[0]}))
        elif losses and losses[-1] < losses[0]:
            findings.append(Finding(
                "GENERALIZATION_GAP", "warning",
                "Training loss improved while held-out loss did not.",
                {"first_val": val_losses[0], "latest_val": val_losses[-1]}))

    critical = any(f.severity == "critical" for f in findings)
    warning = any(f.severity == "warning" for f in findings)
    health = "critical" if critical else "warning" if warning else "healthy"

    return {
        "doctor_version": 1,
        "health": health,
        "observations": {
            "rows": len(rows),
            "step_samples": len(steps),
            "val_samples": len(vals),
            "loss_first": losses[0] if losses else None,
            "loss_latest": losses[-1] if losses else None,
            "grad_norm_latest": grads[-1] if grads else None,
            "chars_per_s_latest": rates[-1] if rates else None,
        },
        "findings": [asdict(f) for f in findings],
    }


def diagnose_file(path: str | Path) -> dict:
    return diagnose(read_history(path))
