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


def _kind(row: dict) -> str | None:
    """
    What a history row calls its type.

    train.py's recorder tags rows "kind"; the doctor's own fixtures used
    "event". Reading only one of them is how the doctor went a year of being
    trusted while looking at nothing: zero samples found, zero findings,
    HEALTHY - on healthy runs and on runs that had crashed. Both spellings are
    accepted so neither a real history nor a test can be quietly unreadable.
    """
    value = row.get("kind", row.get("event"))
    return value if isinstance(value, str) else None


def diagnose(rows: Iterable[dict]) -> dict:
    rows = list(rows)
    steps = [r for r in rows if _kind(r) == "step"]
    vals = [r for r in rows if _kind(r) == "val"]
    findings: list[Finding] = []

    # A run that began and never finished is the failure mode a loss-based
    # doctor cannot see: there is no loss to be bad. The recorder opens a run
    # with "start" and closes it with "done", so their absence from each other
    # is a crash - the process died between them, which is how both bugs fixed
    # in the integration pass presented.
    started = any(_kind(r) == "start" for r in rows)
    finished = any(_kind(r) == "done" for r in rows)
    if started and not finished:
        findings.append(Finding(
            "RUN_TRUNCATED", "critical",
            "The run wrote a start record and no completion record: it died "
            "partway through. Nothing in the loss series can show this.",
            {"rows": len(rows),
             "step_samples": len(steps),
             "val_samples": len(vals)}))
    if rows and not steps:
        findings.append(Finding(
            "NO_TELEMETRY", "critical",
            "The history holds rows but no training telemetry at all. Either "
            "the run produced nothing, or this history is not the schema the "
            "doctor reads - both look identical from inside, and the second "
            "one is how this doctor was inert for so long.",
            {"rows": len(rows)}))
    if steps and not vals:
        findings.append(Finding(
            "NO_EVALUATION", "warning",
            "The run took training steps and never evaluated held-out text. "
            "A run that dies at its first evaluation looks exactly like a run "
            "that is simply still early.",
            {"step_samples": len(steps)}))

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
        "doctor_version": 2,
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


def resume_preflight(weights_dir: str | Path, build=None) -> dict:
    """
    Whether a restart from `weights_dir` can actually take a training step.

    A history-based doctor cannot answer this. The failure it looks for kills
    the run at its first optimiser step, before a single telemetry row is
    written, so the history is empty and every loss-based check passes
    vacuously. This performs the restore the training loop performs and then
    asks the only question that matters: is the optimiser in a state where
    `opt.step()` runs, or will it raise before training anything?

    `build` is injected in tests; by default it builds the model the way
    `train.py stream` does, because that is the path a restart actually takes
    - and it is not the same model `build_paged` builds. `build_paged` gives
    the pool stacked slot tensors, so its parameters are named pool.w1 and the
    moment restore matches none of them; the trainer's pool exposes
    pool.experts.N.w1, and those are the ones that come back without a step
    counter. Checking the wrong build would report a restart as safe.
    """
    import torch

    from minagi import store

    findings: list[Finding] = []
    path = Path(weights_dir)
    if build is None:
        build = _build_like_the_trainer

    manifest_path = path / "manifest.json"
    if not manifest_path.exists():
        return {"doctor_version": 2, "health": "critical",
                "observations": {"weights_dir": str(path)},
                "findings": [asdict(Finding(
                    "NO_CHECKPOINT", "critical",
                    "There is no manifest here, so there is nothing to resume.",
                    {"weights_dir": str(path)}))]}

    import json
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    missing = [entry["file"] for entry in manifest.get("experts", [])
               if not (path / store.EXPERTS / entry["file"]).exists()]
    if missing:
        findings.append(Finding(
            "MISSING_EXPERT_FILE", "critical",
            "The manifest lists expert files that are not on disk. Resuming "
            "would load a pool the manifest does not describe.",
            {"count": len(missing), "examples": missing[:5]}))

    try:
        model, _cfg, pool, _man = build(str(path), torch.device("cpu"))
    except Exception as exc:
        return _preflight_report(path, manifest, findings + [Finding(
            "RESTORE_WILL_NOT_LOAD", "critical",
            "Building the model this checkpoint needs raised, so a restart "
            "here fails before it trains anything.",
            {"error": f"{type(exc).__name__}: {exc}"})], [], None)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-6)
    if hasattr(pool, "attach_optimiser"):
        # only the paged pool routes its slots through the optimiser's hooks;
        # the trainer's pool restores expert moments at load time instead
        pool.attach_optimiser(opt)
    store.load(model, str(path), opt=opt, device=torch.device("cpu"))
    store._load_optim(opt, model, str(path))
    if hasattr(pool, "_own_moments"):
        # the pool fills its slots' moments just before a step, and an arriving
        # expert brings moments with no counter of its own
        pool._own_moments()

    restored = [(n, st) for n, st in _named_state(model, opt)
                if "exp_avg" in st]
    counterless = [n for n, st in restored if "step" not in st]
    if counterless:
        findings.append(Finding(
            "MISSING_OPTIM_STEP", "critical",
            "Restored optimiser moments carry no step counter. AdamW reads "
            "state['step'] on its first step, so this restart would die with "
            "KeyError: 'step' before training a batch.",
            {"restored": len(restored), "without_step": len(counterless),
             "examples": counterless[:5]}))
    elif restored:
        t = max(float(st["step"]) for _n, st in restored)
        findings.append(Finding(
            "OPTIM_RESTORED", "pass",
            "The optimiser came back with its moments and its step counter; "
            "the restart can step.",
            {"restored": len(restored), "steps_of_history": t}))

    crashed = None
    try:
        loss = sum(p.square().mean() for p in model.parameters())
        loss.backward()
        opt.step()
    except Exception as exc:                      # the restart itself failed
        crashed = f"{type(exc).__name__}: {exc}"
        findings.append(Finding(
            "RESTART_STEP_FAILED", "critical",
            "Replaying the restore and taking an optimiser step raised, so a "
            "run resumed from here would crash rather than continue learning.",
            {"error": crashed}))

    critical = any(f.severity == "critical" for f in findings)
    warning = any(f.severity == "warning" for f in findings)
    return {
        "doctor_version": 2,
        "health": "critical" if critical else "warning" if warning else "healthy",
        "observations": {
            "weights_dir": str(path),
            "step_on_disk": manifest.get("step"),
            "experts": len(manifest.get("experts", [])),
            "restored_moments": len(restored),
            "restored_without_step": len(counterless),
            "first_step_raised": crashed,
        },
        "findings": [asdict(f) for f in findings],
    }


def _build_like_the_trainer(wdir, device):
    """
    Build the model `train.py stream` would build from this directory.

    Taken from cmd_stream: the checkpoint's own recorded shape, the tokenizer's
    vocabulary, and the pool resized to what the manifest says the pool holds.
    """
    import json

    from minagi.recur import RecurCoder, RecurConfig

    manifest = json.loads((Path(wdir) / "manifest.json").read_text(
        encoding="utf-8"))
    cfgd = dict(manifest.get("cfg") or {})
    cfgd.update(vocab_size=265, use_pool=True)
    # Router rows are as wide as the model was BUILT, which is neither the
    # expert count nor the growth ceiling: a pool that never grew keeps the
    # width config.yaml gave it. The checkpoint's own router tensors are the
    # authority, so read their width rather than inferring it.
    cfgd["pool_max"] = _saved_router_width(wdir) or int(
        cfgd.get("pool_max") or manifest.get("n_experts") or 64)
    cfg = RecurConfig(**{k: v for k, v in cfgd.items()
                         if k in RecurConfig.__dataclass_fields__})
    model = RecurCoder(cfg).to(device)
    want = int(manifest.get("n_experts") or 0)
    have = model.pool.n_experts()
    if want > have:
        model.pool.add_experts(want - have)
    return model, cfg, model.pool, manifest


def _saved_router_width(wdir) -> int | None:
    """How many rows the router tensors in this checkpoint actually have."""
    import os

    import numpy as np

    path = os.path.join(str(wdir), "routers.npz")
    if not os.path.exists(path):
        return None
    try:
        z = np.load(path)
        for name in z.files:
            if name.endswith("router.weight"):
                return int(z[name].shape[0])
    except Exception:
        return None
    return None




def _preflight_report(path, manifest, findings, restored, crashed):
    critical = any(f.severity == "critical" for f in findings)
    warning = any(f.severity == "warning" for f in findings)
    return {
        "doctor_version": 2,
        "health": "critical" if critical else "warning" if warning else "healthy",
        "observations": {
            "weights_dir": str(path),
            "step_on_disk": manifest.get("step"),
            "experts": len(manifest.get("experts", [])),
            "restored_moments": len(restored),
            "restored_without_step": 0,
            "first_step_raised": crashed,
        },
        "findings": [asdict(f) for f in findings],
    }


def _named_state(model, opt):
    names = {id(p): n for n, p in model.named_parameters()}
    for group in opt.param_groups:
        for p in group["params"]:
            state = opt.state.get(p)
            if state:
                yield names.get(id(p), "?"), state
