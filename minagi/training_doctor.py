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


# -- routing: where the experts' attention actually went --------------------
#
# A router that has collapsed still trains. Its loss falls, because the model
# leans harder on whichever few experts it kept choosing and the trunk does the
# rest - and the pool behind it is dead weight that no loss number reports.
# Only the per-expert counters show it.

def diagnose_experts(records: Iterable[dict]) -> dict:
    """
    Expert routing across a run, from the per-expert history records.

    Each record is one checkpoint's view of the pool: how much routing mass
    every expert took (`use`), its gate, how often it was admitted, and its
    stable id. Records are positional lists paired with `uid`, because a prune
    renumbers positions and renames nothing.
    """
    records = [r for r in records if isinstance(r, dict)]
    findings: list[Finding] = []
    if not records:
        return _verdict(2, findings, {"records": 0})

    last = records[-1]
    use = [float(v) for v in last.get("use") or () if _finite(v)]
    gate = [float(v) for v in last.get("gate") or () if _finite(v)]
    admits = [float(v) for v in last.get("admits") or () if _finite(v)]
    total = sum(use)
    n = len(use)

    if n and total > 0:
        share = sorted((v / total for v in use), reverse=True)
        top10 = sum(share[:10])
        hhi = sum(s * s for s in share)
        # A pool with no preference splits routing perfectly evenly, and a pool
        # with total preference hands it to one expert. Measured between those
        # two, so the reading does not drift with the size of the pool: 0 is
        # perfectly even, 1 is a single expert taking everything. A raw HHI
        # cannot do this - uniform is 1/n, which is 1/32 for a small pool and
        # swamps any real concentration.
        uniform = 1.0 / n
        preference = max(0.0, (hhi - uniform) / (1.0 - uniform)) if n > 1 else 0.0
        if top10 >= 0.80:
            findings.append(Finding(
                "ROUTING_COLLAPSED", "critical",
                "Ten experts took at least 80% of all routing. The loss can "
                "still fall - the trunk covers - so nothing but these "
                "counters shows the rest of the pool is dead weight.",
                {"experts": n, "top10_share": round(top10, 4),
                 "preference": round(preference, 4)}))
        elif preference < 0.05:
            findings.append(Finding(
                "ROUTING_HAS_NO_PREFERENCE", "warning",
                "Routing is spread almost perfectly evenly, so the router is "
                "not preferring anything: every expert gets about the same "
                "share. That is what a pool full of interchangeable experts "
                "looks like, and it is invisible in the loss.",
                {"experts": n, "preference": round(preference, 5),
                 "herfindahl": round(hhi, 5), "uniform": round(uniform, 5)}))

        idle = sum(1 for v in use if v <= 0.0)
        if n >= 8 and idle >= max(1, n // 4):
            findings.append(Finding(
                "DEAD_EXPERTS", "warning",
                "A quarter or more of the pool has never been routed to. They "
                "are occupying slots, disk and router rows for nothing.",
                {"experts": n, "never_used": idle}))

    if len(gate) >= 8:
        srt = sorted(gate)
        p10, p90 = srt[len(srt) // 10], srt[(len(srt) * 9) // 10]
        if p10 > 0 and p90 / p10 < 1.05:
            findings.append(Finding(
                "GATES_NOT_SEPARATING", "warning",
                "The gates are all within 5% of each other. A gate that never "
                "moves cannot express a preference, so admission falls back "
                "on routing mass alone.",
                {"gate_p10": round(p10, 6), "gate_p90": round(p90, 6),
                 "ratio": round(p90 / p10, 4)}))
        elif max(gate) <= 0.0:
            findings.append(Finding(
                "GATES_ALL_ZERO", "critical",
                "Every gate is zero, so nothing can be routed to.",
                {"experts": len(gate)}))

    if len(admits) >= 8:
        segments = float(last.get("segments") or 0)
        mean_admits = statistics.fmean(admits)
        if segments > 0 and mean_admits > 8 * len(admits):
            findings.append(Finding(
                "EXPERT_CHURN", "warning",
                "Experts are being admitted far more often than they are "
                "learned from. A working set that turns over this fast never "
                "carries anything forward.",
                {"mean_admits": round(mean_admits, 2),
                 "admits_per_segment": round(mean_admits / segments, 3)}))

    # experts that vanished while the run kept going
    vanished = 0
    chars_seen = None
    for earlier, later in zip(records, records[1:]):
        before = {int(u) for u in earlier.get("uid") or ()}
        after = {int(u) for u in later.get("uid") or ()}
        vanished += len(before - after)
        chars_seen = later.get("chars")
    if vanished:
        findings.append(Finding(
            "EXPERTS_PRUNED", "warning",
            "Experts were removed from the pool during the run. Each one was "
            "something the model had learned and now cannot.",
            {"removed": vanished, "chars_at_last_record": chars_seen}))

    return _verdict(2, findings, {
        "records": len(records),
        "experts": n,
        "chars": last.get("chars"),
        "segments": last.get("segments"),
        "total_use": total,
        "top10_use_share": round(sum(sorted((v / total for v in use),
                                            reverse=True)[:10]), 4) if total else None,
        "preference": round(preference, 5) if total else None,
        "never_used": sum(1 for v in use if v <= 0.0),
        "admits_total": sum(admits),
    })


# -- retention: what one domain learned while another was being read ----------

def diagnose_retention(rows: Iterable[dict]) -> dict:
    """
    Per-domain held-out loss over a run.

    Averaging across domains is how catastrophic forgetting hides: a pool of
    experts that keeps every subject alive at once scores well on the mean and
    can still have lost one entirely. The recorder writes each domain's loss
    separately, so this reads them separately.
    """
    rows = list(rows)
    findings: list[Finding] = []
    vals = [r for r in rows if _kind(r) == "val" and isinstance(r.get("per_domain"), dict)]
    if not vals:
        return _verdict(2, findings, {"evaluations": 0, "domains": 0})

    domains = sorted({d for r in vals for d in r["per_domain"]})
    tracked = [d for d in domains
               if sum(1 for r in vals if _finite(r["per_domain"].get(d))) >= 2]
    series = {d: [float(r["per_domain"][d]) for r in vals
                  if _finite(r["per_domain"].get(d))] for d in tracked}

    for domain, points in series.items():
        first, last = points[0], points[-1]
        if last > first * 1.10 and last > first + 0.01:
            worse = (last - first) / max(first, 1e-9)
            findings.append(Finding(
                "DOMAIN_FORGOTTEN", "warning",
                f"{domain} got measurably worse while the run went on. The "
                "run average can improve with this happening.",
                {"domain": domain, "first": round(first, 5),
                 "latest": round(last, 5), "relative": round(worse, 4)}))
        elif last > first * 1.02:
            findings.append(Finding(
                "DOMAIN_SLIPPING", "warning",
                f"{domain} drifted upwards. Small on its own, but it is the "
                "direction forgetting goes.",
                {"domain": domain, "first": round(first, 5),
                 "latest": round(last, 5)}))

    latest_all = {d: points[-1] for d, points in series.items()}
    if len(latest_all) >= 2:
        worst = max(latest_all, key=lambda d: latest_all[d])
        best = min(latest_all, key=lambda d: latest_all[d])
        if latest_all[best] > 0 and latest_all[worst] > latest_all[best] * 3:
            findings.append(Finding(
                "DOMAIN_IMBALANCE", "warning",
                f"{worst} is over three times worse than {best} on the same "
                "model. That is a routing or capacity problem, not a general "
                "one.",
                {"worst": worst, "worst_val": round(latest_all[worst], 5),
                 "best": best, "best_val": round(latest_all[best], 5)}))

    return _verdict(2, findings, {
        "evaluations": len(vals),
        "domains": len(domains),
        "tracked": len(tracked),
        "per_domain_latest": {d: round(series[d][-1], 5) for d in tracked},
    })


def _verdict(version: int, findings: list[Finding], observations: dict) -> dict:
    critical = any(f.severity == "critical" for f in findings)
    warning = any(f.severity == "warning" for f in findings)
    return {
        "doctor_version": version,
        "health": "critical" if critical else "warning" if warning else "healthy",
        "observations": observations,
        "findings": [asdict(f) for f in findings],
    }


# -- generation: what actually came out, against what the loss said -----------
#
# The failure this exists for is the one that looks like success: loss falls,
# perplexity falls, the graph descends - and the model writes "bfjjk hkdkgm".
# Nothing in a loss series can see that, because the loss is computed over the
# text the model was trained on, while the damage is in the text it produces.

_WORD = __import__("re").compile(r"[A-Za-z]{2,}")
_VOWELS = frozenset("aeiouy")


def _max_consonant_run(text: str) -> int:
    """
    Longest run of consonants with no vowel in it.

    This is what actually catches "bfjjk hkdkgm tkkitk". A word list is the
    obvious tool and the wrong one - the corpus is code, chess and arithmetic
    as well as prose, so every dictionary flags valid output. But English has
    no six consonants in a row without a vowel, not even in "strengths"; real
    text tops out around five. Run it across the whole sample rather than per
    token, because the nonsense does not respect word boundaries - and it is
    the run *across* boundaries that reaches double digits.
    """
    run = longest = 0
    for ch in text:
        if ch.isalpha() and ch.lower() not in _VOWELS:
            run += 1
            longest = max(longest, run)
        else:
            run = 0
    return longest


def text_quality(text: str, n: int = 8) -> dict:
    """
    Degeneration metrics for one generated sample.

    Deliberately dictionary-free. The corpus is code, chess, arithmetic and
    prose in one run, so any word list flags valid output as misspelled; these
    measure shape, which is what actually breaks.
    """
    import math as _math

    stripped = text.strip()
    if not stripped:
        return {"chars": 0, "repeated_ngrams": None, "longest_run": 0,
                "char_entropy": None, "distinct_ratio": None,
                "word_like_rate": None, "max_consonant_run": 0}

    grams = [stripped[i:i + n] for i in range(len(stripped) - n + 1)]
    seen: set[str] = set()
    repeats = 0
    for g in grams:
        if g in seen:
            repeats += 1
        else:
            seen.add(g)

    longest = run = 1
    for a, b in zip(stripped, stripped[1:]):
        run = run + 1 if a == b else 1
        longest = max(longest, run)

    counts: dict[str, int] = {}
    for ch in stripped:
        counts[ch] = counts.get(ch, 0) + 1
    total = len(stripped)
    entropy = -_math.fsum((c / total) * _math.log2(c / total)
                          for c in counts.values())

    tokens = stripped.split()
    word_like = sum(1 for t in tokens if _WORD.fullmatch(t))

    return {
        "chars": total,
        "repeated_ngrams": (repeats / len(grams)) if grams else None,
        "longest_run": longest,
        "char_entropy": entropy,
        "distinct_ratio": (len(set(t.lower() for t in tokens)) / len(tokens))
                          if tokens else None,
        "word_like_rate": (word_like / len(tokens)) if tokens else None,
        "max_consonant_run": _max_consonant_run(stripped),
    }


_STEP_RE = __import__("re").compile(
    r"^step ([\d,]+)\s+([\d.]+)M of .*?\(.*?\)\s+(\d+) min\s+(\d+) experts")
_VAL_RE = __import__("re").compile(r"^held-out loss ([\d.]+) \+/-")
_REPEAT_RE = __import__("re").compile(r"\[(raw|adapted)\]\s+repeated \d+-grams (\d+)%")
_PROMPT_RE = __import__("re").compile(r"^prompt: '")
_DOMAIN_RE = __import__("re").compile(r"^--- (.+) ---$")


def parse_samples(path: str | Path) -> list[dict]:
    """
    Read runs/samples.txt into per-checkpoint blocks.

    The sampler writes prose, not JSON, so this reads the shape the file has:
    a step header, the held-out loss for that step, then one section per domain
    with a prompt and the raw and repetition-guarded continuations.
    """
    import re

    blocks: list[dict] = []
    current: dict | None = None
    domain: str | None = None
    pending_raw = False

    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.rstrip("\n")
            m = _STEP_RE.match(line)
            if m:
                current = {"step": int(m.group(1).replace(",", "")),
                           "chars_m": float(m.group(2)),
                           "minutes": int(m.group(3)),
                           "experts": int(m.group(4)),
                           "val": None, "samples": []}
                blocks.append(current)
                domain = None
                continue
            if current is None:
                continue
            m = _VAL_RE.match(line)
            if m:
                current["val"] = float(m.group(1))
                continue
            m = _DOMAIN_RE.match(line)
            if m:
                domain = m.group(1).strip()
                pending_raw = False
                continue
            m = _REPEAT_RE.match(line)
            if m and domain is not None:
                current["samples"].append({
                    "domain": domain, "mode": m.group(1),
                    "repeated": int(m.group(2)) / 100.0})
                pending_raw = m.group(1)
                continue
            if pending_raw is not None and domain is not None:
                if _PROMPT_RE.match(line):
                    pending_raw = None
                    continue
                if line.startswith("--- ") or line.startswith("="):
                    pending_raw = None
                    continue
                # The continuation is the line under its marker. Both are kept:
                # the guarded one is what the reader sees, the raw one is what
                # the model actually wanted to emit.
                current["samples"].append({
                    "domain": domain, "mode": pending_raw + "_text",
                    "text": line})
                pending_raw = None
    return blocks


def diagnose_generation(blocks: Iterable[dict]) -> dict:
    """
    Generation quality against the loss that was falling next to it.

    The comparison is early run against late run, thirds of the way through, so
    a single bad sample cannot condemn a run and a single good one cannot save
    it.
    """
    import math

    blocks = [b for b in blocks if isinstance(b, dict) and b.get("samples")]
    findings: list[Finding] = []
    if not blocks:
        return _verdict(2, findings, {"blocks": 0})

    def adapted(block):
        return [s for s in block["samples"]
                if s["mode"] == "adapted" and s.get("repeated") is not None]

    def raw(block):
        return [s for s in block["samples"]
                if s["mode"] == "raw" and s.get("repeated") is not None]

    thirds = max(1, len(blocks) // 3)

    def mean_repeat(sel, group):
        vals = [s["repeated"] for b in group for s in sel(b)]
        return sum(vals) / len(vals) if vals else None

    early, late = blocks[:thirds], blocks[-thirds:]
    early_adapted = mean_repeat(adapted, early)
    late_adapted = mean_repeat(adapted, late)
    early_raw = mean_repeat(raw, early)
    late_raw = mean_repeat(raw, late)

    def mean_val(group):
        vals = [b["val"] for b in group if _finite(b.get("val"))]
        return sum(vals) / len(vals) if vals else None

    early_val, late_val = mean_val(early), mean_val(late)

    if early_val and late_val and late_val < early_val * 0.95:
        improving = True
    else:
        improving = False
    if early_adapted is not None and late_adapted is not None:
        moved = late_adapted - early_adapted
        if improving and moved > 0.05:
            findings.append(Finding(
                "LOSS_IMPROVING_OUTPUT_COLLAPSING", "critical",
                "Held-out loss fell while repetition in what the model writes "
                "went UP. The loss is measured over text it was trained on; "
                "this is the discrepancy the two measure separately.",
                {"held_out_early": round(early_val, 4),
                 "held_out_late": round(late_val, 4),
                 "repeated_early": round(early_adapted, 4),
                 "repeated_late": round(late_adapted, 4),
                 "repeated_rise": round(moved, 4)}))
        elif improving and moved > 0.005:
            # Not yet a collapse, and deliberately not reported as one. But the
            # two numbers moved in opposite directions, and the direction is
            # the thing worth watching - a threshold loose enough to catch a
            # 27% rise in repetition is loose enough to fire on noise, and a
            # threshold tight enough to avoid noise will miss a real slide.
            findings.append(Finding(
                "LOSS_AND_OUTPUT_DIVERGING", "warning",
                "Held-out loss improved and repetition in the output rose "
                "slightly. Too small to call degeneration, and the direction "
                "is what forgetting looks like starting.",
                {"held_out_early": round(early_val, 4),
                 "held_out_late": round(late_val, 4),
                 "repeated_early": round(early_adapted, 4),
                 "repeated_late": round(late_adapted, 4),
                 "repeated_rise": round(moved, 4),
                 "relative_rise": round(moved / max(early_adapted, 1e-9), 4)}))
        elif late_adapted >= 0.35:
            findings.append(Finding(
                "GENERATION_STILL_REPETITIVE", "warning",
                "A third or more of every 8-gram in the guarded output is a "
                "repeat. The guard hides the runaway loops; the model still "
                "cannot write without them.",
                {"repeated_late": round(late_adapted, 4)}))
    if early_raw is not None and late_raw is not None and late_raw >= 0.90:
        findings.append(Finding(
            "RAW_OUTPUT_DEGENERATE", "critical",
            "Unguarded output is 90% repeated 8-grams by the end of the run: "
            "the model is emitting loops, not text.",
            {"repeated_late": round(late_raw, 4),
             "repeated_early": round(early_raw, 4)}))

    texts = [s["text"] for b in blocks[-thirds:] for s in b["samples"]
             if s["mode"] in ("raw_text", "adapted_text")]
    if texts:
        qualities = [text_quality(t) for t in texts]
        entropies = [q["char_entropy"] for q in qualities
                     if _finite(q["char_entropy"])]
        runs = [q["longest_run"] for q in qualities]
        wordy = [q["word_like_rate"] for q in qualities
                 if _finite(q["word_like_rate"])]
        conson = [q["max_consonant_run"] for q in qualities]
        if entropies and statistics.fmean(entropies) < 2.5:
            findings.append(Finding(
                "LOW_CHARACTER_ENTROPY", "warning",
                "The output carries about as much information per character "
                "as a small alphabet. That is the shape of a loop, not of "
                "language.",
                {"mean_char_entropy": round(statistics.fmean(entropies), 3),
                 "samples": len(entropies)}))
        if runs and max(runs) >= 40:
            findings.append(Finding(
                "REPETITION_RUNAWAY", "warning",
                "One character repeated without interruption for "
                f"{max(runs)} characters in a single sample.",
                {"longest_run": max(runs)}))
        if conson and max(conson) >= 6:
            findings.append(Finding(
                "OUTPUT_NOT_WORD_SHAPED", "warning",
                "The output carries runs of consonants no English word has - "
                "six or more in a row without a vowel. This is what 'bfjjk "
                "hkdkgm' measures, and it survives into the guarded output.",
                {"max_consonant_run": max(conson),
                 "samples": len(conson)}))
        elif wordy and statistics.fmean(wordy) < 0.55:
            findings.append(Finding(
                "OUTPUT_MOSTLY_NON_ALPHA", "warning",
                "Most whitespace-separated tokens in the late output contain "
                "no letters at all.",
                {"word_like_rate": round(statistics.fmean(wordy), 3),
                 "samples": len(wordy)}))

    domains = sorted({s["domain"] for b in blocks for s in b["samples"]
                      if s.get("domain")})
    worst = None
    if domains and late:
        worst = max(domains, key=lambda d: (
            statistics.fmean([s["repeated"] for b in late
                              for s in adapted(b) if s["domain"] == d] or [0])
            if any(s["domain"] == d for b in late for s in adapted(b))
            else -1))
    return _verdict(2, findings, {
        "blocks": len(blocks),
        "steps": [blocks[0]["step"], blocks[-1]["step"]],
        "domains": domains,
        "held_out_early": round(early_val, 4) if early_val else None,
        "held_out_late": round(late_val, 4) if late_val else None,
        "repeated_early": round(early_adapted, 4) if early_adapted is not None else None,
        "repeated_late": round(late_adapted, 4) if late_adapted is not None else None,
        "raw_repeated_late": round(late_raw, 4) if late_raw is not None else None,
        "worst_domain": worst,
    })
