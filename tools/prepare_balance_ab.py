#!/usr/bin/env python3
"""Prepare reproducible baseline/candidate configs for the router-balance A/B.

This tool does not launch training. It freezes the only intended routing
difference into two generated YAML profiles and records a manifest so both runs
can be executed from the same source config/checkpoint/corpus/command.
"""

import argparse
import copy
import hashlib
import json
import shlex
from pathlib import Path

import yaml


# Both arms disable legacy exploration so the balance coefficient is the ONLY change.
# This does not compare against the production R5 explore_bias=0.65 default;
# that requires a separately declared reference arm.
BASELINE = {"explore_bias": 0.0, "balance": 0.0}
CANDIDATE = {"explore_bias": 0.0, "balance": 0.001}


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def prepare(source, out_dir, train_command):
    source = Path(source).resolve()
    out_dir = Path(out_dir).resolve()
    cfg = yaml.safe_load(source.read_text()) or {}
    if not isinstance(cfg, dict):
        raise ValueError("source config must contain a YAML mapping")
    pool = cfg.get("pool")
    if not isinstance(pool, dict):
        raise ValueError("source config must contain a pool mapping")

    # train.py stream defaults --weights-dir to "weights", independently
    # of the config. A command without explicit per-arm output paths could
    # overwrite the same checkpoint twice and invalidate the experiment.
    if ("--weights-dir {weights_dir}" not in train_command
            or "--out {run_dir}" not in train_command):
        raise ValueError(
            "training command template requires "
            "'--weights-dir {weights_dir} --out {run_dir}'")
    names = ("baseline", "candidate")
    targets = [out_dir / "manifest.json"]
    for name in names:
        targets += [out_dir / f"{name}.yaml",
                    out_dir / name / "weights",
                    out_dir / name / "runs"]
    if any(path.exists() or path.is_symlink() for path in targets):
        raise FileExistsError("A/B profiles/outputs already exist; use a fresh output directory")

    out_dir.mkdir(parents=True, exist_ok=True)
    generated = {}
    for name, routing in (("baseline", BASELINE), ("candidate", CANDIDATE)):
        profile = copy.deepcopy(cfg)
        profile["pool"].update(routing)
        path = out_dir / f"{name}.yaml"
        weights_dir = out_dir / name / "weights"
        run_dir = out_dir / name / "runs"
        command = train_command.replace("{weights_dir}", shlex.quote(str(weights_dir)))
        command = command.replace("{run_dir}", shlex.quote(str(run_dir)))
        path.write_text(yaml.safe_dump(profile, sort_keys=False))
        generated[name] = {
            "config": str(path),
            "config_sha256": _sha256(path),
            "routing": routing,
            "weights_dir": str(weights_dir),
            "run_dir": str(run_dir),
            "command": f"GREENLIGHT_CONFIG={shlex.quote(str(path))} {command}",
        }

    manifest = {
        "source_config": str(source),
        "source_config_sha256": _sha256(source),
        "rule": "same start checkpoint, corpus, seed, steps and command; only pool.balance differs",
        "production_reference": "R5 defaults may use explore_bias>0, which this two-arm trial does not evaluate",
        "checkpoint_rule": "manually seed BOTH separate weights_dir paths from the SAME immutable checkpoint; this tool copies no files",
        "baseline": generated["baseline"],
        "candidate": generated["candidate"],
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--command", required=True,
                   help="template with --weights-dir {weights_dir} and --out {run_dir}, never executed")
    args = p.parse_args()
    manifest = prepare(args.config, args.out, args.command)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
