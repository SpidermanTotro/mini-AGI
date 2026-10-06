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
from pathlib import Path

import yaml


BASELINE = {"explore_bias": 0.65, "balance": 0.0}
CANDIDATE = {"explore_bias": 0.0, "balance": 3.5e-4}


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

    out_dir.mkdir(parents=True, exist_ok=True)
    generated = {}
    for name, routing in (("baseline", BASELINE), ("candidate", CANDIDATE)):
        profile = copy.deepcopy(cfg)
        profile["pool"].update(routing)
        path = out_dir / f"{name}.yaml"
        path.write_text(yaml.safe_dump(profile, sort_keys=False))
        generated[name] = {
            "config": str(path),
            "config_sha256": _sha256(path),
            "routing": routing,
            "command": f"GREENLIGHT_CONFIG={path} {train_command}",
        }

    manifest = {
        "source_config": str(source),
        "source_config_sha256": _sha256(source),
        "rule": "same start checkpoint, corpus, seed, steps and command; routing mode only",
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
                   help="identical training command used for both arms")
    args = p.parse_args()
    manifest = prepare(args.config, args.out, args.command)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
