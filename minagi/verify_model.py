"""Validate the on-disk model artifact before calling a run complete."""

import argparse
import json
from pathlib import Path

import numpy as np


REQUIRED_BUNDLES = ("manifest.json", "core.npz", "routers.npz")


def _bundle_params(path):
    """Count tensor elements in an npz bundle from the artifact itself."""
    with np.load(path, allow_pickle=False) as bundle:
        return sum(int(np.prod(bundle[name].shape)) for name in bundle.files)


def inspect_model_artifact(path):
    """Return a compact model-artifact report or raise on an incomplete model."""
    root = Path(path)
    missing = [name for name in REQUIRED_BUNDLES if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"incomplete model artifact {root}: missing {', '.join(missing)}"
        )
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    entries = manifest.get("experts")
    if not isinstance(entries, list) or not entries:
        raise ValueError("model manifest contains no expert file inventory")
    declared = int(manifest.get("n_experts", len(entries)))
    if declared != len(entries):
        raise ValueError(
            f"model manifest declares {declared} experts but inventories {len(entries)}"
        )
    missing_experts = [
        entry.get("file") for entry in entries
        if not entry.get("file") or not (root / "experts" / entry["file"]).is_file()
    ]
    if missing_experts:
        raise FileNotFoundError(
            "model artifact is missing expert files: "
            + ", ".join(str(name) for name in missing_experts[:8])
        )

    expert_params = []
    for entry in entries:
        if "params" not in entry:
            raise ValueError(f"expert inventory {entry.get('file')} has no parameter count")
        n = int(entry["params"])
        if n <= 0:
            raise ValueError(
                f"expert inventory {entry.get('file')} has invalid parameter count {n}"
            )
        expert_params.append(n)

    core_params = _bundle_params(root / "core.npz")
    router_params = _bundle_params(root / "routers.npz")
    experts_params = sum(expert_params)
    cfg = manifest.get("cfg") or {}
    resident = max(0, min(int(cfg.get("pool_resident", declared)), declared))

    return {
        "path": str(root),
        "step": manifest.get("step"),
        "val": manifest.get("val"),
        "read_chars": manifest.get("read_chars"),
        "n_experts": declared,
        "d_model": manifest.get("d_model"),
        "d_ff": manifest.get("d_ff"),
        "paged": bool(manifest.get("paged")),
        "total_bytes": manifest.get("total_bytes"),
        "params": {
            "core": core_params,
            "routers": router_params,
            "experts": experts_params,
            "total": core_params + router_params + experts_params,
            "resident_experts": resident,
            "resident": core_params + router_params + sum(expert_params[:resident]),
        },
    }


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Verify that a Greenlight weights directory is a complete model artifact"
    )
    parser.add_argument("weights")
    args = parser.parse_args(argv)
    report = inspect_model_artifact(args.weights)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
