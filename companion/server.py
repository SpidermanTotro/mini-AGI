"""Read-only DragonForge companion API served by the Greenlight Recur Linux core."""

import json
import os
from pathlib import Path

from flask import Flask, jsonify

app = Flask(__name__)


def _read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError):
        return {}


@app.get("/api/v1/status")
def status():
    history = _read_json(os.environ.get("GREENLIGHT_STATUS_JSON", "run-status.json"))
    return jsonify({
        "service": "dragonforge-companion",
        "mode": "read-only",
        "training": history.get("training", False),
        "step": history.get("step"),
        "loss": history.get("loss"),
        "experts": history.get("experts"),
        "vram_gb": history.get("vram_gb"),
        "doctor": history.get("doctor", "unknown"),
    })


@app.get("/api/v1/health")
def health():
    return jsonify({"ok": True, "mode": "read-only"})


@app.get("/api/v1/capabilities")
def capabilities():
    """Stable discovery contract shared by DragonForge companion clients."""
    return jsonify({
        "api_version": 1,
        "api_min_client_version": 1,
        "project": "DragonForge",
        "core": "Greenlight Recur",
        "platform": "linux",
        "features": {
            "status": True,
            "doctor": True,
            "telemetry": True,
            "experiments": False,
            "training_control": False,
        },
    })


if __name__ == "__main__":
    app.run(
        host=os.environ.get("GREENLIGHT_COMPANION_HOST", "127.0.0.1"),
        port=int(os.environ.get("GREENLIGHT_COMPANION_PORT", "8765")),
    )
