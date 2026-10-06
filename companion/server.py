"""Read-only local companion API for the iPhone dashboard."""

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
        "service": "greenlight-companion",
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


if __name__ == "__main__":
    app.run(host=os.environ.get("GREENLIGHT_COMPANION_HOST", "127.0.0.1"),
            port=int(os.environ.get("GREENLIGHT_COMPANION_PORT", "8765")))
