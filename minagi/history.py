"""Small JSONL history helpers shared by training entry points."""

import json
import os


def truncate_history(path, chars):
    """Drop parseable history rows beyond a resumed character position.

    Unparseable rows are preserved so cleanup never destroys data it cannot
    understand. Returns the number of rows removed.
    """
    if not path or not os.path.exists(path) or chars <= 0:
        return 0

    with open(path) as f:
        rows = f.read().splitlines()

    keep = []
    dropped = 0
    for line in rows:
        try:
            if json.loads(line).get("chars", 0) > chars:
                dropped += 1
                continue
        except (json.JSONDecodeError, TypeError, AttributeError):
            pass
        keep.append(line)

    if dropped:
        with open(path, "w") as f:
            f.write("\n".join(keep) + ("\n" if keep else ""))

    return dropped


def append_jsonl(path, row, *, compact=False):
    """Append one JSON object as one line, creating its parent directory."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    separators = (",", ":") if compact else None
    with open(path, "a") as f:
        f.write(json.dumps(row, separators=separators) + "\n")


class JsonlRecorder:
    """Line-buffered JSONL recorder whose file is always explicitly closable."""

    def __init__(self, path):
        self.path = path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._file = open(path, "a", buffering=1)

    def record(self, kind, step=None, **fields):
        self._file.write(
            json.dumps({"kind": kind, "step": step, **fields}) + "\n"
        )

    def close(self):
        self._file.close()

    @property
    def closed(self):
        return self._file.closed

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False
