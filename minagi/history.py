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
