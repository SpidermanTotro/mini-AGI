"""Pure training schedule and cadence policies."""

import math


def lr_at(step, total, base, warmup, floor_frac=0.1):
    """Warm up linearly, then cosine-decay to a fixed learning-rate floor."""
    if warmup > 0 and step < warmup:
        return base * (step + 1) / warmup
    prog = (step - warmup) / max(1, total - warmup)
    prog = min(1.0, max(0.0, prog))
    return base * (
        floor_frac
        + (1 - floor_frac) * 0.5 * (1 + math.cos(math.pi * prog))
    )


def chars_to_steps(chars, chunk, least=1):
    """Convert a character cadence to optimizer steps without returning zero."""
    return max(least, int(round(int(chars) / max(1, int(chunk)))))


def growth_held_due(step, grow_every_steps):
    """Return whether the lower-frequency held-growth diagnostic is due."""
    cadence = max(1, int(grow_every_steps)) * 10
    return int(step) % cadence == 0
