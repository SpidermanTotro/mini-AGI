"""Adaptive-depth probe owned by Greenlight training."""

import random

import numpy as np
import torch

from minagi.recur import load_recur
from minagi.tokenizer import load_tokenizer


def run_ponder_probe(args):
    """Measure whether recurrence depth rises with arithmetic difficulty."""
    import corpora.arithmetic as math_data

    device = torch.device(args.device)
    model, _checkpoint = load_recur(args.ckpt, device)
    tok = load_tokenizer(args.data)
    rng = random.Random(0)

    print(f"{'digits':>7} {'mean steps':>11} {'max':>5}  {'example':<34}")
    rows = []
    for digits in range(1, args.max_digits + 1):
        steps = []
        example = ""
        for _ in range(args.n):
            fn, cap, _ = math_data.TASKS[args.task]
            line = fn(rng, min(digits, cap), False)
            prompt = line.rpartition("=")[0] + "="
            example = example or prompt
            ids = torch.tensor([tok.encode(prompt).ids], device=device)
            with torch.no_grad():
                _, extra = model(ids, collect=True)
            steps.append(float(extra["steps"][0, -1]))

        mean_steps = float(np.mean(steps))
        rows.append((digits, mean_steps, max(steps)))
        print(
            f"{digits:>7} {mean_steps:>11.2f} {max(steps):>5.0f}  "
            f"{example:<34}"
        )

    lo = rows[0][1]
    hi = rows[-1][1]
    print(
        f"\n1-digit {lo:.2f} steps -> {args.max_digits}-digit {hi:.2f} steps "
        f"({hi-lo:+.2f})"
    )
    print(
        "adaptive compute is working"
        if hi - lo > 0.15
        else "FLAT - the halting head is not responding to difficulty"
    )
    return 0
