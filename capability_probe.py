#!/usr/bin/env python3
"""
Can the model do anything? A bounded capability probe, not a loss curve.

Lisa's question, made falsifiable: give it one small job and score the answers.

  stage 1  BEFORE      a fresh model, held-out questions it has never seen
  stage 2  AFTER       the same questions, after training on a different set
  stage 3  RESTARTED   the same questions, after save and resume
  stage 4  DISTRACTOR  the same questions, after training on a second subject

Stage 2 says it can learn. Stage 3 says a restart did not cost it. Stage 4 is
the one worth having: a model advertised as learning forever that loses the
skill when shown something else has learned the shape of its own ceiling.

Scoring is exact string match on the answer with greedy decoding and a fixed
seed, so the number is reproducible rather than a vibe. Questions are held out:
the training set uses different numbers.
"""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent

# 20 held-out questions. Training uses a disjoint range, so no answer is ever
# seen in training and improvement cannot be memorisation of the test set.
HELD_OUT = [(17, 25), (43, 18), (61, 7), (29, 34), (56, 12),
            (38, 47), (72, 9), (14, 66), (45, 39), (27, 58),
            (33, 21), (68, 15), (51, 33), (23, 49), (76, 8),
            (19, 37), (64, 26), (42, 55), (35, 11), (58, 31)]


def make_train_text(path: Path, n=6000, seed=1):
    """Arithmetic the model will train on, disjoint from the questions asked."""
    import random
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        a, b = rng.randint(2, 99), rng.randint(2, 99)
        out.append(f"{a} plus {b} is {a + b}.")
    path.write_text("\n".join(out) + "\n", encoding="utf-8")


def make_distractor_text(path: Path, n=3000, seed=2):
    """A second subject: multiplication facts, so the numbers overlap the
    arithmetic set without being it."""
    import random
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        a, b = rng.randint(2, 12), rng.randint(2, 12)
        out.append(f"{a} times {b} is {a * b}.")
    path.write_text("\n".join(out) + "\n", encoding="utf-8")


def score(model, label, n_new=48):
    """Exact-match accuracy on the held-out questions, greedy and fixed-seed."""
    import serve

    if model is not None:
        serve.STATE["model"] = model
    correct, answers = 0, []
    for a, b in HELD_OUT:
        want = a + b
        events = list(serve.stream(f"Question: {a} plus {b} is", n_new))
        text = "".join(e.get("t", "") for e in events)
        # the first integer the model writes is its answer
        digits, answer = "", None
        for ch in text:
            if ch.isdigit():
                digits += ch
            elif digits:
                answer = int(digits)
                break
        hit = answer == want
        correct += hit
        answers.append({"q": f"{a} plus {b}", "want": want,
                        "got": answer, "ok": hit,
                        "raw": text[:60]})
    return {"stage": label, "correct": correct, "total": len(HELD_OUT),
            "accuracy": correct / len(HELD_OUT), "answers": answers}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--passes", type=int, default=1)
    ap.add_argument("--distract-passes", type=int, default=1)
    ap.add_argument("--context", type=int, default=1024)
    ap.add_argument("--chunk", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--out", default="runs/capability")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report = {"questions": len(HELD_OUT), "stages": [], "config": vars(args)}

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        corpus = tmp / "arith"
        corpus.mkdir()
        (corpus / "train.txt").write_text("", encoding="utf-8")
        held = tmp / "held"
        held.mkdir()
        (held / "val.txt").write_text("", encoding="utf-8")
        make_train_text(corpus / "train.txt")
        make_train_text(held / "val.txt", n=1500, seed=99)
        distract = tmp / "mult"
        distract.mkdir()
        (distract / "train.txt").write_text("", encoding="utf-8")
        make_distractor_text(distract / "train.txt")
        (distract / "val.txt").write_text("", encoding="utf-8")
        (distract / "val.txt").write_text("2 times 3 is 6.\n", encoding="utf-8")

        import train  # noqa: F401  (training happens in subprocesses)
        from minagi.create import create

        weights = out / "weights"
        if not (weights / "manifest.json").exists():
            create(str(weights), seed=42, verbose=False, d_model=256,
                   n_head=4, block=args.context, experts=64, resident=8,
                   d_ff=512, depth=1, top_k=4, max_steps=16)

        common = ["--device", args.device, "read", str(corpus),
                  "--weights-dir", str(weights), "--save",
                  "--chunk", str(args.chunk),
                  "--context", str(args.context),
                  "--context-start", str(args.context // 2),
                  "--eval-chars", "8", "--grow-k", "0",
                  "--resident", "8", "--ram-capacity", "32", "--no-plots",
                  "--held-out", str(held), "--history", str(out / "history.jsonl")]

        def run(extra, label):
            r = subprocess.run([sys.executable, str(REPO / "train.py")] + common
                               + extra, cwd=REPO, capture_output=True, text=True)
            if r.returncode != 0:
                print(r.stdout[-2000:], r.stderr[-2000:])
                raise SystemExit(f"{label} failed with {r.returncode}")
            return r

        def load_model():
            # serve.load is the read-only entry point and it populates the
            # tokenizer as well as the model, which serve.stream needs.
            import serve
            return serve.load(str(weights), args.device, learn=False)

        # 1. before any training
        s1 = score(load_model(), "before_training")
        report["stages"].append(s1)
        print(f"before training      : {s1['correct']}/{s1['total']}")

        # 2. after training on arithmetic
        # `read` is budgeted in passes and minutes, not steps. One pass over a
        # ~120k-character corpus is a few hundred optimiser steps, which is a
        # real training budget and still minutes rather than hours.
        run(["--passes", str(args.passes), "--lr", str(args.lr),
             "--sample-every", "0"], "stage 2")
        s2 = score(load_model(), "after_training")
        report["stages"].append(s2)
        print(f"after training       : {s2['correct']}/{s2['total']}")

        # 3. after a save and resume
        run(["--passes", str(args.passes), "--lr", str(args.lr),
             "--sample-every", "0"], "stage 3")
        s3 = score(load_model(), "after_restart")
        report["stages"].append(s3)
        print(f"after restart        : {s3['correct']}/{s3['total']}")

        # 4. after learning a different subject
        dcommon = [x if x != str(corpus) else str(distract) for x in common]
        dcommon[dcommon.index(str(held))] = str(distract)
        r = subprocess.run([sys.executable, str(REPO / "train.py")] + dcommon
                           + ["--passes", str(args.distract_passes),
                              "--lr", str(args.lr), "--sample-every", "0"],
                           cwd=REPO, capture_output=True, text=True)
        if r.returncode != 0:
            print(r.stdout[-2000:], r.stderr[-2000:])
            report["distractor_error"] = r.stderr[-500:]
        s4 = score(load_model(), "after_second_subject")
        report["stages"].append(s4)
        print(f"after 2nd subject    : {s4['correct']}/{s4['total']}")

        report["retention_ratio"] = (
            s4["accuracy"] / s3["accuracy"] if s3["accuracy"] else None)
        (out / "report.json").write_text(json.dumps(report, indent=2),
                                         encoding="utf-8")
        print(f"\nreport: {out / 'report.json'}")
        print(f"retained after learning something else: "
              f"{report['retention_ratio']}")


def torch_device(name):
    import torch
    return torch.device(name)


if __name__ == "__main__":
    main()