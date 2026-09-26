#!/usr/bin/env python3
"""Run bounded Greenlight train/evaluate rounds with reviewed feedback.

Uses an isolated lab checkpoint by default. Generated model text is never
recycled automatically; only explicit, human-reviewed corrections are added.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
R1_WEIGHTS = (ROOT / "greenlight-16g-r1").resolve()


def read_feedback(path: Path | None) -> list[tuple[str, str]]:
    if path is None:
        return []
    pairs = []
    with path.open(encoding="utf-8") as source:
        for line_no, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {error}") from error
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: each record must be an object")
            prompt, target = row.get("prompt"), row.get("target")
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError(f"{path}:{line_no}: prompt must be non-empty text")
            if not isinstance(target, str) or not target.strip():
                raise ValueError(f"{path}:{line_no}: target must be non-empty text")
            pairs.append((prompt.strip(), target.strip()))
    return pairs


def write_feedback(path: Path, pairs: list[tuple[str, str]]) -> Path | None:
    if not pairs:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output:
        for prompt, target in pairs:
            prompt = prompt.rstrip()
            output.write(f"{prompt}\n{target.rstrip()}\n")
            if prompt.endswith("<bot>"):
                output.write("</bot>\n")
    return path


def collect_interactive_feedback(report_path: Path) -> list[tuple[str, str]]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    pairs = []
    print("Enter a reviewed replacement completion; blank skips it.")
    for result in report.get("results", []):
        print(f"\n[{result['name']}]\n{result['prompt']}{result['completion']}")
        target = input("Correction: ").strip()
        if target:
            pairs.append((result["prompt"], target))
    return pairs


def _run(command: list[str], cwd: Path, env: dict[str, str],
         log_path: Path | None = None) -> int:
    if log_path is None:
        return subprocess.run(command, cwd=cwd, env=env, check=False).returncode
    with log_path.open("w", encoding="utf-8") as log:
        return subprocess.run(command, cwd=cwd, env=env, stdout=log,
                              stderr=subprocess.STDOUT, check=False).returncode


def run_lab(*, train_corpus: Path, held_out: Path, weights_dir: Path,
            config: Path, rounds: int = 3, minutes_per_round: int = 1,
            feedback_path: Path | None = None, interactive: bool = False,
            device: str = "cuda", seed: int = 1234, lr: float | None = None,
            out_dir: Path | None = None, runner: Any | None = None) -> int:
    train_corpus = train_corpus.expanduser().resolve()
    held_out = held_out.expanduser().resolve()
    weights_dir = weights_dir.expanduser().resolve()
    config = config.expanduser().resolve()
    out_dir = (out_dir or ROOT / "runs" / "greenlight-lab").expanduser().resolve()
    if rounds < 1:
        raise ValueError("rounds must be at least one")
    if minutes_per_round < 1:
        raise ValueError("minutes-per-round must be at least 1 for bounded lab runs")
    if not train_corpus.is_file():
        raise FileNotFoundError(f"training corpus file does not exist: {train_corpus}")
    if not held_out.exists():
        raise FileNotFoundError(f"held-out path does not exist: {held_out}")
    if not config.is_file():
        raise FileNotFoundError(f"config file does not exist: {config}")
    if weights_dir == R1_WEIGHTS:
        raise ValueError(f"refusing to train the R1 baseline at {weights_dir}")
    if feedback_path is not None:
        feedback_path = feedback_path.expanduser().resolve()
        if not feedback_path.is_file():
            raise FileNotFoundError(f"feedback JSONL does not exist: {feedback_path}")

    out_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["GREENLIGHT_CONFIG"] = str(config)
    env["PYTHONHASHSEED"] = str(seed)
    execute = runner or _run
    reviewed = read_feedback(feedback_path)

    for round_no in range(1, rounds + 1):
        started = time.time()
        round_corpus = out_dir / f"train_round_{round_no}.txt"
        shutil.copyfile(train_corpus, round_corpus)
        review_file = write_feedback(
            out_dir / f"approved-feedback-round-{round_no:02d}.txt", reviewed)
        if review_file is not None:
            with round_corpus.open("a", encoding="utf-8") as output, \
                    review_file.open(encoding="utf-8") as additions:
                output.write("\n")
                shutil.copyfileobj(additions, output)

        command = [sys.executable, "train.py", "--device", device,
                "read", str(round_corpus),
                    "--weights-dir", str(weights_dir), "--held-out", str(held_out),
                   "--save", "--minutes", str(minutes_per_round),
                    "--precision", "bf16", "--seed", str(seed),
                   "--history", str(out_dir / "expert_history.jsonl"),
                   "--sample-log", str(out_dir / "samples.txt"), "--no-plots"]
        if lr is not None:
            command.extend(("--lr", str(lr)))
        train_log = out_dir / f"train_round_{round_no}.log"
        result = execute(command, ROOT, env, train_log)
        if result:
            raise RuntimeError(f"training round {round_no} failed ({result}); see {train_log}")

        report_path = out_dir / f"eval_round_{round_no}.json"
        evaluate = [sys.executable, "-m", "minagi.evaluate", "--weights",
                    str(weights_dir), "--output", str(report_path),
                    "--device", device, "--precision", "bf16"]
        eval_log = out_dir / f"eval_round_{round_no}.log"
        result = execute(evaluate, ROOT, env, eval_log)
        if result:
            raise RuntimeError(f"evaluation round {round_no} failed ({result}); see {eval_log}")

        report = json.loads(report_path.read_text(encoding="utf-8"))
        matches = [item["exact_match"] for item in report.get("results", [])
                   if "exact_match" in item]
        accuracy = sum(matches) / len(matches) if matches else None
        print(f"round {round_no}/{rounds}: step={report.get('step')} "
              f"params={report.get('params')} exact-match="
              f"{accuracy if accuracy is not None else 'n/a'} "
              f"elapsed={time.time() - started:.1f}s report={report_path}")

        if interactive and round_no < rounds:
            reviewed.extend(collect_interactive_feedback(report_path))
    print(f"Lab complete: weights={weights_dir}; reports/logs={out_dir}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Bounded Greenlight training/evaluation with reviewed feedback",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("train_corpus", type=Path, help="Base training text file")
    parser.add_argument("held_out", type=Path, help="Held-out file or directory")
    parser.add_argument("--weights-dir", type=Path, default=ROOT / "greenlight-16g-lab",
                        help="Separate lab checkpoint; protects R1")
    parser.add_argument("--config", type=Path, default=ROOT / "config-16gb.yaml")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--minutes-per-round", type=int, default=1)
    parser.add_argument("--feedback", type=Path, help="Reviewed JSONL prompt/target pairs")
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "runs" / "greenlight-lab")
    args = parser.parse_args(argv)
    try:
        return run_lab(train_corpus=args.train_corpus, held_out=args.held_out,
                       weights_dir=args.weights_dir, config=args.config,
                       rounds=args.rounds, minutes_per_round=args.minutes_per_round,
                       feedback_path=args.feedback, interactive=args.interactive,
                       device=args.device, seed=args.seed, lr=args.lr,
                       out_dir=args.out_dir)
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        parser.error(str(error))
    return 2


if __name__ == "__main__":
    sys.exit(main())
