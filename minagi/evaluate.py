"""Run a small repeatable capability probe against a saved model."""

import argparse
import json
import re
from pathlib import Path

import torch

from .recur import load_any
from .tokenizer import ByteTokenizer
from .precision import set_compute_dtype


PROMPTS = [
    {
        "name": "continuation",
        "prompt": "The old observatory stood above the harbor. At dawn, the keeper",
    },
    {
        "name": "instruction",
        "prompt": "<user>\nName three colors. Reply as a comma-separated list.\n</user>\n<bot>\n",
    },
    {
        "name": "arithmetic",
        "prompt": "<user>\nWhat is 37 + 58? Reply with only the number.\n</user>\n<bot>\n",
        "expected": "95",
    },
    {
        "name": "arithmetic-carry",
        "prompt": "<user>\nWhat is 467 + 589? Reply with only the number.\n</user>\n<bot>\n",
        "expected": "1056",
    },
    {
        "name": "arithmetic-subtraction",
        "prompt": "<user>\nWhat is 1000 - 37? Reply with only the number.\n</user>\n<bot>\n",
        "expected": "963",
    },
    {
        "name": "arithmetic-multiplication",
        "prompt": "<user>\nWhat is 17 * 6? Reply with only the number.\n</user>\n<bot>\n",
        "expected": "102",
    },
    {
        "name": "code",
        "prompt": "<user>\nWrite a Python function named clamp(value, low, high).\n</user>\n<bot>\n",
    },
    {
        "name": "explanation",
        "prompt": "<user>\nIn one sentence, explain why ice floats on water.\n</user>\n<bot>\n",
    },
]


def exact_numeric_answer(completion):
    if "</think>" in completion:
        completion = completion.rsplit("</think>", 1)[1]
    lines = [line.strip().strip("`") for line in completion.splitlines() if line.strip()]
    if not lines:
        return None
    answer = lines[-1].strip().rstrip(".")
    return answer if re.fullmatch(r"[+-]?\d+", answer) else None


@torch.inference_mode()
def evaluate_cases(model, device, max_new_tokens=128, cases=PROMPTS):
    tokenizer = ByteTokenizer()
    results = []
    for case in cases:
        prompt_ids = tokenizer.encode(case["prompt"]).ids
        tokens = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        output = model.generate(tokens, max_new_tokens=max_new_tokens)
        completion = tokenizer.decode(output[0, len(prompt_ids):].tolist())
        result = {
            "name": case["name"],
            "prompt": case["prompt"],
            "completion": completion,
        }
        if "expected" in case:
            result["expected"] = case["expected"]
            result["exact_match"] = exact_numeric_answer(completion) == case["expected"]
        results.append(result)
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Save repeatable prompt outputs for comparing mini-AGI checkpoints")
    parser.add_argument("--weights", default="greenlight-16g-r1")
    parser.add_argument("--output", default="runs/greenlight-16g-r1-benchmark.json")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--precision", choices=("bf16", "fp16", "fp32"), default="bf16")
    args = parser.parse_args()
    if args.max_new_tokens <= 0:
        parser.error("--max-new-tokens must be positive")

    device = torch.device(args.device)
    model, metadata = load_any(args.weights, device, read_only=True)
    set_compute_dtype(args.precision)
    results = evaluate_cases(model, device, args.max_new_tokens)
    report = {
        "weights": args.weights,
        "step": metadata.get("step"),
        "params": model.n_params(),
        "device": str(device),
        "precision": args.precision,
        "context": model.cfg.block,
        "max_new_tokens": args.max_new_tokens,
        "results": results,
        "exact_match_accuracy": (
            sum(result["exact_match"] for result in results if "exact_match" in result)
            / max(1, sum("exact_match" in result for result in results))
        ),
        "note": "Qualitative prompt probe, not a standardized capability score.",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")
    for result in results:
        print(f"\n[{result['name']}]\n{result['prompt']}{result['completion']}")
    print(f"\nSaved benchmark outputs to {output}")


if __name__ == "__main__":
    main()
