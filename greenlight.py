#!/usr/bin/env python3
"""One local entry point for Greenlight training, checks and comparisons."""
import argparse
import importlib.util
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def positive(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError('must be finite and greater than zero')
    return number


def run(command, env):
    print('+ ' + ' '.join(map(str, command)), flush=True)
    subprocess.run([sys.executable, *map(str, command)], cwd=ROOT, env=env, check=True)


def separate(a, b):
    return a != b and a not in b.parents and b not in a.parents


def validate_data(train, held_out):
    from minagi.ingest import collect
    training = {Path(p).resolve() for p in collect([str(train)], cache=None)}
    validation = {Path(p).resolve() for p in collect([str(held_out)], cache=None)}
    if not training or not validation:
        raise ValueError('training and held-out paths must each contain readable text')
    if training & validation:
        raise ValueError('training and held-out files overlap; use a separate validation set')


def train_command(args, weights, output):
    return [ROOT / 'train.py', '--device', args.device, 'read', args.train,
            '--held-out', args.held_out, '--weights-dir', weights, '--save',
            '--minutes', args.minutes, '--passes', args.passes, '--seed', args.seed,
            '--precision', args.precision, '--no-plots',
            '--history', output / 'expert_history.jsonl',
            '--sample-log', output / 'samples.txt']


def benchmark_command(args, weights, output):
    return ['-m', 'minagi.benchmark', '--weights', weights, '--config', args.config,
            '--output', output, '--device', args.device, '--precision', args.precision,
            '--seed', args.seed, '--max-new-tokens', args.max_new_tokens,
            '--held-out', args.held_out]


def doctor(args):
    ok = True
    print('Python:', sys.version.split()[0])
    for module, package in [('torch', 'torch'), ('numpy', 'numpy'), ('yaml', 'PyYAML')]:
        found = importlib.util.find_spec(module) is not None
        print(f'{package}: {"installed" if found else "MISSING"}')
        ok &= found
    if importlib.util.find_spec('torch'):
        import torch
        print('PyTorch:', torch.__version__, '| CUDA:', torch.cuda.is_available())
        if torch.cuda.is_available():
            print('GPU:', torch.cuda.get_device_name(0))
            print('BF16:', torch.cuda.is_bf16_supported())
        elif args.device == 'cuda':
            ok = False
            print('CUDA training unavailable; use --device cpu only for small smoke tests.')
    print('Config:', args.config, '|', 'found' if args.config.is_file() else 'MISSING')
    return 0 if ok and args.config.is_file() else 1


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ['chat']:
        options = argv[1:]
        if options[:1] == ['--']:
            options = options[1:]
        return subprocess.call([sys.executable, '-m', 'mini_agent', *options], cwd=ROOT)
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('doctor', 'train', 'compare'):
        p = sub.add_parser(name)
        p.add_argument('--config', type=Path, default=ROOT / 'config-16gb.yaml')
        p.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
        if name == 'doctor':
            p.add_argument('--weights', type=Path,
                           help='optional checkpoint to verify and resume-preflight')
            p.add_argument('--history', type=Path,
                           help='optional training history.jsonl to diagnose')
            p.add_argument('--expert-history', type=Path,
                           help='optional expert_history.jsonl to diagnose routing')
            p.add_argument('--samples', type=Path,
                           help='optional samples.txt to diagnose generation')
            continue
        p.add_argument('--train', type=Path, default=ROOT / 'data/train')
        p.add_argument('--held-out', type=Path, default=ROOT / 'data/val')
        p.add_argument('--minutes', type=positive, default=6)
        p.add_argument('--passes', type=int, default=1)
        p.add_argument('--seed', type=int, default=1234)
        p.add_argument('--precision', choices=('fp32', 'bf16', 'fp16'), default='bf16')
        p.add_argument('--out', type=Path, required=True, help='new directory for this run')
        p.add_argument('--max-new-tokens', type=int, default=128)
        if name == 'train':
            p.add_argument('--weights', type=Path, default=ROOT / 'greenlight-16g-r1')
        else:
            p.add_argument('--old', type=Path, required=True, help='existing frozen checkpoint')
            p.add_argument('--new', type=Path, required=True, help='new candidate directory')
    p = sub.add_parser('generate', help='generate directly from your Greenlight checkpoint')
    p.add_argument('--weights', type=Path, required=True)
    p.add_argument('--prompt', required=True)
    p.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    p.add_argument('--max-new-tokens', type=int, default=128)
    p.add_argument('--precision', choices=('fp32', 'bf16'), default='bf16')
    p = sub.add_parser('chat', help='local Ollama assistant (not Greenlight weights)')
    p.add_argument('options', nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    for key, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, key, value.expanduser().resolve())
    try:
        if args.command == 'generate':
            if args.max_new_tokens < 1 or not args.prompt:
                raise ValueError('provide a nonempty prompt and positive max-new-tokens')
            import torch
            from minagi.recur import load_any
            from minagi.evaluate import evaluate_cases
            from minagi.precision import set_compute_dtype
            set_compute_dtype(args.precision)
            model, _ = load_any(str(args.weights), torch.device(args.device), read_only=True)
            result = evaluate_cases(model, torch.device(args.device), args.max_new_tokens,
                                    [{'name': 'cli', 'prompt': args.prompt}])
            print(result[0]['completion'])
            return 0
        if args.command == 'doctor':
            status = doctor(args)
            from minagi.training_doctor import (
                diagnose_experts, diagnose_file, diagnose_generation,
                parse_samples, read_history, resume_preflight,
            )
            from minagi.verify_model import inspect_model_artifact
            checks = []
            if args.weights:
                inspect_model_artifact(args.weights)
                checks.append(("resume", resume_preflight(args.weights)))
            if args.history:
                checks.append(("history", diagnose_file(args.history)))
            if args.expert_history:
                checks.append(("experts", diagnose_experts(
                    read_history(args.expert_history))))
            if args.samples:
                checks.append(("samples", diagnose_generation(
                    parse_samples(args.samples))))
            for name, report in checks:
                print(f'{name}: {report["health"]}')
                for finding in report["findings"]:
                    print(f'  [{finding["severity"]}] {finding["code"]}: '
                          f'{finding["message"]}')
                if report["health"] == "critical":
                    status = 1
            return status
        if not args.config.is_file():
            raise ValueError(f'config does not exist: {args.config}')
        if args.passes < 1 or args.max_new_tokens < 1:
            raise ValueError('passes and max-new-tokens must be positive')
        validate_data(args.train, args.held_out)
        if doctor(args):
            raise ValueError('resolve the missing runtime requirements before training')
        env = os.environ.copy()
        env['GREENLIGHT_CONFIG'] = str(args.config)
        env['MINI_AGI_CONFIG'] = str(args.config)
        env['PYTHONHASHSEED'] = str(args.seed)
        from minagi.verify_model import inspect_model_artifact
        weights = args.weights if args.command == 'train' else args.new
        paths = [args.train, args.held_out, weights]
        if args.command == 'compare':
            inspect_model_artifact(args.old)
            paths.append(args.old)
            if any(not separate(args.old, p) for p in (args.train, args.held_out)):
                raise ValueError('baseline checkpoint must be separate from data directories')
            if args.new.exists():
                raise ValueError('candidate already exists; choose a new --new directory')
            if not separate(args.old, args.new):
                raise ValueError('old and new checkpoints must be separate directories')
        if any(not separate(args.out, p) for p in paths):
            raise ValueError('output must be separate from data and checkpoint directories')
        if any(not separate(weights, p) for p in (args.train, args.held_out)):
            raise ValueError('checkpoint must be separate from data directories')
        args.out.mkdir(parents=True, exist_ok=False)
        if args.command == 'compare':
            run(benchmark_command(args, args.old, args.out / 'old.json'), env)
            shutil.copytree(args.old, args.new)
        run(train_command(args, weights, args.out), env)
        inspect_model_artifact(weights)
        command = benchmark_command(args, weights, args.out / 'new.json')
        if args.command == 'compare':
            command += ['--compare', args.out / 'old.json']
        run(command, env)
        print(f'Complete. Checkpoint: {weights}\nReport: {args.out / "new.json"}')
        return 0
    except (OSError, ValueError, RuntimeError, ImportError, subprocess.CalledProcessError) as error:
        print(f'Greenlight: {error}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
