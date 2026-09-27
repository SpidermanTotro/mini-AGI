import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import greenlight
from minagi.benchmark import compare_reports
from mini_agent.ollama import OllamaClient


class GreenlightCliTests(unittest.TestCase):
    def test_chat_forwards_options_in_original_order(self):
        with patch('greenlight.subprocess.call', return_value=0) as call:
            self.assertEqual(greenlight.main(['chat', '--model', 'qwen3:8b', '--workspace', '/tmp']), 0)
        self.assertEqual(call.call_args.args[0][-4:], ['--model', 'qwen3:8b', '--workspace', '/tmp'])

    def test_rejects_nonfinite_training_duration(self):
        for value in ('nan', 'inf', '-1', '0'):
            with self.subTest(value=value), self.assertRaises(Exception):
                greenlight.positive(value)

    def test_data_overlap_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'text.txt'
            path.write_text('training and validation must be separate')
            with self.assertRaisesRegex(ValueError, 'overlap'):
                greenlight.validate_data(Path(tmp), path)

    def test_comparison_preserves_baseline_and_uses_matching_settings(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old = root / 'old'
            old.mkdir()
            (old / 'original').write_text('baseline')
            config = root / 'config.yaml'
            config.write_text('model: {}')
            commands = []
            def execute(command, env):
                commands.append(command)
                self.assertEqual(env['GREENLIGHT_CONFIG'], str(config))
                if 'read' in command:
                    (root / 'new' / 'original').write_text('trained candidate')
            with patch('greenlight.validate_data'), patch('greenlight.doctor', return_value=0), \
                 patch('minagi.verify_model.inspect_model_artifact'), \
                 patch('greenlight.run', side_effect=execute):
                status = greenlight.main(['compare', '--old', str(old), '--new', str(root/'new'),
                     '--config', str(config), '--out', str(root/'reports'), '--device', 'cpu'])
            self.assertEqual(status, 0)
            self.assertEqual((old/'original').read_text(), 'baseline')
            self.assertEqual((root/'new'/'original').read_text(), 'trained candidate')
            self.assertEqual(len(commands), 3)
            self.assertIn('--compare', commands[-1])

    def test_mismatched_comparison_does_not_claim_accuracy_gain(self):
        baseline = {'model': {'context': 128}, 'summary': {'exact_match_accuracy': 0.0}}
        candidate = {'model': {'context': 256}, 'summary': {'exact_match_accuracy': 1.0}}
        report = compare_reports(baseline, candidate)
        self.assertFalse(report['protocol_matches'])
        self.assertIsNone(report['exact_match_accuracy_delta'])

    def test_ollama_timeout_becomes_actionable_error(self):
        with patch('mini_agent.ollama.urlopen', side_effect=TimeoutError):
            with self.assertRaisesRegex(RuntimeError, 'Cannot reach Ollama'):
                OllamaClient().chat([], [])

    def test_benchmark_main_activates_explicit_config(self):
        from minagi import benchmark
        class StopAfterConfig(Exception):
            pass
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp)/'profile.yaml'
            config.write_text('model: {}')
            def check(*args, **kwargs):
                self.assertEqual(os.environ['GREENLIGHT_CONFIG'], str(config))
                raise StopAfterConfig
            with patch.dict(os.environ), patch('sys.argv', ['benchmark', '--device', 'cpu', '--config', str(config)]), \
                 patch.object(benchmark, 'load_any', side_effect=check):
                with self.assertRaises(StopAfterConfig):
                    benchmark.main()


class HeldOutFileTests(unittest.TestCase):
    def test_evaluator_accepts_single_file(self):
        from minagi.stream import FolderEvaluator
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'validation.txt'
            path.write_text('A separate validation sentence.')
            evaluator = FolderEvaluator(None, str(path), 16, 32, 'cpu')
            self.assertEqual(evaluator.groups, {'all': [str(path)]})
