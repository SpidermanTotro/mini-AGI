import unittest

import torch

from minagi.decode import contrastive_generate
from minagi.report import render_report, weight_stats
from minagi.recur import RecurCoder, RecurConfig


def tiny_config(**overrides):
    values = {
        "vocab_size": 37,
        "d_model": 24,
        "n_head": 4,
        "d_ff": 40,
        "block": 12,
        "n_prelude": 1,
        "n_recur": 1,
        "n_coda": 0,
        "max_steps": 3,
        "min_steps": 1,
        "bptt_window": 2,
    }
    values.update(overrides)
    return RecurConfig(**values)


class ModelStressTests(unittest.TestCase):
    def test_rejects_invalid_transformer_and_recurrence_configs(self):
        for overrides in (
            {"d_model": 10, "n_head": 3},
            {"d_model": 15, "n_head": 3},
            {"block": 0},
            {"max_steps": 0},
            {"min_steps": 4, "max_steps": 3},
            {"halt_prior": 0.0},
            {"halt_thresh": 1.1},
            {"bptt_window": -1},
            {"pool_experts": 4, "pool_top_k": 5},
            {"pool_experts": 4, "pool_max": 3},
            {"pool_depth": 0},
        ):
            with self.subTest(overrides=overrides):
                with self.assertRaises(ValueError):
                    tiny_config(**overrides)

    def test_chunked_cache_matches_full_context(self):
        torch.manual_seed(11)
        model = RecurCoder(tiny_config()).eval()
        tokens = torch.randint(model.cfg.vocab_size, (2, 9))
        with torch.no_grad():
            expected = model(tokens)[0]
            caches = model.empty_caches()
            chunks = [
                model(tokens[:, start:start + 3], caches=caches,
                      pos_offset=start)[0]
                for start in range(0, tokens.shape[1], 3)
            ]
        torch.testing.assert_close(torch.cat(chunks, dim=1), expected,
                                   atol=1e-5, rtol=1e-5)

    def test_sampled_depth_stays_fixed_for_live_cache(self):
        model = RecurCoder(tiny_config(max_steps=4, train_steps_mean=1.0)).train()
        samples = iter((4, 1))
        model.sample_depth = lambda: next(samples)
        caches = model.empty_caches()
        for start in (0, 2):
            tokens = torch.randint(model.cfg.vocab_size, (1, 2))
            model(tokens, caches=caches, pos_offset=start)
        self.assertEqual(caches[0]["_depth"], 4)
        self.assertEqual({cache["k"].shape[-2] for cache in caches}, {4})

    def test_training_gradients_and_contrastive_decode(self):
        model = RecurCoder(tiny_config())
        tokens = torch.randint(model.cfg.vocab_size, (2, 6))
        _, loss = model(tokens[:, :-1], targets=tokens[:, 1:])
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(
            param.grad is None or torch.isfinite(param.grad).all()
            for param in model.parameters()))

        generated = contrastive_generate(model, tokens[:1, :3], 2, top_k=2)
        self.assertEqual(tuple(generated.shape), (1, 5))
        for arguments in ((-1, 2, 0.5), (1, 0, 0.5), (1, 2, 1.5)):
            with self.subTest(arguments=arguments):
                with self.assertRaises(ValueError):
                    contrastive_generate(
                        model, tokens[:1, :3], arguments[0],
                        top_k=arguments[1], alpha=arguments[2])

    def test_weight_report_uses_recurrent_block_layout(self):
        model = RecurCoder(tiny_config(n_prelude=1, n_recur=1, n_coda=1,
                                       max_steps=2))
        stats = weight_stats(model)
        self.assertEqual(stats["n_layer"], 5)
        self.assertEqual(stats["n_block"], 3)
        self.assertIn("L5", render_report(stats))

    def test_shared_pool_forward_and_backward(self):
        model = RecurCoder(tiny_config(
            n_prelude=0, use_pool=True, pool_experts=4, pool_max=4,
            pool_d_ff=8, pool_top_k=2))
        tokens = torch.randint(model.cfg.vocab_size, (2, 5))
        _, loss = model(tokens[:, :-1], targets=tokens[:, 1:])
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(model.pool.gate.grad).all())

    def test_invalid_forward_inputs_fail_at_boundary(self):
        model = RecurCoder(tiny_config())
        tokens = torch.ones((1, 2), dtype=torch.long)
        calls = (
            lambda: model(tokens[0]),
            lambda: model(tokens[:, :0]),
            lambda: model(tokens, pos_offset=-1),
            lambda: model(tokens, caches=[]),
            lambda: model(tokens, targets=tokens[:, :1]),
            lambda: model(tokens, targets=tokens.to(torch.int32)),
        )
        for call in calls:
            with self.subTest(call=call):
                with self.assertRaises(ValueError):
                    call()


if __name__ == "__main__":
    unittest.main()
