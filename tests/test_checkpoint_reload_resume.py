import json
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest

import torch

from minagi.recur import RecurCoder, RecurConfig
from minagi import store


class CheckpointReloadResumeTests(unittest.TestCase):
    def _model(self):
        cfg = RecurConfig(
            vocab_size=265, d_model=8, n_head=1, d_ff=16, block=8,
            n_prelude=1, n_recur=1, n_coda=0, max_steps=1,
            use_pool=True, pool_experts=2, pool_d_ff=8, pool_top_k=1,
            pool_max=2,
        )
        return RecurCoder(cfg), cfg

    def _step(self, model, opt):
        tokens = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
        loss = model(tokens)[0].float().square().mean()
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)

    def test_save_reload_restores_adam_state_and_can_step(self):
        """R5 regression: prove resume across the on-disk checkpoint boundary."""
        model, cfg = self._model()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        self._step(model, opt)

        with tempfile.TemporaryDirectory() as tmp:
            store.save(model, tmp, step=1, val=1.0, opt=opt,
                       cfg=cfg.__dict__)

            restarted, _ = self._model()
            restarted_opt = torch.optim.AdamW(restarted.parameters(), lr=1e-3)
            store.load(restarted, tmp, opt=restarted_opt, device="cpu")

            restored = [state for state in restarted_opt.state.values()
                        if "exp_avg" in state]
            self.assertTrue(restored, "checkpoint restored no Adam moments")
            self.assertTrue(all("exp_avg_sq" in state for state in restored))
            self.assertTrue(all("step" in state for state in restored))
            self.assertTrue(all(float(state["step"]) == 1.0
                                for state in restored))

            # This is the operation that used to fail only after a restart.
            self._step(restarted, restarted_opt)
            stepped = [float(state["step"]) for state
                       in restarted_opt.state.values() if "step" in state]
            self.assertTrue(stepped)
            self.assertTrue(all(step == 2.0 for step in stepped))


    def test_cold_process_reload_resumes_evaluates_and_generates(self):
        """R5 gate: a separate Python process can cold-load and keep working."""
        model, cfg = self._model()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        self._step(model, opt)

        with tempfile.TemporaryDirectory() as tmp:
            store.save(model, tmp, step=1, val=1.0, opt=opt,
                       cfg=cfg.__dict__)

            script = textwrap.dedent(r"""
                import json
                import os
                import sys
                import torch

                from minagi.recur import RecurCoder, RecurConfig
                from minagi import store

                path = sys.argv[1]
                cfg = RecurConfig(
                    vocab_size=265, d_model=8, n_head=1, d_ff=16, block=8,
                    n_prelude=1, n_recur=1, n_coda=0, max_steps=1,
                    use_pool=True, pool_experts=2, pool_d_ff=8, pool_top_k=1,
                    pool_max=2,
                )
                model = RecurCoder(cfg)
                opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
                man, missing, unexpected = store.load(
                    model, path, opt=opt, device="cpu")
                assert not unexpected, unexpected

                tokens = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
                with torch.no_grad():
                    before = model(tokens)[0]
                assert torch.isfinite(before).all()

                loss = model(tokens)[0].float().square().mean()
                loss.backward()
                opt.step()
                opt.zero_grad(set_to_none=True)

                steps = [float(state["step"]) for state in opt.state.values()
                         if "step" in state]
                assert steps and all(step == 2.0 for step in steps), steps

                # Generation smoke: use the same autoregressive forward path
                # without depending on a tokenizer or CLI process.
                generated = tokens.clone()
                with torch.no_grad():
                    for _ in range(2):
                        logits = model(generated[:, -cfg.block:])[0]
                        next_token = logits[:, -1].argmax(dim=-1, keepdim=True)
                        generated = torch.cat((generated, next_token), dim=1)
                assert generated.shape == (1, 6)
                assert torch.isfinite(model(generated[:, -cfg.block:])[0]).all()

                print(json.dumps({
                    "manifest_step": man.get("step"),
                    "optimizer_steps": steps,
                    "generated_tokens": generated.tolist(),
                }))
            """)

            env = os.environ.copy()
            env["PYTHONPATH"] = os.getcwd() + os.pathsep + env.get(
                "PYTHONPATH", "")
            proc = subprocess.run(
                [sys.executable, "-c", script, tmp],
                cwd=os.getcwd(),
                env=env,
                text=True,
                capture_output=True,
                timeout=60,
            )
            self.assertEqual(
                proc.returncode, 0,
                msg=f"cold restart failed:\nSTDOUT:\n{proc.stdout}\n"
                    f"STDERR:\n{proc.stderr}",
            )
            report = json.loads(proc.stdout.strip().splitlines()[-1])
            self.assertEqual(report["manifest_step"], 1)
            self.assertTrue(all(step == 2.0
                                for step in report["optimizer_steps"]))
            self.assertEqual(len(report["generated_tokens"][0]), 6)


class PagedCheckpointReloadResumeTests(unittest.TestCase):
    def test_paged_checkpoint_reloads_and_steps_with_own_expert_moments(self):
        """R5 regression for the path used by continual paged training."""
        from train import build_paged

        seed = RecurCoder(RecurConfig(
            vocab_size=265, d_model=8, n_head=1, d_ff=16, block=8,
            n_prelude=1, n_recur=1, n_coda=0, max_steps=1,
            use_pool=True, pool_experts=2, pool_d_ff=8, pool_top_k=1,
            pool_max=2,
        ))
        cfg = seed.cfg

        with tempfile.TemporaryDirectory() as tmp:
            store.save(seed, tmp, step=0, val=1.0, cfg=cfg.__dict__)

            model, _, pool, _ = build_paged(
                tmp, torch.device("cpu"), resident=1, ram_capacity=2)
            opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
            pool.attach_optimiser(opt)

            tokens = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
            loss = model(tokens)[0].float().square().mean()
            loss.backward()
            # attach_optimiser installs PagedPool's pre/post step hooks, so
            # the production path is simply opt.step().
            opt.step()
            opt.zero_grad(set_to_none=True)
            store.save(model, tmp, step=1, val=1.0, opt=opt,
                       cfg=cfg.__dict__)

            restarted, _, restarted_pool, _ = build_paged(
                tmp, torch.device("cpu"), resident=1, ram_capacity=2)
            restarted_opt = torch.optim.AdamW(
                restarted.parameters(), lr=1e-3)
            restarted_pool.attach_optimiser(restarted_opt)
            store._load_optim(restarted_opt, restarted, tmp)

            loss = restarted(tokens)[0].float().square().mean()
            loss.backward()
            # attach_optimiser owns the expert-moment handoff around this step.
            restarted_opt.step()  # historical restart failure boundary


if __name__ == "__main__":
    unittest.main()
