import tempfile
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
