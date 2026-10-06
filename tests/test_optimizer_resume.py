import unittest

import torch

from minagi.store import _stamp_missing_steps


class OptimizerResumeTests(unittest.TestCase):
    def test_restored_moments_without_step_can_resume_adamw(self):
        """Regression for the restart crash: restored Adam moments need a step."""
        p = torch.nn.Parameter(torch.tensor([1.0, -1.0]))
        opt = torch.optim.AdamW([p], lr=1e-3)

        # Simulate a checkpoint restore where one parameter has complete Adam
        # moments and another restored state supplies the optimiser's counter.
        anchor = torch.nn.Parameter(torch.tensor([0.0]))
        opt.add_param_group({"params": [anchor]})
        opt.state[anchor]["step"] = torch.tensor(37.0)
        opt.state[anchor]["exp_avg"] = torch.zeros_like(anchor)
        opt.state[anchor]["exp_avg_sq"] = torch.zeros_like(anchor)

        opt.state[p]["exp_avg"] = torch.zeros_like(p)
        opt.state[p]["exp_avg_sq"] = torch.zeros_like(p)
        self.assertNotIn("step", opt.state[p])

        _stamp_missing_steps(opt)

        self.assertEqual(float(opt.state[p]["step"]), 37.0)
        p.grad = torch.ones_like(p)
        anchor.grad = torch.zeros_like(anchor)
        opt.step()  # Used to raise KeyError: 'step' after restart.
        self.assertEqual(float(opt.state[p]["step"]), 38.0)

    def test_no_counter_does_not_invent_optimizer_age(self):
        p = torch.nn.Parameter(torch.ones(2))
        opt = torch.optim.AdamW([p], lr=1e-3)
        opt.state[p]["exp_avg"] = torch.zeros_like(p)
        opt.state[p]["exp_avg_sq"] = torch.zeros_like(p)

        _stamp_missing_steps(opt)

        self.assertNotIn("step", opt.state[p])


if __name__ == "__main__":
    unittest.main()
