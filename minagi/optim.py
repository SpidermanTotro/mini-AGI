"""
A meter for how much of a gradient is signal.
"""

import torch


class GradSNR:
    """
    How much of the gradient is signal, measured every step.

    The held-out signal this project steers the learning rate by moves 0.0014
    per evaluation against 0.018 of noise, so it takes over a hundred readings
    to say anything - twenty hours at a ten-minute checkpoint. This is the
    same question asked of a quantity that is available on every step.

    Keep an average of the gradient and an average of its squared norm. If
    successive gradients agree, the average keeps its length and the ratio
    ||mean||^2 / mean(||g||^2) approaches one. If they are independent noise
    the average shrinks toward zero and so does the ratio. It is the gradient
    noise scale of McCandlish et al. 2018, in the cheapest form that answers
    the question: two scalars, no extra tensors.

    Reported, not acted on. A signal is watched for a while before anything is
    allowed to steer on it.
    """

    def __init__(self, beta=0.98):
        self.beta = beta
        self.m = None
        self.sq = 0.0
        self.n = 0
        self._grad_signature = None

    @torch.no_grad()
    def observe(self, params):
        active = [(i, p.grad) for i, p in enumerate(params) if p.grad is not None]
        if not active:
            return None
        signature = tuple((i, tuple(g.shape)) for i, g in active)
        flat = torch.cat([g.detach().float().reshape(-1) for _, g in active])

        # Recurrence and conditional paths can change which parameters receive
        # gradients from one step to the next.  An EMA only has meaning while
        # its coordinates describe the same parameters, so start a new window
        # whenever that active set changes.  Upstream hit the same wall from
        # the halting head - a few in a million steps, one of which stopped a
        # run 346 minutes in.  The signature covers both that and any swap of
        # equally sized parameters, which a length check alone would miss.
        if self.m is None or signature != self._grad_signature:
            self.m = flat.clone()
            self.sq = float((flat * flat).sum())
            self.n = 1
            self._grad_signature = signature
            return self.ratio()

        self.m.mul_(self.beta).add_(flat, alpha=1 - self.beta)
        s = float((flat * flat).sum())
        self.sq = self.beta * self.sq + (1 - self.beta) * s
        self.n += 1
        return self.ratio()

    def ratio(self):
        """0 = pure noise, 1 = every step pointing the same way."""
        if self.m is None or self.sq <= 0 or self.n < 8:
            return None
        # No bias correction: both averages start at the first reading, not
        # at zero, so neither is biased toward zero. Dividing by 1 - beta^n
        # here inflated the ratio by that factor - 6.7x at the eighth reading,
        # and a constant gradient read above one. (from mini-AGI PR #28,
        # still open upstream at the time of this integration)
        return float(self.m.pow(2).sum() / self.sq)
