"""Tests for safe AdamW state recovery on checkpoint reload."""
import pytest

torch = pytest.importorskip("torch")
from minagi.store import _stamp_missing_steps


def _optimizer():
    p = torch.nn.Parameter(torch.tensor([1.0]))
    q = torch.nn.Parameter(torch.tensor([2.0]))
    return torch.optim.AdamW([p, q]), p, q


def _moments(opt, p, step=None):
    state = opt.state[p]
    state["exp_avg"] = torch.ones_like(p)
    state["exp_avg_sq"] = torch.ones_like(p)
    if step is not None:
        state["step"] = torch.tensor(float(step))


def test_no_step_provenance_fails_closed_without_mutating_moments():
    opt, p, q = _optimizer()
    _moments(opt, p)
    before = opt.state[p]["exp_avg"].clone()
    with pytest.raises(RuntimeError, match="no step counters"):
        _stamp_missing_steps(opt)
    assert "step" not in opt.state[p]
    assert torch.equal(before, opt.state[p]["exp_avg"])


def test_missing_step_inherits_existing_counter():
    opt, p, q = _optimizer()
    _moments(opt, p, step=12)
    _moments(opt, q)
    _stamp_missing_steps(opt)
    assert float(opt.state[q]["step"]) == pytest.approx(12)
    assert float(opt.state[p]["step"]) == pytest.approx(12)


def test_empty_optimizer_state_is_valid():
    opt, p, q = _optimizer()
    _stamp_missing_steps(opt)
    assert not opt.state[p] and not opt.state[q]


def test_repaired_optimizer_can_step():
    opt, p, q = _optimizer()
    _moments(opt, p, step=4)
    _moments(opt, q)
    _stamp_missing_steps(opt)
    p.grad = torch.ones_like(p)
    q.grad = torch.ones_like(q)
    opt.step()
    assert float(opt.state[q]["step"]) == pytest.approx(5)
