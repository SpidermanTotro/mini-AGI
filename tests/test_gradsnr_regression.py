"""Regression tests for conditional-gradient GradSNR telemetry.

Run: python -m pytest -q tests/test_gradsnr_regression.py
"""
import pytest

torch = pytest.importorskip("torch")
from minagi.optim import GradSNR


def _step(meter, params, grads):
    for p, g in zip(params, grads):
        p.grad = None if g is None else torch.tensor(g, dtype=p.dtype)
    return meter.observe(params)


def test_missing_gradient_does_not_crash_or_mix_coordinates():
    params = [torch.nn.Parameter(torch.zeros(2)), torch.nn.Parameter(torch.zeros(3))]
    meter = GradSNR(beta=0.9)
    assert _step(meter, params, [[1, 2], [3, 4, 5]]) is None
    for _ in range(7):
        _step(meter, params, [[1, 2], [3, 4, 5]])
    assert meter.ratio() == pytest.approx(1.0, abs=1e-6)
    # Dropping a conditional branch resets the measurement window.
    assert _step(meter, params, [[1, 2], None]) is None
    assert meter.n == 1
    assert _step(meter, params, [[1, 2], [3, 4, 5]]) is None
    assert meter.n == 1


def test_equal_size_parameter_swap_resets_window():
    params = [torch.nn.Parameter(torch.zeros(2)), torch.nn.Parameter(torch.zeros(2))]
    meter = GradSNR()
    _step(meter, params, [[1, 1], None])
    for _ in range(7):
        _step(meter, params, [[1, 1], None])
    assert meter.ratio() is not None
    # Same flattened length, different parameter: must not mix EMA coordinates.
    assert _step(meter, params, [None, [1, 1]]) is None
    assert meter.n == 1


def test_all_missing_grads_leave_existing_measurement_intact():
    params = [torch.nn.Parameter(torch.zeros(1))]
    meter = GradSNR()
    _step(meter, params, [[2]])
    assert _step(meter, params, [None]) is None
    assert meter.n == 1
    assert _step(meter, params, [[2]]) is None
    assert meter.n == 2


def test_constant_gradient_ratio_is_bounded_at_one():
    params = [torch.nn.Parameter(torch.zeros(3))]
    meter = GradSNR(beta=0.98)
    for _ in range(16):
        _step(meter, params, [[1, 2, 3]])
    assert meter.ratio() == pytest.approx(1.0, abs=1e-6)


def test_zero_gradient_does_not_divide_by_zero():
    params = [torch.nn.Parameter(torch.zeros(2))]
    meter = GradSNR()
    for _ in range(10):
        assert _step(meter, params, [[0, 0]]) is None
