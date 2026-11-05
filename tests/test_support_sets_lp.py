"""Tests for exp_2: LP-based numerical support-set construction against the
analytic G-BS support set (paper Section 3.2, Eq. (3.5))."""
from __future__ import annotations

import numpy as np

from src.gbs import gbs_generating_function, gbs_support_set_sigma_bounds
from src.support_sets import support_set_lp, w_grid

SIGMA_LOW, SIGMA_HIGH = 0.15, 0.35


def test_lp_support_set_matches_analytic_bounds():
    x = 100.0
    W = w_grid(bound=10.0, n_points=15)
    G_func = lambda x_, r, p, H: float(gbs_generating_function(x_, r, p, H, SIGMA_LOW, SIGMA_HIGH))
    out = support_set_lp(x, G_func, W)
    assert out["feasible"]
    sigma_sq_low, sigma_sq_high = gbs_support_set_sigma_bounds(x, SIGMA_LOW, SIGMA_HIGH)
    assert out["sigma_min"] == pytest_approx(float(sigma_sq_low))
    assert out["sigma_max"] == pytest_approx(float(sigma_sq_high))


def pytest_approx(value, rel=1e-2):
    class _Approx:
        def __eq__(self, other):
            return abs(other - value) <= rel * max(abs(value), 1.0)

    return _Approx()
