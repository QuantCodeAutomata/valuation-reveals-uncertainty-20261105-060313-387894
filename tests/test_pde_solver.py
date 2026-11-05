"""Unit tests for the G-BS nonlinear PDE finite-difference solver
(paper Eq. (3.3), Remark 3.1)."""
from __future__ import annotations

import numpy as np
import pytest

from src.gbs import bs_put_price
from src.pde_solver import interpolate_at, make_grid, solve_gbs_pde

SIGMA_LOW, SIGMA_HIGH = 0.15, 0.35
X0 = 100.0
X_MIN, X_MAX = 1e-3, 500.0


def test_short_time_solution_close_to_initial_condition():
    """As T -> 0, v(T, .) -> f (initial condition), i.e. the PDE evolves
    continuously from its initial data."""
    f = lambda x: np.maximum(100.0 - x, 0.0)
    x_grid, v_grid = solve_gbs_pde(f, SIGMA_LOW, SIGMA_HIGH, 1e-4, X_MIN, X_MAX, 300, 5)
    f_grid = f(x_grid)
    # away from the kink at K=100 the solution should be very close to f
    mask = np.abs(x_grid - 100.0) > 5.0
    assert np.max(np.abs(v_grid[mask] - f_grid[mask])) < 0.5


def test_convex_put_payoff_matches_bs_with_sigma_high():
    """The Black-Scholes-Barenblatt equation with a convex terminal payoff
    (a put) is solved by the Black-Scholes price at the upper volatility
    bound, since v_xx > 0 everywhere for a convex payoff (Section 3.4)."""
    K, T = 100.0, 0.5
    payoff = lambda x: np.maximum(K - x, 0.0)
    x_grid, v_grid = solve_gbs_pde(payoff, SIGMA_LOW, SIGMA_HIGH, T, X_MIN, X_MAX, 400, 200)
    v_at_x0 = interpolate_at(x_grid, v_grid, X0)
    analytic = float(bs_put_price(X0, K, T, SIGMA_HIGH))
    assert v_at_x0 == pytest.approx(analytic, abs=0.05)


def test_pde_value_between_bs_low_and_bs_high_for_general_payoff():
    """For a non-monotone-curvature payoff the PDE value should still lie
    between the two constant-volatility Black-Scholes bounds (sublinearity
    / comparison with the support set, Section 3.2)."""
    K, T = 100.0, 0.5
    payoff = lambda x: np.maximum(K - x, 0.0)
    x_grid, v_grid = solve_gbs_pde(payoff, SIGMA_LOW, SIGMA_HIGH, T, X_MIN, X_MAX, 400, 200)
    v_at_x0 = interpolate_at(x_grid, v_grid, X0)
    low = float(bs_put_price(X0, K, T, SIGMA_LOW))
    high = float(bs_put_price(X0, K, T, SIGMA_HIGH))
    assert low - 1e-6 <= v_at_x0 <= high + 1e-6


def test_make_grid_rejects_too_few_points():
    with pytest.raises(AssertionError):
        make_grid(0.0, 1.0, 2)
