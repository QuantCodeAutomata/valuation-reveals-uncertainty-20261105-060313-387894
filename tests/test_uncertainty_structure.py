"""Tests for exp_3: Monte Carlo robust valuation and the Lyapunov condition
(paper Sections 3.3-3.5)."""
from __future__ import annotations

import numpy as np

from src.gbs import bs_call_price, bs_put_price
from src.uncertainty_structure import (
    check_lyapunov_condition,
    constant_vol_valuation,
    lyapunov_phi,
    lyapunov_phi_derivatives,
    simulate_terminal_state,
    supremum_robust_valuation,
)

SIGMA_LOW, SIGMA_HIGH = 0.15, 0.35
X0 = 100.0


def test_simulate_terminal_state_is_a_positive_martingale_on_average():
    """Driftless geometric dynamics: E[X_T] should stay close to X0 (no drift,
    B=0 in the G-BS support set, Section 3.3)."""
    vol_func = lambda t, x: np.full_like(x, 0.2)
    X_T = simulate_terminal_state(X0, vol_func, 1.0, 250, 50000, seed=0)
    assert np.all(X_T > 0)
    assert abs(np.mean(X_T) - X0) < 0.05 * X0


def test_constant_vol_valuation_matches_bs_price():
    """Monte Carlo valuation under a constant admissible volatility should
    match the Black-Scholes closed form within a few standard errors."""
    K, T, sigma = 100.0, 0.5, 0.25
    payoff = lambda x: np.maximum(K - x, 0.0)
    result = constant_vol_valuation(payoff, X0, sigma, T, 200, 50000, seed=1)
    analytic = float(bs_put_price(X0, K, T, sigma))
    assert abs(result["mean"] - analytic) < 6 * result["std_error"]


def test_supremum_for_convex_payoff_is_attained_near_sigma_high():
    """For a convex payoff (call), the robust supremum over admissible
    volatility processes should be close to the sigma_high Black-Scholes
    price (G2: positive curvature picks the upper bound, Remark 3.1)."""
    K, T = 100.0, 0.5
    payoff = lambda x: np.maximum(x - K, 0.0)
    out = supremum_robust_valuation(
        payoff, X0, SIGMA_LOW, SIGMA_HIGH, T, 150, 20000, seed=2, thresholds=[90.0, 100.0, 110.0]
    )
    analytic_high = float(bs_call_price(X0, K, T, SIGMA_HIGH))
    analytic_low = float(bs_call_price(X0, K, T, SIGMA_LOW))
    assert out["best"]["mean"] >= analytic_low
    assert abs(out["best"]["mean"] - analytic_high) < 0.05 * analytic_high + 3 * out["best"]["std_error"]


def test_lyapunov_phi_derivatives_are_exact():
    x = np.array([10.0, 50.0, 100.0])
    phi_prime, phi_double_prime = lyapunov_phi_derivatives(x)
    h = 1e-5
    numeric_prime = (lyapunov_phi(x + h) - lyapunov_phi(x - h)) / (2 * h)
    numeric_double = (lyapunov_phi(x + h) - 2 * lyapunov_phi(x) + lyapunov_phi(x - h)) / h**2
    np.testing.assert_allclose(phi_prime, numeric_prime, atol=1e-3)
    np.testing.assert_allclose(phi_double_prime, numeric_double, atol=1e-2)


def test_lyapunov_condition_holds_with_C_equal_sigma_high_sq():
    x_grid = np.linspace(1e-3, 500.0, 2000)
    out = check_lyapunov_condition(x_grid, SIGMA_HIGH)
    assert out["holds"]
    assert out["C_used"] == SIGMA_HIGH**2
