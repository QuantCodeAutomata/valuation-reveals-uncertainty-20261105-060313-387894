"""Monte Carlo robust valuation under the uncertainty structure U(G) and
verification of the Lyapunov condition (paper Sections 3.3-3.5; exp_3).

Custom implementation: simulates the admissible state-process laws
P_x(L^gamma) for the G-BS support set A(x) = {(0, 0, Sigma): x^2 sigma_low(x)^2
<= Sigma <= x^2 sigma_high(x)^2} by Euler-Maruyama discretisation of
dX_t = sigma(t, X_t) X_t dW_t for admissible volatility processes sigma(t, x)
in [sigma_low(x), sigma_high(x)], and searches the supremum over this family
(Eq. (3.13)) by direct evaluation over a parametrised family of admissible
volatility processes (constant, and state-dependent threshold switching), as
suggested in the paper's numerical notes for this experiment.
"""
from __future__ import annotations

from typing import Callable, Sequence, Union

import numpy as np

from src.gbs import VolSpec, _as_callable

Number = Union[float, np.ndarray]


def simulate_terminal_state(
    x0: float,
    vol_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
    T: float,
    n_steps: int,
    n_paths: int,
    seed: int,
) -> np.ndarray:
    """Euler-Maruyama simulation of dX_t = vol_func(t, X_t) * X_t * dW_t.

    vol_func(t, x) returns the instantaneous volatility at (t, x); it may
    depend on both arguments to encode a state-dependent admissible process.
    Returns the array of terminal values X_T (size n_paths); paths that would
    cross into the absorbing boundary of D = (0, infinity) are floored at a
    small positive value (the exit time tau_exp is a.s. infinite for this
    driftless multiplicative dynamics on the untruncated domain, Section 3.4
    discussion, so the floor is a numerical safeguard only).
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    sqrt_dt = np.sqrt(dt)
    X = np.full(n_paths, x0, dtype=float)
    for step in range(n_steps):
        t = step * dt
        Z = rng.standard_normal(n_paths)
        sigma_t = vol_func(t, X)
        X = X * (1.0 + sigma_t * sqrt_dt * Z)
        np.maximum(X, 1e-8, out=X)
    return X


def constant_vol_valuation(
    payoff: Callable[[np.ndarray], np.ndarray],
    x0: float,
    sigma: float,
    T: float,
    n_steps: int,
    n_paths: int,
    seed: int,
) -> dict:
    """E[payoff(X_T)] under constant admissible volatility sigma, with Monte
    Carlo standard error std(payoff(X_T)) / sqrt(n_paths)."""
    vol_func = lambda t, x: np.full_like(x, sigma)
    X_T = simulate_terminal_state(x0, vol_func, T, n_steps, n_paths, seed)
    values = payoff(X_T)
    return {"mean": float(np.mean(values)), "std_error": float(np.std(values) / np.sqrt(n_paths))}


def threshold_vol_valuation(
    payoff: Callable[[np.ndarray], np.ndarray],
    x0: float,
    sigma_low: VolSpec,
    sigma_high: VolSpec,
    threshold: float,
    high_above: bool,
    T: float,
    n_steps: int,
    n_paths: int,
    seed: int,
) -> dict:
    """E[payoff(X_T)] under the state-dependent admissible volatility process
    that switches between sigma_low(x) and sigma_high(x) at `threshold`."""
    sigma_low_f = _as_callable(sigma_low)
    sigma_high_f = _as_callable(sigma_high)

    def vol_func(t, x):
        above = x > threshold
        s_high = np.asarray(sigma_high_f(x), dtype=float)
        s_low = np.asarray(sigma_low_f(x), dtype=float)
        if high_above:
            return np.where(above, s_high, s_low)
        return np.where(above, s_low, s_high)

    X_T = simulate_terminal_state(x0, vol_func, T, n_steps, n_paths, seed)
    values = payoff(X_T)
    return {"mean": float(np.mean(values)), "std_error": float(np.std(values) / np.sqrt(n_paths))}


def supremum_robust_valuation(
    payoff: Callable[[np.ndarray], np.ndarray],
    x0: float,
    sigma_low: VolSpec,
    sigma_high: VolSpec,
    T: float,
    n_steps: int,
    n_paths: int,
    seed: int,
    thresholds: Sequence[float],
) -> dict:
    """Approximate sup_{(A,Q) in U_x(G)} E^Q[payoff(X_T)] (Eq. (3.13), with
    no discounting ambiguity since C = 0 for the G-BS support set) by
    maximising over constant-volatility and threshold-switching admissible
    processes.

    For convex payoffs the supremum is attained at the constant volatility
    sigma_high (G2 picks out the upper bound whenever the local curvature is
    positive); for concave payoffs at sigma_low; the threshold search also
    covers payoffs that are convex on part of the domain and concave on the
    rest.
    """
    sigma_low_f = _as_callable(sigma_low)
    sigma_high_f = _as_callable(sigma_high)
    candidates = []

    # Constant-volatility candidates evaluated at representative levels.
    for sigma_const in (float(sigma_low_f(x0)), float(sigma_high_f(x0))):
        res = constant_vol_valuation(payoff, x0, sigma_const, T, n_steps, n_paths, seed)
        candidates.append({"type": "constant", "sigma": sigma_const, **res})

    # State-dependent threshold-switching candidates.
    for threshold in thresholds:
        for high_above in (True, False):
            res = threshold_vol_valuation(
                payoff, x0, sigma_low, sigma_high, threshold, high_above, T, n_steps, n_paths, seed
            )
            candidates.append(
                {"type": "threshold", "threshold": threshold, "high_above": high_above, **res}
            )

    best = max(candidates, key=lambda c: c["mean"])
    return {"best": best, "candidates": candidates}


def lyapunov_phi(x: Number) -> Number:
    """phi(x) = x^2 + x^-1, the Lyapunov function for the G-BS case
    (Remark 3.1 / exp_3 methodology Step 7)."""
    x = np.asarray(x, dtype=float)
    return x**2 + 1.0 / x


def lyapunov_phi_derivatives(x: Number):
    """phi'(x) = 2x - x^-2, phi''(x) = 2 + 2x^-3 (exact derivatives of phi)."""
    x = np.asarray(x, dtype=float)
    phi_prime = 2.0 * x - 1.0 / x**2
    phi_double_prime = 2.0 + 2.0 / x**3
    return phi_prime, phi_double_prime


def check_lyapunov_condition(x_grid: np.ndarray, sigma_high: VolSpec) -> dict:
    """Verify Assumption 3.3: G(x, phi, phi', phi'') <= C * phi(x) for all x,
    with C = sigma_high^2 as derived analytically in exp_3 methodology Step 7
    (phi'' = 2 + 2/x^3 > 0 for x > 0, so the G-BS generating function picks
    sigma_high at every grid point)."""
    from src.gbs import gbs_generating_function

    sigma_high_f = _as_callable(sigma_high)
    phi = lyapunov_phi(x_grid)
    phi_prime, phi_double_prime = lyapunov_phi_derivatives(x_grid)
    g_values = gbs_generating_function(x_grid, phi, phi_prime, phi_double_prime, 0.0, sigma_high)
    sigma_high_sq = np.asarray(sigma_high_f(x_grid), dtype=float) ** 2
    rhs = sigma_high_sq * phi
    holds = np.all(g_values <= rhs + 1e-8)
    max_violation = float(np.max(g_values - rhs))
    return {
        "holds": bool(holds),
        "max_violation": max_violation,
        "C_used": float(np.max(sigma_high_sq)),
    }
