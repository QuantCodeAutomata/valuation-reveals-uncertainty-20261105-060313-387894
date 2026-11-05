"""Finite-difference solver for the G-BS nonlinear valuation PDE.

Custom implementation (paper Eq. (3.3), specialised to the G-BS generating
function of Remark 3.1):

    d/dt v = G(x, v, d/dx v, d^2/dx^2 v)
           = 1/2 x^2 (sigma_high(x)^2 max(v_xx, 0) + sigma_low(x)^2 max(-v_xx, 0)),
    v(0, x) = f(x),

on the truncated domain D_n ~ [x_min, x_max]. This is the Black-Scholes-
Barenblatt / uncertain-volatility equation of Avellaneda, Levy and Paras
(1995); no function in scipy or the other listed quantitative libraries
solves this nonlinear, solution-dependent-diffusion PDE, so it is discretised
directly. The scheme is implicit (backward Euler) in time with central
differences in space; the nonlinearity max(v_xx, 0) / max(-v_xx, 0) is
resolved by Picard (fixed-point) iteration on the sign of v_xx at each time
step, as suggested in the paper's numerical notes for this experiment.
Boundary conditions are Dirichlet, held fixed at the initial condition's
boundary values (the standard "payoff value" truncation used in finite-
difference option-pricing schemes), since G-BS has no discounting (C = 0)
or drift (B = 0) term that would otherwise move the solution away from the
payoff at the edges of a sufficiently wide truncated domain.
"""
from __future__ import annotations

from typing import Callable, Union

import numpy as np
from scipy.linalg import solve_banded

Number = Union[float, np.ndarray]
VolSpec = Union[float, Callable[[Number], Number]]


def _as_array_callable(sigma: VolSpec):
    if callable(sigma):
        return lambda x: np.asarray(sigma(x), dtype=float)
    value = float(sigma)
    return lambda x: np.full_like(np.asarray(x, dtype=float), value)


def make_grid(x_min: float, x_max: float, n_x: int) -> np.ndarray:
    """Uniform spatial grid of n_x points on [x_min, x_max]."""
    assert n_x >= 5, "n_x must be large enough to resolve second derivatives"
    return np.linspace(x_min, x_max, n_x)


def solve_gbs_pde(
    v0_func: Callable[[np.ndarray], np.ndarray],
    sigma_low: VolSpec,
    sigma_high: VolSpec,
    T: float,
    x_min: float,
    x_max: float,
    n_x: int,
    n_t: int,
    picard_tol: float = 1e-10,
    picard_max_iter: int = 50,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve the G-BS PDE up to maturity T and return (x_grid, v(T, x_grid)).

    Parameters
    ----------
    v0_func : initial condition f, vectorised over the spatial grid.
    sigma_low, sigma_high : volatility bounds, constant or callables of x.
    T : maturity of the solve.
    x_min, x_max, n_x : truncated spatial domain D_n and its resolution.
    n_t : number of implicit time steps (dt = T / n_t).
    picard_tol, picard_max_iter : fixed-point iteration controls for the
        solution-dependent diffusion coefficient.
    """
    assert T > 0, "T must be positive"
    assert n_t >= 1, "n_t must be at least 1"
    x = make_grid(x_min, x_max, n_x)
    dx = x[1] - x[0]
    dt = T / n_t

    sigma_low_f = _as_array_callable(sigma_low)
    sigma_high_f = _as_array_callable(sigma_high)
    sigma_low_sq = sigma_low_f(x) ** 2
    sigma_high_sq = sigma_high_f(x) ** 2

    v = np.asarray(v0_func(x), dtype=float).copy()
    v_bc_low, v_bc_high = v[0], v[-1]

    n_interior = n_x - 2
    x_int = x[1:-1]
    half_x2_over_dx2 = 0.5 * x_int**2 / dx**2

    for _ in range(n_t):
        v_old = v.copy()
        v_guess = v.copy()
        for _picard_iter in range(picard_max_iter):
            v_xx = (v_guess[2:] - 2.0 * v_guess[1:-1] + v_guess[:-2]) / dx**2
            # Split the nonlinear diffusion coefficient by the sign of v_xx:
            # positive curvature uses sigma_high, negative curvature sigma_low.
            diffusion_sq = np.where(v_xx >= 0.0, sigma_high_sq[1:-1], sigma_low_sq[1:-1])
            a = half_x2_over_dx2 * diffusion_sq  # a_i = 0.5 x_i^2 D_eff(x_i) / dx^2

            lower = -dt * a
            upper = -dt * a
            diag = 1.0 + 2.0 * dt * a

            rhs = v_old[1:-1].copy()
            rhs[0] -= lower[0] * v_bc_low
            rhs[-1] -= upper[-1] * v_bc_high

            ab = np.zeros((3, n_interior))
            ab[0, 1:] = upper[:-1]
            ab[1, :] = diag
            ab[2, :-1] = lower[1:]
            v_new_interior = solve_banded((1, 1), ab, rhs)

            v_new = np.empty_like(v_guess)
            v_new[0] = v_bc_low
            v_new[-1] = v_bc_high
            v_new[1:-1] = v_new_interior

            if np.max(np.abs(v_new - v_guess)) < picard_tol:
                v_guess = v_new
                break
            v_guess = v_new
        v = v_guess

    return x, v


def interpolate_at(x_grid: np.ndarray, v_grid: np.ndarray, x0: float) -> float:
    """Cubic-spline value of v at x0 (x0 need not coincide with a grid node)."""
    from scipy.interpolate import CubicSpline

    spline = CubicSpline(x_grid, v_grid)
    return float(spline(x0))
