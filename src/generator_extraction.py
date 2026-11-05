"""Extract the generating function G from the valuation rule by generator
limit (paper Definition 3.2, Eq. (3.1)-(3.2); exp_1).

Custom implementation: solves the nonlinear PDE of src/pde_solver.py for a
sequence of small maturities t_n -> 0 with the bump test function of
src/test_functions.py as initial data, forms the difference quotient
G_approx(t_n) = (v(t_n, x0) - r) / t_n, and Richardson-extrapolates the
sequence to t -> 0.
"""
from __future__ import annotations

from typing import Callable, Sequence, Union

import numpy as np

from src.gbs import VolSpec, gbs_generating_function
from src.pde_solver import interpolate_at, solve_gbs_pde
from src.test_functions import bump_function


def generator_limit_sequence(
    x0: float,
    r: float,
    p: float,
    H: float,
    sigma_low: VolSpec,
    sigma_high: VolSpec,
    t_values: Sequence[float],
    width: float,
    x_min: float,
    x_max: float,
    n_x: int,
    steps_per_solve: int,
) -> np.ndarray:
    """Compute G_approx(t) = (T_t psi(x0) - r) / t for each t in t_values.

    Each T_t psi(x0) is obtained from an independent PDE solve on [0, t]
    with initial data psi (so smaller t uses proportionally fewer, but
    always `steps_per_solve`, implicit time steps for a fixed per-step
    accuracy).
    """
    psi = bump_function(x0, r, p, H, width)
    approximations = []
    for t in t_values:
        x_grid, v_grid = solve_gbs_pde(
            psi, sigma_low, sigma_high, t, x_min, x_max, n_x, steps_per_solve
        )
        v_t_x0 = interpolate_at(x_grid, v_grid, x0)
        approximations.append((v_t_x0 - r) / t)
    return np.asarray(approximations)


def richardson_extrapolate(t_values: Sequence[float], g_values: Sequence[float]) -> float:
    """Richardson extrapolation G(t) = G_true + c*t + O(t^2) using the two
    smallest time steps (t, t/2) to cancel the leading O(t) error term:
    G_true ~ 2*G(t_min) - G(t_prev)."""
    t_values = np.asarray(t_values, dtype=float)
    order = np.argsort(t_values)[::-1]  # descending t, so the last two are the smallest
    g_sorted = np.asarray(g_values, dtype=float)[order]
    t_sorted = t_values[order]
    assert len(t_sorted) >= 2, "need at least two time steps for Richardson extrapolation"
    t_prev, t_last = t_sorted[-2], t_sorted[-1]
    g_prev, g_last = g_sorted[-2], g_sorted[-1]
    ratio = t_prev / t_last
    # Solve the linear system G(t) = G_true + c*t exactly at the two points.
    g_true = (ratio * g_last - g_prev) / (ratio - 1.0)
    return float(g_true)


def extract_generator(
    x0: float,
    r: float,
    p: float,
    H: float,
    sigma_low: VolSpec,
    sigma_high: VolSpec,
    t_values: Sequence[float],
    width: float,
    x_min: float,
    x_max: float,
    n_x: int,
    steps_per_solve: int,
) -> dict:
    """Full exp_1 pipeline for one (x0, r, p, H): sequence, extrapolation,
    and comparison with the analytic G-BS generating function."""
    g_sequence = generator_limit_sequence(
        x0, r, p, H, sigma_low, sigma_high, t_values, width, x_min, x_max, n_x, steps_per_solve
    )
    g_richardson = richardson_extrapolate(t_values, g_sequence)
    g_exact = float(gbs_generating_function(x0, r, p, H, sigma_low, sigma_high))
    return {
        "x0": x0,
        "r": r,
        "p": p,
        "H": H,
        "t_values": list(t_values),
        "g_sequence": g_sequence.tolist(),
        "g_richardson": g_richardson,
        "g_exact": g_exact,
        "abs_error_finest": abs(g_sequence[-1] - g_exact),
        "abs_error_richardson": abs(g_richardson - g_exact),
    }


def check_sublinearity(
    x0: float,
    triples: Sequence[tuple],
    alphas: Sequence[float],
    sigma_low: VolSpec,
    sigma_high: VolSpec,
) -> dict:
    """Verify property (G1): positive homogeneity G(x, a*U) = a*G(x, U) for
    a > 0, and subadditivity G(x, U1 + U2) <= G(x, U1) + G(x, U2), using the
    analytic G-BS generating function (the same function approximated by the
    PDE-based generator extraction above)."""
    homogeneity_errors = []
    for (r, p, H), a in zip(triples, alphas):
        lhs = gbs_generating_function(x0, a * r, a * p, a * H, sigma_low, sigma_high)
        rhs = a * gbs_generating_function(x0, r, p, H, sigma_low, sigma_high)
        homogeneity_errors.append(abs(float(lhs) - float(rhs)))

    subadditivity_slacks = []
    n = len(triples)
    for i in range(n):
        r1, p1, H1 = triples[i]
        r2, p2, H2 = triples[(i + 1) % n]
        lhs = gbs_generating_function(x0, r1 + r2, p1 + p2, H1 + H2, sigma_low, sigma_high)
        rhs = gbs_generating_function(x0, r1, p1, H1, sigma_low, sigma_high) + gbs_generating_function(
            x0, r2, p2, H2, sigma_low, sigma_high
        )
        subadditivity_slacks.append(float(rhs) - float(lhs))  # should be >= 0

    return {
        "max_homogeneity_error": float(np.max(homogeneity_errors)),
        "min_subadditivity_slack": float(np.min(subadditivity_slacks)),
        "subadditivity_holds": bool(np.min(subadditivity_slacks) >= -1e-9),
    }


def check_monotonicity_g2(x0: float, r: float, p: float, H: float, Y: float, sigma_low: VolSpec, sigma_high: VolSpec) -> dict:
    """Verify (G2): H >= Y implies G(x,r,p,H) >= G(x,r,p,Y)."""
    assert H >= Y
    g_h = float(gbs_generating_function(x0, r, p, H, sigma_low, sigma_high))
    g_y = float(gbs_generating_function(x0, r, p, Y, sigma_low, sigma_high))
    return {"G_H": g_h, "G_Y": g_y, "monotone": g_h >= g_y - 1e-12}


def check_antitone_g3(x0: float, r: float, s: float, p: float, H: float, sigma_low: VolSpec, sigma_high: VolSpec) -> dict:
    """Verify (G3): r >= s implies G(x,r,p,H) <= G(x,s,p,H). For the G-BS
    generating function G does not depend on r, so this holds trivially with
    equality (Section 3.4)."""
    assert r >= s
    g_r = float(gbs_generating_function(x0, r, p, H, sigma_low, sigma_high))
    g_s = float(gbs_generating_function(x0, s, p, H, sigma_low, sigma_high))
    return {"G_r": g_r, "G_s": g_s, "antitone": g_r <= g_s + 1e-12}
