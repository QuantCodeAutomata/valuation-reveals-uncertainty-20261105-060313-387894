"""Support sets A(x) from a generating function via convex duality
(paper Section 3.2, Eq. (3.5); exp_2).

Custom implementation. The analytic G-BS support set is coded directly from
the derivation in the exp_2 methodology. The generic (non-G-BS) construction
approximates the semi-infinite constraint "G(x, W) >= L^V(W) for all W" by
discretising W on a finite grid and solving two linear programs (minimise and
maximise Sigma, with C <= 0 and Sigma >= 0) using scipy.optimize.linprog --
the standard scipy tool for linear programming (confirmed via the scipy
documentation; no higher-level convex-optimisation modelling package such as
cvxpy is installed in this environment, and this constraint is already linear
in V for each fixed W, so linprog is the natural and sufficient tool).
"""
from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.optimize import linprog

from src.gbs import VolSpec, gbs_dual_representation, gbs_generating_function, gbs_support_set_sigma_bounds


def w_grid(bound: float, n_points: int) -> np.ndarray:
    """Discretised grid of W = (r, p, H) in [-bound, bound]^3 (flattened)."""
    axis = np.linspace(-bound, bound, n_points)
    r, p, H = np.meshgrid(axis, axis, axis, indexing="ij")
    return np.stack([r.ravel(), p.ravel(), H.ravel()], axis=1)


def support_set_lp(
    x: float,
    G_func: Callable[[float, float, float, float], float],
    W: np.ndarray,
    c_upper_bound: float = 0.0,
    sigma_lower_bound: float = 0.0,
    large_bound: float = 1e6,
) -> dict:
    """Solve for the range of Sigma (1-D case: C, B, Sigma scalars) in
    A(x) = {(C,B,Sigma): C<=0, Sigma>=0, C*r+B*p+0.5*Sigma*H <= G(x,W) for all W in grid}.

    Returns the extremal Sigma values (minimise / maximise) found by linear
    programming, together with the optimal (C, B) at each extreme.
    """
    r_col, p_col, H_col = W[:, 0], W[:, 1], W[:, 2]
    G_values = np.array([G_func(x, r, p, H) for r, p, H in W])

    # Variables ordered as (C, B, Sigma). Constraint: C*r + B*p + 0.5*Sigma*H <= G(x,W).
    A_ub = np.stack([r_col, p_col, 0.5 * H_col], axis=1)
    b_ub = G_values
    bounds = [(-large_bound, c_upper_bound), (-large_bound, large_bound), (sigma_lower_bound, large_bound)]

    result_min = linprog(c=[0.0, 0.0, 1.0], A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
    result_max = linprog(c=[0.0, 0.0, -1.0], A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")

    out = {"feasible": bool(result_min.success and result_max.success)}
    if result_min.success:
        out["sigma_min"] = float(result_min.x[2])
        out["C_at_min"] = float(result_min.x[0])
        out["B_at_min"] = float(result_min.x[1])
    if result_max.success:
        out["sigma_max"] = float(result_max.x[2])
        out["C_at_max"] = float(result_max.x[0])
        out["B_at_max"] = float(result_max.x[1])
    return out


def verify_dual_representation(
    x: float, r: float, p: float, H: float, sigma_low: VolSpec, sigma_high: VolSpec
) -> dict:
    """Check that max_{V in A(x)} L^V(U) equals G(x, U) for the G-BS case."""
    g_direct = float(gbs_generating_function(x, r, p, H, sigma_low, sigma_high))
    g_dual = float(gbs_dual_representation(x, r, p, H, sigma_low, sigma_high))
    return {"G_direct": g_direct, "G_dual": g_dual, "abs_error": abs(g_direct - g_dual)}


def verify_nonempty_compact_convex(x: float, sigma_low: VolSpec, sigma_high: VolSpec) -> dict:
    """Verify the analytic A(x) = {(0,0,Sigma): sigma_sq_low <= Sigma <= sigma_sq_high}
    is nonempty (sigma_sq_low <= sigma_sq_high), compact (bounded interval) and
    convex (an interval is convex; checked by closure under a convex
    combination of two sampled boundary points)."""
    sigma_sq_low, sigma_sq_high = gbs_support_set_sigma_bounds(x, sigma_low, sigma_high)
    sigma_sq_low = float(sigma_sq_low)
    sigma_sq_high = float(sigma_sq_high)
    nonempty = sigma_sq_low <= sigma_sq_high
    compact = np.isfinite(sigma_sq_low) and np.isfinite(sigma_sq_high)
    lam = 0.37
    mid = lam * sigma_sq_low + (1 - lam) * sigma_sq_high
    convex = sigma_sq_low <= mid <= sigma_sq_high
    return {
        "sigma_sq_low": sigma_sq_low,
        "sigma_sq_high": sigma_sq_high,
        "nonempty": bool(nonempty),
        "compact": bool(compact),
        "convex": bool(convex),
    }
