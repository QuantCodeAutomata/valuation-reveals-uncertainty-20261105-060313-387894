"""Recovering local volatility bounds from put bid/ask surfaces via Dupire's
formula (paper Section 3.4, Remark 3.1; exp_4).

Custom implementation of Dupire's formula
    sigma^2(K) = 2 * dP/dT / (K^2 * d^2P/dK^2),
applied separately to the ask surface P^ask(T,K) = T_T p_K(x0) (recovering
sigma_high(K)^2) and the bid surface P^bid(T,K) = -T_T(-p_K)(x0) (recovering
sigma_low(K)^2). scipy.interpolate.CubicSpline supplies the smooth T- and
K-derivatives (confirmed via the scipy documentation as the standard tool for
spline-based numerical differentiation); the Dupire algebra itself is paper-
specific and is implemented directly.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.interpolate import CubicSpline

from src.gbs import VolSpec, bs_put_price
from src.pde_solver import interpolate_at, solve_gbs_pde


def build_constant_vol_put_surface(
    x0: float, strikes: np.ndarray, maturities: np.ndarray, sigma: float
) -> np.ndarray:
    """Black-Scholes put price surface P[i,j] = BS_put(x0, strikes[j], maturities[i], sigma)."""
    T_grid, K_grid = np.meshgrid(maturities, strikes, indexing="ij")
    return bs_put_price(x0, K_grid, T_grid, sigma)


def build_state_dependent_put_surface(
    x0: float,
    strikes: np.ndarray,
    maturities: np.ndarray,
    sigma_low: VolSpec,
    sigma_high: VolSpec,
    side: str,
    x_min: float,
    x_max: float,
    n_x: int,
    n_t_per_unit_time: int,
) -> np.ndarray:
    """P^ask or P^bid surface for state-dependent bounds, solved by PDE per (K,T).

    side="ask": T_T p_K(x0), solved directly with payoff max(K-x,0).
    side="bid": -T_T(-p_K)(x0), solved with payoff -max(K-x,0) and negated.
    """
    assert side in ("ask", "bid")
    surface = np.zeros((len(maturities), len(strikes)))
    for i, T in enumerate(maturities):
        n_t = max(10, int(np.ceil(n_t_per_unit_time * T)))
        for j, K in enumerate(strikes):
            if side == "ask":
                payoff = lambda x, K=K: np.maximum(K - x, 0.0)
                sign = 1.0
            else:
                payoff = lambda x, K=K: -np.maximum(K - x, 0.0)
                sign = -1.0
            x_grid, v_grid = solve_gbs_pde(payoff, sigma_low, sigma_high, T, x_min, x_max, n_x, n_t)
            surface[i, j] = sign * interpolate_at(x_grid, v_grid, x0)
    return surface


def dupire_local_variance(
    strikes: np.ndarray, maturities: np.ndarray, price_surface: np.ndarray
) -> np.ndarray:
    """Dupire local variance surface sigma^2(T,K) = 2*dP/dT / (K^2 * d^2P/dK^2).

    price_surface has shape (len(maturities), len(strikes)). Derivatives use
    cubic splines along each axis (T-derivative per strike column, K second
    derivative per maturity row).
    """
    n_T, n_K = price_surface.shape
    dP_dT = np.zeros_like(price_surface)
    for j in range(n_K):
        spline_T = CubicSpline(maturities, price_surface[:, j])
        dP_dT[:, j] = spline_T(maturities, 1)

    d2P_dK2 = np.zeros_like(price_surface)
    for i in range(n_T):
        spline_K = CubicSpline(strikes, price_surface[i, :])
        d2P_dK2[i, :] = spline_K(strikes, 2)

    K_grid = strikes[np.newaxis, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        local_variance = 2.0 * dP_dT / (K_grid**2 * d2P_dK2)
    return local_variance


def summarize_local_vol_by_strike(local_variance: np.ndarray, min_curvature_mask: np.ndarray) -> dict:
    """Mean and std of the recovered local variance across maturities, for
    each strike, masking out points with ill-conditioned (near-zero or
    negative) curvature in the denominator."""
    masked = np.where(min_curvature_mask, local_variance, np.nan)
    mean_by_strike = np.nanmean(masked, axis=0)
    std_by_strike = np.nanstd(masked, axis=0)
    return {"mean_by_strike": mean_by_strike, "std_by_strike": std_by_strike}
