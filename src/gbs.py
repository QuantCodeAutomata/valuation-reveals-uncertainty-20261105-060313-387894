"""The G-Black-Scholes (G-BS) generating function and its analytic objects.

Custom implementation (paper Remark 3.1, "Identification under structural
restrictions"): the generating function

    G(x, r, p, H) = 1/2 * (x^2 * sigma_high(x)^2 * H_+ + x^2 * sigma_low(x)^2 * H_-)
                  = max_{sigma_low(x) <= s <= sigma_high(x)} 1/2 * x^2 * s^2 * H

on D = (0, infinity), with d = 1 (so S^d = R, S^d_+ = [0, infinity) and H is a
scalar). No library implements this problem-specific sublinear generator, so
it is coded directly from the formula.
"""
from __future__ import annotations

from typing import Callable, Union

import numpy as np
from scipy.stats import norm

Number = Union[float, np.ndarray]
VolSpec = Union[float, Callable[[Number], Number]]


def _as_callable(sigma: VolSpec) -> Callable[[Number], Number]:
    """Wrap a constant volatility bound into a callable of x."""
    if callable(sigma):
        return sigma
    value = float(sigma)
    return lambda x: np.full_like(np.asarray(x, dtype=float), value) if np.ndim(x) else value


def gbs_generating_function(
    x: Number, r: Number, p: Number, H: Number, sigma_low: VolSpec, sigma_high: VolSpec
) -> Number:
    """Evaluate G(x, r, p, H) for the G-BS specification.

    G does not depend on (r, p): properties (G1) sublinearity in (r, p) and
    (G3) antitonicity in r hold trivially for this example (Section 3.4).
    """
    sigma_low_f = _as_callable(sigma_low)
    sigma_high_f = _as_callable(sigma_high)
    x = np.asarray(x, dtype=float)
    H = np.asarray(H, dtype=float)
    sigma_bar = np.asarray(sigma_high_f(x), dtype=float)
    sigma_bar_low = np.asarray(sigma_low_f(x), dtype=float)
    # H = H_+ + H_- with H_+ = max(H,0) >= 0 and H_- = min(H,0) <= 0 (so that,
    # for H < 0, the term sigma_low**2 * H_- is negative): this is the unique
    # decomposition consistent with the paper's own equality
    # G = 1/2(x^2 sigma_high^2 H_+ + x^2 sigma_low^2 H_-)
    #   = max_{sigma_low <= s <= sigma_high} 1/2 x^2 s^2 H
    # (Remark 3.1): the right-hand max, linear in s^2 with slope (1/2)x^2 H,
    # is attained at s = sigma_high for H >= 0 and at s = sigma_low for H < 0,
    # giving (1/2)x^2 sigma_low^2 H (negative) in the latter case.
    H_plus = np.maximum(H, 0.0)
    H_minus = np.minimum(H, 0.0)
    return 0.5 * x**2 * (sigma_bar**2 * H_plus + sigma_bar_low**2 * H_minus)


def gbs_support_set_sigma_bounds(x: Number, sigma_low: VolSpec, sigma_high: VolSpec):
    """Analytic support set A(x) = {(0, 0, Sigma): x^2 sigma_low(x)^2 <= Sigma <= x^2 sigma_high(x)^2}.

    Derived in the exp_2 methodology from the definition
    A(x) = {V in (-inf,0] x R^d x S^d_+ : G(x, W) >= L^V(W) for all W}:
    since G does not depend on (r, p), C = 0 and B = 0 are forced, and the
    H > 0 / H < 0 cases pin down Sigma in [x^2 sigma_low(x)^2, x^2 sigma_high(x)^2].
    """
    sigma_low_f = _as_callable(sigma_low)
    sigma_high_f = _as_callable(sigma_high)
    x = np.asarray(x, dtype=float)
    sigma_sq_low = x**2 * np.asarray(sigma_low_f(x), dtype=float) ** 2
    sigma_sq_high = x**2 * np.asarray(sigma_high_f(x), dtype=float) ** 2
    return sigma_sq_low, sigma_sq_high


def gbs_dual_representation(
    x: Number, r: Number, p: Number, H: Number, sigma_low: VolSpec, sigma_high: VolSpec
) -> Number:
    """max_{V in A(x)} L^V(r, p, H) for the G-BS support set, L^V(r,p,H)=0.5*Sigma*H.

    With A(x) = {(0, 0, Sigma): sigma_sq_low <= Sigma <= sigma_sq_high}, the
    maximum of 0.5 * Sigma * H over Sigma in that interval is attained at the
    upper end when H >= 0 and at the lower end when H < 0, reproducing (3.5).
    """
    sigma_sq_low, sigma_sq_high = gbs_support_set_sigma_bounds(x, sigma_low, sigma_high)
    H = np.asarray(H, dtype=float)
    sigma_star = np.where(H >= 0, sigma_sq_high, sigma_sq_low)
    return 0.5 * sigma_star * H


def linear_functional(V, W):
    """L^V(r, p, H) = 1/2 * Sigma * H + B * p + C * r for V = (C, B, Sigma), d = 1."""
    C, B, Sigma = V
    r, p, H = W
    return 0.5 * Sigma * H + B * p + C * r


def bs_put_price(x0: Number, K: Number, T: Number, sigma: Number, rate: float = 0.0) -> Number:
    """Black-Scholes put price with spot x0, strike K, maturity T, flat vol sigma.

    Standard closed form (scipy.stats.norm for the normal CDF); zero interest
    rate matches the G-BS generating function, which has no discounting (C)
    or drift (B) ambiguity (Section 3.4).
    """
    x0 = np.asarray(x0, dtype=float)
    K = np.asarray(K, dtype=float)
    T = np.asarray(T, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        sqrt_T = np.sqrt(T)
        d1 = (np.log(x0 / K) + (rate + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
        d2 = d1 - sigma * sqrt_T
        disc = np.exp(-rate * T)
        price = K * disc * norm.cdf(-d2) - x0 * norm.cdf(-d1)
        price = np.where(T <= 0, np.maximum(K - x0, 0.0), price)
    return price


def bs_call_price(x0: Number, K: Number, T: Number, sigma: Number, rate: float = 0.0) -> Number:
    """Black-Scholes call price, used as an additional convex test payoff."""
    x0 = np.asarray(x0, dtype=float)
    K = np.asarray(K, dtype=float)
    T = np.asarray(T, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        sqrt_T = np.sqrt(T)
        d1 = (np.log(x0 / K) + (rate + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
        d2 = d1 - sigma * sqrt_T
        disc = np.exp(-rate * T)
        price = x0 * norm.cdf(d1) - K * disc * norm.cdf(d2)
        price = np.where(T <= 0, np.maximum(x0 - K, 0.0), price)
    return price
