"""Smooth bump test functions with prescribed value, gradient and Hessian.

Custom implementation. Proposition 3.1 only asserts the *existence* of a
function psi in C_b^infty(D) with prescribed (psi(x0), psi'(x0), psi''(x0))
= (r, p, H); the paper does not give an explicit construction (see the
"uncertainties" recorded for this experiment). A natural candidate is a
polynomial matching (r, p, H) multiplied by a smooth cutoff equal to 1 near
x0 and decaying away from it.

A quadratic-exponent cutoff exp(-((x - x0)/w)^2), as suggested informally for
this kind of construction, does NOT preserve the Hessian exactly: writing
chi(u) = exp(-(u/w)^2) and g(u) = r + p u + 1/2 H u^2 for u = x - x0, direct
differentiation gives chi'(0) = 0 but chi''(0) = -2/w^2, so
(g * chi)''(0) = H - 2 r / w^2, a spurious O(r/w^2) bias whenever r != 0.
Using the quartic-exponent cutoff chi(u) = exp(-(u/w)^4) instead gives
chi(0) = 1 and chi'(0) = chi''(0) = 0, so (g * chi)(0) = r,
(g * chi)'(0) = p and (g * chi)''(0) = H exactly, for any (r, p, H). This
quartic cutoff is therefore used throughout.
"""
from __future__ import annotations

from typing import Callable

import numpy as np


def bump_function(x0: float, r: float, p: float, H: float, width: float) -> Callable[[np.ndarray], np.ndarray]:
    """Return psi with psi(x0) = r, psi'(x0) = p, psi''(x0) = H exactly.

    psi(x) = (r + p*(x-x0) + 0.5*H*(x-x0)^2) * exp(-((x-x0)/width)^4)
    """
    assert width > 0, "width must be positive"

    def psi(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        u = x - x0
        polynomial = r + p * u + 0.5 * H * u**2
        cutoff = np.exp(-((u / width) ** 4))
        return polynomial * cutoff

    return psi


def verify_bump_function(x0: float, r: float, p: float, H: float, width: float, h: float = 1e-4) -> dict:
    """Finite-difference check that bump_function reproduces (r, p, H) at x0."""
    psi = bump_function(x0, r, p, H, width)
    value = psi(np.array([x0]))[0]
    grad = (psi(np.array([x0 + h])) - psi(np.array([x0 - h])))[0] / (2 * h)
    hess = (psi(np.array([x0 + h])) - 2 * psi(np.array([x0])) + psi(np.array([x0 - h])))[0] / h**2
    return {
        "value_err": abs(value - r),
        "grad_err": abs(grad - p),
        "hess_err": abs(hess - H),
    }
