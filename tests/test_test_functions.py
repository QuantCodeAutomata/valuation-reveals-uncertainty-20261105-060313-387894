"""Unit tests for the bump test-function construction (Proposition 3.1)."""
from __future__ import annotations

import numpy as np
import pytest

from src.test_functions import bump_function, verify_bump_function


@pytest.mark.parametrize(
    "x0,r,p,H",
    [
        (100.0, 0.0, 0.0, 1.0),
        (100.0, 0.0, 0.0, -1.0),
        (100.0, 5.0, 0.5, 2.0),
        (80.0, -3.0, -1.0, 0.5),
    ],
)
def test_bump_matches_prescribed_value_gradient_hessian(x0, r, p, H):
    out = verify_bump_function(x0, r, p, H, width=0.1 * x0)
    assert out["value_err"] < 1e-8
    assert out["grad_err"] < 1e-4
    assert out["hess_err"] < 1e-2


def test_bump_decays_away_from_x0():
    """psi has compact-support-like decay away from x0 (quartic cutoff)."""
    psi = bump_function(100.0, 1.0, 1.0, 1.0, width=10.0)
    near = abs(psi(np.array([100.0]))[0])
    far = abs(psi(np.array([100.0 + 100.0]))[0])
    assert far < 1e-6 * max(near, 1.0)
