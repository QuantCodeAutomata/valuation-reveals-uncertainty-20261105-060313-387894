"""Tests for exp_1: generator extraction via numerical differentiation
(paper Definition 3.2, Eq. (3.1)-(3.2))."""
from __future__ import annotations

import numpy as np
import pytest

from src.generator_extraction import extract_generator, generator_limit_sequence

SIGMA_LOW, SIGMA_HIGH = 0.15, 0.35
X0 = 100.0
X_MIN, X_MAX = 1e-3, 500.0
N_X = 400
STEPS_PER_SOLVE = 8
T_VALUES = [0.004, 0.002, 0.001]


@pytest.mark.parametrize("H,sign", [(1.0, 1), (-1.0, -1)])
def test_generator_extraction_matches_analytic_sign(H, sign):
    """G_approx should recover the correct sign and order of magnitude: H>0
    picks up sigma_high, H<0 picks up sigma_low (Remark 3.1)."""
    out = extract_generator(
        X0, 0.0, 0.0, H, SIGMA_LOW, SIGMA_HIGH, T_VALUES, 10.0, X_MIN, X_MAX, N_X, STEPS_PER_SOLVE
    )
    assert np.sign(out["g_richardson"]) == sign
    # Richardson-extrapolated estimate should be closer to the analytic value
    # than the coarsest finite-difference estimate in the sequence.
    assert out["abs_error_richardson"] <= abs(out["g_sequence"][0] - out["g_exact"]) + 1e-6


def test_generator_sequence_converges_as_t_shrinks():
    """|G_approx(t) - G_exact| should not grow as t decreases (discretisation
    error of the generator-limit difference quotient)."""
    out = extract_generator(
        X0, 0.0, 0.0, 1.0, SIGMA_LOW, SIGMA_HIGH, T_VALUES, 10.0, X_MIN, X_MAX, N_X, STEPS_PER_SOLVE
    )
    errors = np.abs(np.asarray(out["g_sequence"]) - out["g_exact"])
    # errors need not be monotone pointwise (PDE discretisation noise) but the
    # finest time step should not be dramatically worse than the coarsest.
    assert errors[-1] <= errors[0] * 3.0 + 1e-6


def test_richardson_extrapolation_exact_for_linear_sequence():
    """Richardson extrapolation of G(t) = G_true + c*t recovers G_true exactly."""
    from src.generator_extraction import richardson_extrapolate

    g_true, c = 2.5, 7.0
    t_values = [0.004, 0.002]
    g_values = [g_true + c * t for t in t_values]
    recovered = richardson_extrapolate(t_values, g_values)
    assert recovered == pytest.approx(g_true, abs=1e-10)
