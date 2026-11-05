"""Unit tests for the G-BS generating function and its analytic dual objects
(paper Remark 3.1, Section 3.4)."""
from __future__ import annotations

import numpy as np
import pytest

from src.gbs import (
    bs_call_price,
    bs_put_price,
    gbs_dual_representation,
    gbs_generating_function,
    gbs_support_set_sigma_bounds,
)
from src.generator_extraction import (
    check_antitone_g3,
    check_monotonicity_g2,
    check_sublinearity,
)
from src.support_sets import verify_dual_representation, verify_nonempty_compact_convex

SIGMA_LOW, SIGMA_HIGH = 0.15, 0.35


def test_positive_H_uses_sigma_high():
    """For H > 0, G(x,r,p,H) = 0.5*x^2*sigma_high^2*H (Remark 3.1)."""
    g = gbs_generating_function(100.0, 0.0, 0.0, 2.0, SIGMA_LOW, SIGMA_HIGH)
    expected = 0.5 * 100.0**2 * SIGMA_HIGH**2 * 2.0
    assert g == pytest.approx(expected)


def test_negative_H_uses_sigma_low():
    """For H < 0, G(x,r,p,H) = 0.5*x^2*sigma_low^2*H (negative)."""
    g = gbs_generating_function(100.0, 0.0, 0.0, -2.0, SIGMA_LOW, SIGMA_HIGH)
    expected = 0.5 * 100.0**2 * SIGMA_LOW**2 * (-2.0)
    assert g == pytest.approx(expected)
    assert g < 0


def test_zero_H_gives_zero():
    assert gbs_generating_function(100.0, 1.0, 2.0, 0.0, SIGMA_LOW, SIGMA_HIGH) == pytest.approx(0.0)


def test_g1_sublinearity_holds():
    """(G1): positive homogeneity and subadditivity (Section 3.4)."""
    triples = [(1.0, 2.0, 3.0), (-2.0, 0.5, -1.0), (4.0, -1.5, 2.5), (0.5, 0.5, -0.5)]
    alphas = [0.5, 1.0, 2.0, 3.5]
    out = check_sublinearity(100.0, triples, alphas, SIGMA_LOW, SIGMA_HIGH)
    assert out["max_homogeneity_error"] < 1e-9
    assert out["subadditivity_holds"]


def test_g2_monotonicity_in_H():
    """(G2): H >= Y implies G(x,r,p,H) >= G(x,r,p,Y)."""
    out = check_monotonicity_g2(100.0, 0.0, 0.0, 2.0, -1.0, SIGMA_LOW, SIGMA_HIGH)
    assert out["monotone"]
    assert out["G_H"] > out["G_Y"]


def test_g3_antitone_in_r_holds_trivially():
    """(G3): for the G-BS case G does not depend on r, so equality holds."""
    out = check_antitone_g3(100.0, 1.0, 0.0, 0.0, 1.0, SIGMA_LOW, SIGMA_HIGH)
    assert out["antitone"]
    assert out["G_r"] == pytest.approx(out["G_s"])


def test_dual_representation_matches_direct_evaluation():
    """max_{V in A(x)} L^V(U) = G(x,U) (Eq. (3.5))."""
    for (x, r, p, H) in [(80.0, 0.0, 0.0, 1.0), (100.0, 2.0, 1.0, -3.0), (130.0, -1.0, -2.0, 5.0)]:
        out = verify_dual_representation(x, r, p, H, SIGMA_LOW, SIGMA_HIGH)
        assert out["abs_error"] < 1e-9


def test_support_set_bounds_ordering_and_shape():
    """A(x) = {(0,0,Sigma): x^2 sigma_low^2 <= Sigma <= x^2 sigma_high^2} is a
    nonempty, compact, convex interval (exp_2 methodology)."""
    for x in (50.0, 100.0, 200.0):
        out = verify_nonempty_compact_convex(x, SIGMA_LOW, SIGMA_HIGH)
        assert out["nonempty"] and out["compact"] and out["convex"]
        low, high = gbs_support_set_sigma_bounds(x, SIGMA_LOW, SIGMA_HIGH)
        assert low == pytest.approx(x**2 * SIGMA_LOW**2)
        assert high == pytest.approx(x**2 * SIGMA_HIGH**2)
        assert low <= high


def test_state_dependent_vol_bounds_are_respected():
    """The generating function also accepts callables sigma_low(x), sigma_high(x)."""
    sigma_low_fn = lambda x: 0.1 + 0.05 * np.tanh(np.asarray(x) / 50.0)
    sigma_high_fn = lambda x: 0.3 + 0.05 * np.tanh(np.asarray(x) / 50.0)
    x = np.array([80.0, 100.0, 120.0])
    g = gbs_generating_function(x, 0.0, 0.0, 1.0, sigma_low_fn, sigma_high_fn)
    expected = 0.5 * x**2 * sigma_high_fn(x) ** 2 * 1.0
    np.testing.assert_allclose(g, expected)


def test_bs_put_call_parity_zero_rate():
    """With zero interest rate, call - put = x0 - K (put-call parity)."""
    x0, K, T, sigma = 100.0, 95.0, 1.0, 0.25
    call = bs_call_price(x0, K, T, sigma)
    put = bs_put_price(x0, K, T, sigma)
    assert float(call - put) == pytest.approx(x0 - K, abs=1e-8)


def test_bs_put_price_nonnegative_and_decreasing_in_spot():
    prices = [bs_put_price(x0, 100.0, 1.0, 0.2) for x0 in (80.0, 100.0, 120.0)]
    assert all(p >= 0 for p in prices)
    assert prices[0] > prices[1] > prices[2]
