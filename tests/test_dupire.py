"""Tests for exp_4: Dupire local-volatility recovery from put bid/ask
surfaces (paper Section 3.4, Remark 3.1)."""
from __future__ import annotations

import numpy as np

from src.dupire import (
    build_constant_vol_put_surface,
    dupire_local_variance,
    summarize_local_vol_by_strike,
)

X0 = 100.0
SIGMA_LOW, SIGMA_HIGH = 0.15, 0.35


def test_dupire_recovers_constant_volatility_within_tolerance():
    strikes = np.array([70.0, 80.0, 90.0, 100.0, 110.0, 120.0, 130.0])
    maturities = np.array([0.25, 0.5, 0.75, 1.0, 1.5])

    ask_surface = build_constant_vol_put_surface(X0, strikes, maturities, SIGMA_HIGH)
    bid_surface = build_constant_vol_put_surface(X0, strikes, maturities, SIGMA_LOW)

    ask_var = dupire_local_variance(strikes, maturities, ask_surface)
    bid_var = dupire_local_variance(strikes, maturities, bid_surface)

    # Interior strikes (away from the boundary where curvature is small)
    mask = np.ones_like(ask_var, dtype=bool)
    ask_summary = summarize_local_vol_by_strike(ask_var, mask)
    bid_summary = summarize_local_vol_by_strike(bid_var, mask)

    interior = slice(1, -1)
    rel_err_high = np.abs(ask_summary["mean_by_strike"][interior] - SIGMA_HIGH**2) / SIGMA_HIGH**2
    rel_err_low = np.abs(bid_summary["mean_by_strike"][interior] - SIGMA_LOW**2) / SIGMA_LOW**2

    assert np.nanmean(rel_err_high) < 0.1
    assert np.nanmean(rel_err_low) < 0.1
    assert np.all(bid_summary["mean_by_strike"][interior] <= ask_summary["mean_by_strike"][interior] + 1e-3)


def test_dupire_local_variance_ratio_is_approximately_T_independent():
    """Compatibility condition: the recovered sigma^2 should not depend on
    maturity T for the constant-volatility case (paper Section 3.4)."""
    strikes = np.array([80.0, 90.0, 100.0, 110.0, 120.0])
    maturities = np.array([0.25, 0.5, 0.75, 1.0, 1.5])
    surface = build_constant_vol_put_surface(X0, strikes, maturities, SIGMA_HIGH)
    local_var = dupire_local_variance(strikes, maturities, surface)
    at_atm = local_var[:, strikes.tolist().index(100.0)]
    assert np.nanstd(at_atm) < 0.1 * np.nanmean(np.abs(at_atm))
