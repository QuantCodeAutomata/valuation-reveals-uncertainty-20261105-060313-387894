"""exp_4: Recovering local volatility bounds from put bid/ask price surfaces
via Dupire's formula (paper Section 3.4, Remark 3.1).

Run: python -m exp.exp4_dupire [--full]
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from src import config
from src.dupire import (
    build_constant_vol_put_surface,
    build_state_dependent_put_surface,
    dupire_local_variance,
    summarize_local_vol_by_strike,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def _curvature_mask(price_surface: np.ndarray, strikes: np.ndarray, rel_tol: float) -> np.ndarray:
    """Mask out (T,K) points whose K^2*d2P/dK2 (Dupire's denominator) is
    smaller in magnitude than rel_tol times its own maximum, i.e. where the
    strike curvature is too small to divide by reliably."""
    from scipy.interpolate import CubicSpline

    n_T, n_K = price_surface.shape
    d2P_dK2 = np.zeros_like(price_surface)
    for i in range(n_T):
        spline_K = CubicSpline(strikes, price_surface[i, :])
        d2P_dK2[i, :] = spline_K(strikes, 2)
    denom = strikes[np.newaxis, :] ** 2 * d2P_dK2
    return denom > rel_tol * np.max(denom)


def run_constant_vol_case(strikes: np.ndarray, maturities: np.ndarray) -> dict:
    logger.info("exp_4 constant-volatility case: sigma_low=%.3f sigma_high=%.3f", config.SIGMA_LOW, config.SIGMA_HIGH)
    ask_surface = build_constant_vol_put_surface(config.X0, strikes, maturities, config.SIGMA_HIGH)
    bid_surface = build_constant_vol_put_surface(config.X0, strikes, maturities, config.SIGMA_LOW)

    ask_local_var = dupire_local_variance(strikes, maturities, ask_surface)
    bid_local_var = dupire_local_variance(strikes, maturities, bid_surface)

    ask_mask = _curvature_mask(ask_surface, strikes, config.EXP4_CURVATURE_REL_TOL)
    bid_mask = _curvature_mask(bid_surface, strikes, config.EXP4_CURVATURE_REL_TOL)

    ask_summary = summarize_local_vol_by_strike(ask_local_var, ask_mask)
    bid_summary = summarize_local_vol_by_strike(bid_local_var, bid_mask)

    sigma_high_sq_recovered = ask_summary["mean_by_strike"]
    sigma_low_sq_recovered = bid_summary["mean_by_strike"]

    rel_err_high = np.abs(sigma_high_sq_recovered - config.SIGMA_HIGH**2) / config.SIGMA_HIGH**2
    rel_err_low = np.abs(sigma_low_sq_recovered - config.SIGMA_LOW**2) / config.SIGMA_LOW**2
    valid = ~(np.isnan(sigma_low_sq_recovered) | np.isnan(sigma_high_sq_recovered))
    ordering_holds = bool(np.all(
        sigma_low_sq_recovered[valid] <= sigma_high_sq_recovered[valid] + 1e-6
    )) if np.any(valid) else False

    return {
        "strikes": strikes.tolist(),
        "sigma_high_sq_recovered_mean": np.where(np.isnan(sigma_high_sq_recovered), None, sigma_high_sq_recovered).tolist(),
        "sigma_low_sq_recovered_mean": np.where(np.isnan(sigma_low_sq_recovered), None, sigma_low_sq_recovered).tolist(),
        "sigma_high_sq_recovered_std": np.where(np.isnan(ask_summary["std_by_strike"]), None, ask_summary["std_by_strike"]).tolist(),
        "sigma_low_sq_recovered_std": np.where(np.isnan(bid_summary["std_by_strike"]), None, bid_summary["std_by_strike"]).tolist(),
        "mean_rel_err_sigma_high_sq": float(np.nanmean(rel_err_high)),
        "mean_rel_err_sigma_low_sq": float(np.nanmean(rel_err_low)),
        "ordering_holds": ordering_holds,
    }


def run_state_dependent_case(strikes: np.ndarray, maturities: np.ndarray, full: bool) -> dict:
    n_x = config.EXP4_STATE_DEP_N_X_PAPER if full else config.EXP4_STATE_DEP_N_X_RUN
    n_t_per_unit = (
        config.EXP4_STATE_DEP_N_T_PER_UNIT_TIME_PAPER if full else config.EXP4_STATE_DEP_N_T_PER_UNIT_TIME_RUN
    )
    logger.info("exp_4 state-dependent case: n_x=%d, n_t_per_unit_time=%d, full=%s", n_x, n_t_per_unit, full)

    ask_surface = build_state_dependent_put_surface(
        config.X0, strikes, maturities, config.sigma_low_state_dependent, config.sigma_high_state_dependent,
        "ask", config.X_MIN, config.X_MAX, n_x, n_t_per_unit,
    )
    bid_surface = build_state_dependent_put_surface(
        config.X0, strikes, maturities, config.sigma_low_state_dependent, config.sigma_high_state_dependent,
        "bid", config.X_MIN, config.X_MAX, n_x, n_t_per_unit,
    )

    ask_local_var = dupire_local_variance(strikes, maturities, ask_surface)
    bid_local_var = dupire_local_variance(strikes, maturities, bid_surface)

    ask_mask = _curvature_mask(ask_surface, strikes, config.EXP4_CURVATURE_REL_TOL)
    bid_mask = _curvature_mask(bid_surface, strikes, config.EXP4_CURVATURE_REL_TOL)

    ask_summary = summarize_local_vol_by_strike(ask_local_var, ask_mask)
    bid_summary = summarize_local_vol_by_strike(bid_local_var, bid_mask)

    true_sigma_high_sq = config.sigma_high_state_dependent(strikes) ** 2
    true_sigma_low_sq = config.sigma_low_state_dependent(strikes) ** 2

    rel_err_high = np.abs(ask_summary["mean_by_strike"] - true_sigma_high_sq) / true_sigma_high_sq
    rel_err_low = np.abs(bid_summary["mean_by_strike"] - true_sigma_low_sq) / true_sigma_low_sq

    # Ordering sigma_low(K)^2 <= sigma_high(K)^2 at every strike where both
    # bounds pass the curvature mask (paper Remark 3.1 compatibility check).
    both_valid = ~np.isnan(ask_summary["mean_by_strike"]) & ~np.isnan(bid_summary["mean_by_strike"])
    ordering_holds = bool(np.all(
        bid_summary["mean_by_strike"][both_valid] <= ask_summary["mean_by_strike"][both_valid]
    ))

    return {
        "strikes": strikes.tolist(),
        "true_sigma_high_sq": true_sigma_high_sq.tolist(),
        "true_sigma_low_sq": true_sigma_low_sq.tolist(),
        "sigma_high_sq_recovered_mean": np.where(np.isnan(ask_summary["mean_by_strike"]), None, ask_summary["mean_by_strike"]).tolist(),
        "sigma_low_sq_recovered_mean": np.where(np.isnan(bid_summary["mean_by_strike"]), None, bid_summary["mean_by_strike"]).tolist(),
        "mean_rel_err_sigma_high_sq": float(np.nanmean(rel_err_high)),
        "mean_rel_err_sigma_low_sq": float(np.nanmean(rel_err_low)),
        "ordering_holds": ordering_holds,
        "n_x": n_x,
        "n_t_per_unit_time": n_t_per_unit,
    }


def run(full: bool) -> dict:
    strikes = np.asarray(config.EXP4_STRIKES, dtype=float)
    maturities = np.asarray(config.EXP4_MATURITIES, dtype=float)

    constant_case = run_constant_vol_case(strikes, maturities)
    state_dependent_case = run_state_dependent_case(strikes, maturities, full)

    output = {
        "config": {
            "strikes": strikes.tolist(),
            "maturities": maturities.tolist(),
            "sigma_low": config.SIGMA_LOW,
            "sigma_high": config.SIGMA_HIGH,
            "full": full,
        },
        "constant_volatility_case": constant_case,
        "state_dependent_case": state_dependent_case,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "exp4_dupire.json"
    out_path.write_text(json.dumps(output, indent=2))
    logger.info("Saved results to %s", out_path)

    _plot_recovered_vol(constant_case, state_dependent_case, strikes, RESULTS_DIR / "exp4_dupire_recovery.png")
    return output


def _plot_recovered_vol(constant_case: dict, state_dependent_case: dict, strikes: np.ndarray, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    ax = axes[0]
    ax.plot(strikes, constant_case["sigma_high_sq_recovered_mean"], "o-", label="recovered sigma_high^2")
    ax.plot(strikes, constant_case["sigma_low_sq_recovered_mean"], "s-", label="recovered sigma_low^2")
    ax.axhline(config.SIGMA_HIGH**2, color="C0", linestyle="--", alpha=0.6, label="true sigma_high^2")
    ax.axhline(config.SIGMA_LOW**2, color="C1", linestyle="--", alpha=0.6, label="true sigma_low^2")
    ax.set_xlabel("strike K")
    ax.set_ylabel("local variance")
    ax.set_title("exp_4: constant-bound case")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(strikes, state_dependent_case["sigma_high_sq_recovered_mean"], "o-", label="recovered sigma_high(K)^2")
    ax.plot(strikes, state_dependent_case["true_sigma_high_sq"], "--", label="true sigma_high(K)^2")
    ax.plot(strikes, state_dependent_case["sigma_low_sq_recovered_mean"], "s-", label="recovered sigma_low(K)^2")
    ax.plot(strikes, state_dependent_case["true_sigma_low_sq"], "--", label="true sigma_low(K)^2")
    ax.set_xlabel("strike K")
    ax.set_ylabel("local variance")
    ax.set_title("exp_4: state-dependent bound case")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="use the paper's full PDE resolution for the state-dependent case")
    args = parser.parse_args()
    run(full=args.full)
