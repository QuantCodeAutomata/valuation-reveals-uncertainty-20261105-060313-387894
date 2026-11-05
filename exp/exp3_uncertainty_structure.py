"""exp_3: Constructing and simulating the uncertainty structure U(G); robust
valuation via Monte Carlo compared to the PDE solution (paper Sections
3.3-3.5, Theorem 3.4).

Run: python -m exp.exp3_uncertainty_structure [--full]
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from src import config
from src.gbs import bs_call_price, bs_put_price
from src.pde_solver import interpolate_at, solve_gbs_pde
from src.uncertainty_structure import check_lyapunov_condition, supremum_robust_valuation

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def run(full: bool) -> dict:
    n_steps = config.EXP3_N_STEPS_PAPER if full else config.EXP3_N_STEPS_RUN
    n_paths = config.EXP3_N_PATHS_PAPER if full else config.EXP3_N_PATHS_RUN
    pde_n_x = config.EXP3_PDE_N_X_PAPER if full else config.EXP3_PDE_N_X_RUN
    pde_n_t = config.EXP3_PDE_N_T_PAPER if full else config.EXP3_PDE_N_T_RUN
    T = config.EXP3_MATURITY
    logger.info(
        "exp_3 running with n_steps=%d, n_paths=%d, pde_n_x=%d, pde_n_t=%d, full=%s",
        n_steps, n_paths, pde_n_x, pde_n_t, full,
    )

    records = []
    for K in config.EXP3_STRIKES:
        for kind, payoff, analytic_high, analytic_low in (
            ("put", lambda x, K=K: np.maximum(K - x, 0.0),
             bs_put_price(config.X0, K, T, config.SIGMA_HIGH),
             bs_put_price(config.X0, K, T, config.SIGMA_LOW)),
            ("call", lambda x, K=K: np.maximum(x - K, 0.0),
             bs_call_price(config.X0, K, T, config.SIGMA_HIGH),
             bs_call_price(config.X0, K, T, config.SIGMA_LOW)),
        ):
            mc_result = supremum_robust_valuation(
                payoff, config.X0, config.SIGMA_LOW, config.SIGMA_HIGH, T,
                n_steps, n_paths, config.EXP3_MC_SEED, config.EXP3_THRESHOLDS,
            )
            x_grid, v_grid = solve_gbs_pde(
                payoff, config.SIGMA_LOW, config.SIGMA_HIGH, T,
                config.X_MIN, config.X_MAX, pde_n_x, pde_n_t,
            )
            pde_value = interpolate_at(x_grid, v_grid, config.X0)

            mc_best = mc_result["best"]
            record = {
                "K": K,
                "kind": kind,
                "mc_best_mean": mc_best["mean"],
                "mc_best_std_error": mc_best["std_error"],
                "mc_best_type": mc_best["type"],
                "pde_value": pde_value,
                "analytic_bs_high_vol": float(analytic_high),
                "analytic_bs_low_vol": float(analytic_low),
                "abs_error_mc_vs_pde": abs(mc_best["mean"] - pde_value),
                "abs_error_pde_vs_analytic_high": abs(pde_value - float(analytic_high)),
            }
            records.append(record)
            logger.info(
                "K=%.0f %s: MC_sup=%.4f (+-%.4f) PDE=%.4f analytic(sigma_high)=%.4f",
                K, kind, mc_best["mean"], mc_best["std_error"], pde_value, analytic_high,
            )

    x_lyap = np.linspace(config.X_MIN, config.X_MAX, config.EXP3_LYAPUNOV_X_GRID_POINTS)
    lyapunov = check_lyapunov_condition(x_lyap, config.SIGMA_HIGH)
    logger.info("Lyapunov condition check: %s", lyapunov)

    output = {
        "config": {
            "n_steps": n_steps,
            "n_paths": n_paths,
            "pde_n_x": pde_n_x,
            "pde_n_t": pde_n_t,
            "maturity": T,
            "sigma_low": config.SIGMA_LOW,
            "sigma_high": config.SIGMA_HIGH,
            "full": full,
        },
        "records": records,
        "mean_abs_error_mc_vs_pde": float(np.mean([r["abs_error_mc_vs_pde"] for r in records])),
        "mean_abs_error_pde_vs_analytic_high": float(
            np.mean([r["abs_error_pde_vs_analytic_high"] for r in records])
        ),
        "lyapunov_condition": lyapunov,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "exp3_uncertainty_structure.json"
    out_path.write_text(json.dumps(output, indent=2))
    logger.info("Saved results to %s", out_path)

    _plot_comparison(records, RESULTS_DIR / "exp3_mc_vs_pde.png")
    return output


def _plot_comparison(records: list[dict], out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    puts = [r for r in records if r["kind"] == "put"]
    calls = [r for r in records if r["kind"] == "call"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, data, title in ((axes[0], puts, "Put"), (axes[1], calls, "Call")):
        strikes = [r["K"] for r in data]
        mc = [r["mc_best_mean"] for r in data]
        pde = [r["pde_value"] for r in data]
        analytic = [r["analytic_bs_high_vol"] for r in data]
        ax.plot(strikes, mc, "o", label="Monte Carlo sup")
        ax.plot(strikes, pde, "x", label="PDE (uncertain vol)")
        ax.plot(strikes, analytic, "-", label="BS @ sigma_high (analytic)")
        ax.set_xlabel("strike K")
        ax.set_ylabel("value")
        ax.set_title(f"exp_3: {title} robust valuation, T=1")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="use the paper's full Monte Carlo and PDE resolution")
    args = parser.parse_args()
    run(full=args.full)
