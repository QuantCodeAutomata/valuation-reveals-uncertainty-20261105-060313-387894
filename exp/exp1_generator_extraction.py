"""exp_1: Extracting the generating function from the valuation rule via
numerical differentiation (paper Section 3.1-3.2, Proposition 3.1).

Run: python -m exp.exp1_generator_extraction [--full]
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from src import config
from src.generator_extraction import (
    check_antitone_g3,
    check_monotonicity_g2,
    check_sublinearity,
    extract_generator,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def run(full: bool) -> dict:
    n_x = config.EXP1_N_X_PAPER if full else config.EXP1_N_X_RUN
    logger.info("exp_1 running with n_x=%d, t_values=%s, full=%s", n_x, config.EXP1_T_VALUES, full)

    records = []
    for x0 in config.EXP1_TEST_POINTS_X0:
        width = config.EXP1_BUMP_WIDTH_FACTOR * x0
        for (r, p, H) in config.EXP1_TARGET_TRIPLES:
            result = extract_generator(
                x0, r, p, H, config.SIGMA_LOW, config.SIGMA_HIGH,
                config.EXP1_T_VALUES, width, config.X_MIN, config.X_MAX,
                n_x, config.EXP1_STEPS_PER_SOLVE,
            )
            records.append(result)
            logger.info(
                "x0=%.1f (r,p,H)=(%.2f,%.2f,%.2f): G_exact=%.6f G_richardson=%.6f abs_err=%.3e",
                x0, r, p, H, result["g_exact"], result["g_richardson"], result["abs_error_richardson"],
            )

    sublinearity = check_sublinearity(
        config.X0, config.EXP1_SUBLINEARITY_TRIPLES, config.EXP1_SUBLINEARITY_ALPHAS,
        config.SIGMA_LOW, config.SIGMA_HIGH,
    )
    monotonicity = check_monotonicity_g2(config.X0, 0.0, 0.0, 2.0, -1.0, config.SIGMA_LOW, config.SIGMA_HIGH)
    antitone = check_antitone_g3(config.X0, 1.0, 0.0, 0.0, 1.0, config.SIGMA_LOW, config.SIGMA_HIGH)

    convergence_errors = [rec["abs_error_finest"] for rec in records]
    richardson_errors = [rec["abs_error_richardson"] for rec in records]

    output = {
        "config": {
            "n_x": n_x,
            "t_values": config.EXP1_T_VALUES,
            "sigma_low": config.SIGMA_LOW,
            "sigma_high": config.SIGMA_HIGH,
            "full": full,
        },
        "records": records,
        "mean_abs_error_finest_t": float(np.mean(convergence_errors)),
        "mean_abs_error_richardson": float(np.mean(richardson_errors)),
        "properties": {
            "G1_sublinearity": sublinearity,
            "G2_monotonicity_in_H": monotonicity,
            "G3_antitone_in_r": antitone,
        },
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "exp1_generator_extraction.json"
    out_path.write_text(json.dumps(output, indent=2))
    logger.info("Saved results to %s", out_path)

    _plot_convergence(records, RESULTS_DIR / "exp1_convergence.png")
    return output


def _plot_convergence(records: list[dict], out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    for rec in records[:5]:
        t_values = np.asarray(rec["t_values"])
        errors = np.abs(np.asarray(rec["g_sequence"]) - rec["g_exact"])
        label = f"x0={rec['x0']:.0f}, H={rec['H']:.1f}"
        ax.loglog(t_values, errors + 1e-16, marker="o", label=label)
    ax.set_xlabel("time step t")
    ax.set_ylabel("|G_approx(t) - G_exact|")
    ax.set_title("exp_1: convergence of the generator-limit approximation")
    ax.legend(fontsize=7)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="use the paper's full grid resolution")
    args = parser.parse_args()
    run(full=args.full)
