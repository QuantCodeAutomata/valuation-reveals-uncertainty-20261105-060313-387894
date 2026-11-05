"""exp_2: Constructing support sets from a generating function via convex
duality (paper Section 3.2, Eq. (3.5)).

Run: python -m exp.exp2_support_sets [--full]
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from src import config
from src.gbs import gbs_generating_function
from src.support_sets import (
    support_set_lp,
    verify_dual_representation,
    verify_nonempty_compact_convex,
    w_grid,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def run(full: bool) -> dict:
    grid_points = config.EXP2_W_GRID_POINTS_PAPER if full else config.EXP2_W_GRID_POINTS_RUN
    W = w_grid(config.EXP2_W_GRID_BOUND_PAPER, grid_points)
    logger.info("exp_2 running with W grid %d^3 = %d points, full=%s", grid_points, len(W), full)

    def G_func(x, r, p, H):
        return float(gbs_generating_function(x, r, p, H, config.SIGMA_LOW, config.SIGMA_HIGH))

    analytic_records = []
    lp_records = []
    for x in config.EXP2_TEST_POINTS_X:
        properties = verify_nonempty_compact_convex(x, config.SIGMA_LOW, config.SIGMA_HIGH)
        analytic_records.append({"x": x, **properties})

        lp_result = support_set_lp(x, G_func, W)
        lp_records.append({"x": x, **lp_result})
        logger.info(
            "x=%.1f: analytic Sigma in [%.4f, %.4f], LP Sigma in [%.4f, %.4f]",
            x, properties["sigma_sq_low"], properties["sigma_sq_high"],
            lp_result.get("sigma_min", float("nan")), lp_result.get("sigma_max", float("nan")),
        )

    dual_records = []
    for x in config.EXP2_TEST_POINTS_X:
        for (r, p, H) in config.EXP2_TEST_TRIPLES_U:
            dual_check = verify_dual_representation(x, r, p, H, config.SIGMA_LOW, config.SIGMA_HIGH)
            dual_records.append({"x": x, "r": r, "p": p, "H": H, **dual_check})

    lp_abs_errors = []
    for analytic, lp in zip(analytic_records, lp_records):
        if lp.get("feasible"):
            lp_abs_errors.append(abs(lp["sigma_min"] - analytic["sigma_sq_low"]))
            lp_abs_errors.append(abs(lp["sigma_max"] - analytic["sigma_sq_high"]))

    output = {
        "config": {
            "w_grid_bound": config.EXP2_W_GRID_BOUND_PAPER,
            "w_grid_points_per_axis": grid_points,
            "sigma_low": config.SIGMA_LOW,
            "sigma_high": config.SIGMA_HIGH,
            "full": full,
        },
        "analytic_support_sets": analytic_records,
        "lp_support_sets": lp_records,
        "dual_representation_checks": dual_records,
        "max_dual_abs_error": max(rec["abs_error"] for rec in dual_records),
        "max_lp_vs_analytic_abs_error": max(lp_abs_errors) if lp_abs_errors else None,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "exp2_support_sets.json"
    out_path.write_text(json.dumps(output, indent=2, default=float))
    logger.info("Saved results to %s", out_path)

    _plot_support_sets(analytic_records, RESULTS_DIR / "exp2_support_sets.png")
    return output


def _plot_support_sets(analytic_records: list[dict], out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs = [rec["x"] for rec in analytic_records]
    low = [rec["sigma_sq_low"] for rec in analytic_records]
    high = [rec["sigma_sq_high"] for rec in analytic_records]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.fill_between(xs, low, high, alpha=0.3, label="A(x) range of Sigma")
    ax.plot(xs, low, "o-", label="x^2 sigma_low(x)^2")
    ax.plot(xs, high, "o-", label="x^2 sigma_high(x)^2")
    ax.set_xlabel("x")
    ax.set_ylabel("Sigma")
    ax.set_title("exp_2: G-BS support set A(x)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="use the paper's full LP grid density")
    args = parser.parse_args()
    run(full=args.full)
