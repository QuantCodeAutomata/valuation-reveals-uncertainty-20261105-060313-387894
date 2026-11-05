#!/usr/bin/env bash
# Reproduce the replication of "Valuation Reveals Uncertainty": run the test
# suite and all four experiments in their reduced (default) configuration.
set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

echo "=== Running tests ==="
python -m pytest -q || exit 1

echo "=== exp_1: generator extraction ==="
python -m exp.exp1_generator_extraction

echo "=== exp_2: support sets ==="
python -m exp.exp2_support_sets

echo "=== exp_3: uncertainty structure (Monte Carlo + PDE) ==="
python -m exp.exp3_uncertainty_structure

echo "=== exp_4: Dupire recovery ==="
python -m exp.exp4_dupire

echo "=== Done. See results/RESULTS.md for the comparison with the paper. ==="
