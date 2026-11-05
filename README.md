# Valuation Reveals Uncertainty — Replication

Paper: *Valuation Reveals Uncertainty* (sublinear valuation rules, generating
functions and the G-Black-Scholes model).

This repository reproduces four experiments built on the paper's dynamic
sublinear valuation rules and their G-Black-Scholes (G-BS) specification:

- **exp_1 — Extracting the generating function via numerical differentiation**
  (Section 3.1–3.2). Recovers the generating function `G(x, r, p, H)` of a
  dynamic sublinear valuation rule `T` as the infinitesimal-generator limit
  `G[f](x) = lim_{t->0} (T_t f(x) - f(x)) / t`, computed by solving the
  nonlinear valuation PDE `d_t v = G(x, v, v_x, v_xx)` for test functions with
  prescribed value, gradient and Hessian at a point, and comparing against the
  closed-form G-BS generating function.
- **exp_2 — Support sets via convex duality** (Section 3.2, Eq. (3.5)).
  Constructs the support sets `A(x)` representing `G` as
  `G(x, U) = max_{V in A(x)} L^V(U)`, both analytically for G-BS and
  numerically via a discretised linear-programming feasibility problem.
- **exp_3 — Robust valuation: Monte Carlo and PDE verification**
  (Section 3.3–3.5). Builds the admissible coefficient class from the support
  sets, computes the robust valuation
  `sup_{(A,Q)} E^Q[e^{-A_t} f(X_t) 1_{tau_exp > t}]` by Monte Carlo simulation
  over admissible volatility processes, and checks it against the PDE
  valuation together with the Lyapunov condition of Assumption 3.3.
- **exp_4 — Recovering local volatility bounds via Dupire's formula**
  (Section 3.4, Remark 3.1). Generates ask/bid put price surfaces from the
  G-BS model and recovers the local-volatility bounds `sigma_high(K)^2` and
  `sigma_low(K)^2` with Dupire's formula, checking the maturity-independence
  (compatibility) condition and the ordering `sigma_low <= sigma_high`.

All four experiments are self-contained: the paper specifies the G-BS model
as a concrete test case with no external market data, so every input is
generated analytically or by the PDE/Monte-Carlo code in this repository (see
`data/generate_synthetic.py`). No market-data loader is used in this
replication.

## Repository layout

```
src/     generating functions, PDE solver, generator extraction, support
         sets, uncertainty structure / robust valuation, Dupire recovery,
         and src/config.py with every experiment parameter
exp/     one script per experiment, exp.exp1_generator_extraction ..
         exp.exp4_dupire
data/    data/generate_synthetic.py — synthetic test functions, volatility
         specifications and strike/maturity grids used by the experiments
tests/   pytest unit tests for every src/ module
results/ metrics (JSON) and plots (PNG) written by the experiment scripts,
         plus RESULTS.md with the comparison to the paper
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

No API keys or external data are required for this replication.

## Running the tests

```bash
python -m pytest
```

## Running the experiments

Each experiment runs in a reduced configuration by default (chosen so that
all four experiments finish in a few minutes) and accepts `--full` to use the
paper's resolution (larger PDE grids, more Monte Carlo paths, the paper's LP
grid density — see `src/config.py` for the exact paper vs. executed values).

```bash
python -m exp.exp1_generator_extraction            # reduced grid
python -m exp.exp1_generator_extraction --full      # paper's grid (N_x = 2000)

python -m exp.exp2_support_sets
python -m exp.exp2_support_sets --full              # paper's LP grid (50 points/axis)

python -m exp.exp3_uncertainty_structure
python -m exp.exp3_uncertainty_structure --full     # paper's N_s=1000, N_mc=100000

python -m exp.exp4_dupire
python -m exp.exp4_dupire --full                    # paper's PDE resolution
```

Each script writes its metrics as JSON and its plots as PNG directly into
`results/`, overwriting any previous run's files.

## Reproduce everything

```bash
./reproduce.sh
```

runs the test suite and all four experiments in their reduced (default)
configuration.

## Results

See [`results/RESULTS.md`](results/RESULTS.md) for the metrics produced by
the saved runs, their comparison with the paper's expected outcomes, and the
deviations from the paper.
