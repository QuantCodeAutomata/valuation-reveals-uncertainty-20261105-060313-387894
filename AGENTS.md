# AGENTS.md

Guide for coding agents and contributors working in this repository.

## What this repository implements

A replication of four experiments from the paper *Valuation Reveals
Uncertainty*, built around its G-Black-Scholes (G-BS) example of a dynamic
sublinear valuation rule:

- `exp.exp1_generator_extraction` — recovers the generating function `G` of a
  valuation rule `T` as the infinitesimal-generator limit
  `lim_{t->0} (T_t f - f)/t`, via a finite-difference solver for the
  nonlinear valuation PDE.
- `exp.exp2_support_sets` — constructs the convex-duality support sets `A(x)`
  representing `G(x,U) = max_{V in A(x)} L^V(U)`, analytically and via linear
  programming.
- `exp.exp3_uncertainty_structure` — builds the admissible coefficient class
  from the support sets and compares the Monte Carlo robust valuation with
  the PDE valuation and the Lyapunov condition of Assumption 3.3.
- `exp.exp4_dupire` — recovers local-volatility bounds from synthetic put
  bid/ask surfaces via Dupire's formula.

See `README.md` for the paper sections each experiment reproduces and
`results/RESULTS.md` for the metrics, comparison with the paper and
deviations.

## Layout

```
src/     gbs.py (generating function), pde_solver.py (nonlinear valuation
         PDE), test_functions.py (bump-function construction),
         generator_extraction.py, support_sets.py, uncertainty_structure.py,
         dupire.py, config.py (every experiment parameter)
exp/     exp1_generator_extraction.py .. exp4_dupire.py; each calls only the
         src/ functions above and accepts --full for the paper's resolution
data/    generate_synthetic.py — the test functions, volatility
         specifications, strikes and maturities used by the experiments;
         no market data loader (the paper's G-BS example needs none)
tests/   pytest unit tests for every src/ module (small fixtures / synthetic
         inputs, no external calls)
results/ JSON metrics and PNG plots written by exp/ scripts, plus
         RESULTS.md (not overwritten automatically)
```

## Conventions

- Every experiment parameter lives in `src/config.py` as a module-level
  `UPPER_CASE` constant with a comment on its meaning, unit and source (a
  paper section or "not specified in the paper"). Do not hardcode parameters
  elsewhere.
- Reduced-vs-paper configurations are two adjacent constants, e.g.
  `EXP1_N_X_PAPER` / `EXP1_N_X_RUN`; each experiment script's `--full` flag
  selects the paper values.
- Data preparation code lives only in `data/` (`generate_synthetic.py` here,
  since all four experiments use synthetic/analytical inputs); `src/` and
  `exp/` must not define their own generators or loaders.
- No API keys are used by this repository: there is no market-data path.
  Were one added, keys would be read from an uncommitted `.env`
  (`.env.example` documents the variable names only).
- Run experiments from the repository root as `python -m exp.<name>` and
  tests as `python -m pytest`; do not edit `sys.path` in code.

## Commands

```bash
pip install -r requirements.txt
python -m pytest -q                                 # full suite
python -m pytest -q -m "not market_data"             # (no market_data tests exist in this repository)
python -m exp.exp1_generator_extraction [--full]
python -m exp.exp2_support_sets [--full]
python -m exp.exp3_uncertainty_structure [--full]
python -m exp.exp4_dupire [--full]
./reproduce.sh                                       # tests + all four experiments, reduced configuration
```

## Notes for future changes

- Re-running an experiment command overwrites its files in `results/`; the
  numbers and verdicts written into `results/RESULTS.md` are not
  auto-generated and must be checked against the newly written files and
  updated by hand after any change that affects them.
- Tests check the paper's structural properties (signs, monotonicity,
  absence of look-ahead in the PDE time-stepping, convergence order, the
  Lyapunov identity) on small fixtures, not the exact numbers reported in
  `results/RESULTS.md`.
