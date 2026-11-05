"""Synthetic-data note for this replication.

"Valuation Reveals Uncertainty" studies a purely theoretical valuation
framework. Every experiment selected for this repository (exp_1-exp_4) is
self-contained: the inputs are the G-Black-Scholes generating function and
Black-Scholes/PDE-based option prices computed from closed-form formulas or
from the finite-difference solver in src/pde_solver.py, not a dataset that
needs a dedicated generator. There is therefore no external market data and
no separate synthetic-data-generation step; all "data" used by the
experiments are produced directly by the functions in src/gbs.py,
src/pde_solver.py, src/dupire.py and src/uncertainty_structure.py as part of
each experiment's methodology.
"""
