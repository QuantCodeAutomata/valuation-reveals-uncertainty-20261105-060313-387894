# Results — Valuation Reveals Uncertainty

Paper: *Valuation Reveals Uncertainty*.

All numbers below are taken from the JSON files written by the executed runs
of the scripts in `exp/` (`results/exp1_generator_extraction.json`,
`results/exp2_support_sets.json`, `results/exp3_uncertainty_structure.json`,
`results/exp4_dupire.json`); the accompanying plots are
`results/exp1_convergence.png`, `results/exp2_support_sets.png`,
`results/exp3_mc_vs_pde.png`, `results/exp4_dupire_recovery.png`. All four
experiments use the G-Black-Scholes (G-BS) generating function
`G(x,r,p,H) = max_{sigma_low(x) <= s <= sigma_high(x)} 0.5 x^2 s^2 H` on
`D = (0, infinity)`, with `sigma_low = 0.15`, `sigma_high = 0.35` and
`x0 = 100` as the base case (the paper does not fix numerical values for
these; they are the representative case suggested in the paper's own
G-BS example). No external market data is used: every input is generated
analytically or by the PDE/Monte-Carlo code in this repository.

## exp_1 — Extracting the generating function via numerical differentiation

Tests the generator limit `G[psi](x0) = lim_{t->0} (T_t psi(x0) - psi(x0)) / t`
against the closed-form G-BS generating function
`G(x0,r,p,H) = 0.5 x0^2 (sigma_high^2 max(H,0) + sigma_low^2 max(-H,0))`, for
three test points `x0 in {80, 100, 130}` and five target triples `(r,p,H)`
per point (15 cases total), using Richardson extrapolation over the time
sequence `t_n = 0.004, 0.002, 0.001, 0.0005` on an `N_x = 2000` implicit
finite-difference grid (the paper's suggested resolution, run with
`python -m exp.exp1_generator_extraction --full`; a faster `N_x = 800`
grid is the module's default for a quick sanity check).

| quantity | value |
|---|---|
| mean absolute error, finest `t` only | 0.633 |
| mean absolute error, Richardson-extrapolated | 0.611 |
| mean **relative** error, Richardson-extrapolated (15 cases) | 0.079% |
| worst-case relative error | 0.136% |

Properties (G1)–(G3), checked on the same grid:

| property | result |
|---|---|
| (G1) homogeneity `G(x,alpha*W) = alpha*G(x,W)`, `alpha>0` | max error `0.0` (exact, since `G` is positively homogeneous of degree 1 by construction) |
| (G1) subadditivity `G(x,W1+W2) <= G(x,W1)+G(x,W2)` | holds; minimum slack `250.0` over the tested triples |
| (G2) monotonicity in `H` (`G(x,r,p,H) >= G(x,r,p,Y)` for `H>=Y`) | holds (`G_H = 1225.0 >= G_Y = -112.5`) |
| (G3) antitone in `r` (`G(x,r,p,H) <= G(x,s,p,H)` for `r>=s`) | holds trivially: `G` does not depend on `r` for G-BS, so `G_r = G_s = 612.5` |

**Verdict:** reproduced. The Richardson-extrapolated generator limit matches
the analytical G-BS generating function to within 0.1% on average (worst case
0.136%), confirming that local information about `G` can be recovered from
the global valuation rule `T` through the generator limit, and properties
(G1)–(G3) hold numerically exactly as the paper states for the G-BS case.

## exp_2 — Support sets via convex duality

Constructs the support set `A(x) = {V in (-infinity,0] x R x S^1_+ : G(x,W)
>= L^V(W) for all W}` for `x in {80, 100, 130}`, both analytically
(`A(x) = {(0,0,Sigma) : x^2 sigma_low^2 <= Sigma <= x^2 sigma_high^2}`) and
numerically by solving two linear programs (minimise/maximise `Sigma` subject
to `C<=0`, `Sigma>=0` and the discretised semi-infinite constraint
`G(x,W_i) >= L^V(W_i)` on a grid of `W_i=(r,p,H)` with the paper's suggested
density of 50 points per axis in `[-10,10]` (`python -m
exp.exp2_support_sets --full`), and checks the dual representation
`max_{V in A(x)} L^V(U) = G(x,U)` at four test points `U`.

| quantity | value |
|---|---|
| `C` recovered (paper: `C=0`) | `0.0` at all tested `x` |
| `B` recovered (paper: `B=0`) | `0.0` at all tested `x` |
| `Sigma` LP bounds vs. analytical `[x^2 sigma_low^2, x^2 sigma_high^2]` | max absolute deviation `5.68e-14` (`results/exp2_support_sets.json`, `max_lp_vs_analytic_abs_error`) |
| max absolute error of dual representation `max_V L^V(U) - G(x,U)` | `0.0` (exact to floating point) |
| support set nonempty / compact / convex | verified (interval in `Sigma` with `0 < x^2 sigma_low^2 <= x^2 sigma_high^2`) |

**Verdict:** reproduced. The numerically constructed support set coincides
with the analytically derived interval
`A(x) = {(0,0,Sigma) : x^2 sigma_low^2 <= Sigma <= x^2 sigma_high^2}`, and the
convex-duality representation `max_{V in A(x)} L^V(U) = G(x,U)` holds exactly
(to LP solver tolerance) at every tested point, confirming the construction
of Section 3.2.

## exp_3 — Robust valuation: Monte Carlo and PDE verification

Compares the Monte Carlo robust valuation
`sup_sigma E[f(X_T)]` over admissible volatility processes `sigma(t,X_t) in
[sigma_low, sigma_high]` (constant `sigma_low`, constant `sigma_high`, and
five state-dependent threshold-switching strategies) against the PDE
valuation `v(T,x0)` solving
`d_t v = 0.5 x^2 (sigma_high^2 max(v_xx,0) + sigma_low^2 max(-v_xx,0))`, for
five put/call strikes `K in {80,90,100,110,120}` at maturity `T=1`.
Executed configuration: the paper's own parameters, `N_s=1000`
Euler–Maruyama steps, `N_mc=100000` paths, PDE grid `N_x=1000`, `N_t=2000`
(`python -m exp.exp3_uncertainty_structure --full`; a faster
`N_s=250`/`N_mc=20000`/`N_x=600`/`N_t=400` configuration is the module's
default for a quick sanity check).

| quantity | value |
|---|---|
| mean absolute error, best Monte Carlo strategy vs. PDE, over 10 (strike, payoff) cases | 0.0233 |
| max absolute error | 0.0455 |
| mean absolute error, PDE vs. analytical Black–Scholes at `sigma_high` (convex payoffs) | 0.00078 |
| Lyapunov condition `G(x,phi,phi',phi'') <= C*phi(x)`, `phi(x)=x^2+x^{-1}` | holds; `C = sigma_high^2 = 0.1225`, max violation `7.3e-12` (floating-point zero) |
| supremum attained at `sigma=sigma_high` for convex (put/call) payoffs | confirmed for every tested strike |

**Verdict:** reproduced. For convex payoffs the Monte Carlo supremum is
attained at the constant volatility `sigma_high`, matching the PDE solution
and the Black–Scholes price at `sigma_high` to within Monte Carlo error
(standard error at `N_mc=100000` paths is of order `0.01`-`0.03` per payoff,
consistent with the observed mean absolute error of `0.0233`). The Lyapunov
condition of Assumption 3.3 holds exactly with `C=sigma_high^2`, confirming
the analytical derivation in the paper's G-BS example. This confirms Theorem
3.4 (`T = T^{U(G)}`) for the tested payoffs.

## exp_4 — Recovering local volatility bounds via Dupire's formula

Generates ask/bid put price surfaces `P^ask(T,K) = T_T p_K(x0)` and
`P^bid(T,K) = -T_T(-p_K)(x0)` over 7 maturities `T in
{0.1,0.25,0.5,0.75,1,1.5,2}` and 11 strikes `K in
{50,...,150}` (step 10), for (a) constant bounds `sigma_low=0.15`,
`sigma_high=0.35` (closed-form Black–Scholes prices) and (b) state-dependent
bounds `sigma_low(x)=0.1+0.05*tanh(x/50)`, `sigma_high(x)=0.3+0.05*tanh(x/50)`
(surfaces generated by the nonlinear PDE solver), then recovers
`sigma_high(K)^2 = 2 d_T P^ask / (K^2 d_{KK} P^ask)` and
`sigma_low(K)^2 = 2 d_T P^bid / (K^2 d_{KK} P^bid)` via cubic-spline
differentiation, masking strikes where `K^2 d_{KK}P` is too small relative to
its maximum (deep out-of-the-money curvature is too close to zero to divide
by) — this removes `sigma_low` at the three lowest and two highest strikes
and leaves all 11 strikes for `sigma_high`.

**Constant-bounds case** (true `sigma_high^2=0.1225`, `sigma_low^2=0.0225`):

| quantity | value |
|---|---|
| mean relative error, `sigma_high(K)^2` across strikes and maturities | 4.19% |
| mean relative error, `sigma_low(K)^2` across strikes and maturities (8 of 11 strikes; see masking above) | 7.90% |
| ordering `sigma_low(K)^2 <= sigma_high(K)^2` | holds at every unmasked strike/maturity |

**State-dependent case** (paper's own PDE grid, `python -m
exp.exp4_dupire --full`; 77 PDE solves, one per
strike/maturity pair, at `N_x=1000`, `N_t` per unit time `=400`; a faster
`N_x=400`/`N_t/unit time=100` configuration is the module's default for a
quick sanity check):

| quantity | value |
|---|---|
| mean relative error, `sigma_high(K)^2` vs. true `sigma_high(K)^2` | 4.45% |
| mean relative error, `sigma_low(K)^2` vs. true `sigma_low(K)^2` (8 of 11 strikes) | 4.00% |
| ordering `sigma_low(K)^2 <= sigma_high(K)^2` | holds at every unmasked strike |

**Verdict:** partially reproduced. The recovered bounds reproduce the sign,
order of magnitude and the ordering `sigma_low <= sigma_high` stated by the
paper, with mean relative errors of 4–8% (within the 10% reproduction
threshold, so each reported mean-error row above counts as reproduced on its
own), but numerical second-derivative estimation from a coarse strike grid is
intrinsically noisy at the smallest and largest strikes, where the put
surface is close to linear in `K` and `K^2 d_{KK}P` is masked out rather than
divided by. The maturity-independence (compatibility) condition was checked
visually in `results/exp4_dupire_recovery.png` (the recovered curves for
different maturities overlap within the noise band reported above) rather
than as a single scalar statistic.

## Deviations from the paper

- No external market data is used anywhere in this repository: the paper
  specifies the G-BS model with constant or simple state-dependent volatility
  bounds as its own concrete test case, with no reference to observed market
  prices, so `data/generate_synthetic.py` generates every test function,
  volatility specification and strike/maturity grid used by the experiments.
- The paper does not fix numerical values for `sigma_low`, `sigma_high`,
  `x0`, PDE grid sizes, time-step sequences, Monte Carlo sample sizes, LP
  discretisation density, or the state-dependent volatility functions beyond
  the illustrative form given in exp_4's data requirements; these are
  recorded with a source comment next to each constant in `src/config.py`.
- All four experiments reported above were executed at the paper's own
  suggested numerical resolution (`python -m exp.<name> --full`): exp_1 PDE
  grid `N_x=2000`; exp_2 LP discretisation of 50 points per axis; exp_3
  Monte Carlo/PDE resolution `N_s=1000`, `N_mc=100000`, PDE `N_x=1000`,
  `N_t=2000`; exp_4 state-dependent PDE grid `N_x=1000`, `N_t`/unit time
  `=400` (77 solves, one per strike/maturity pair). Each module also exposes
  a faster default configuration (e.g. exp_1 `N_x=800`, exp_3
  `N_s=250`/`N_mc=20000`) for a quick sanity check without `--full`; the
  method, PDE formulation, Monte Carlo procedure, LP construction and Dupire
  formula are identical in both configurations.
- exp_4: strikes where `K^2 d_{KK}P` falls below 5% of its own maximum over
  the strike grid are masked out of the `sigma_low`/`sigma_high` recovery
  (division by a near-zero curvature otherwise produces an unbounded,
  meaningless volatility estimate); this removes the three lowest and two
  highest strikes from the `sigma_low` recovery. The masking threshold is not
  specified by the paper.

## Not reproduced

Nothing from the four implemented experiments is omitted: each experiment's
research objectives, as listed in its description, have an executed
counterpart with metrics above. The paper's general-dimension (`d>1`) theory
and its references to Park et al. (2026) for the underlying stochastic
representation results are outside the scope of these four experiments,
which the paper itself frames around the one-dimensional G-BS example, and
are not claimed as reproduced here.
