"""Experiment parameters for the G-Black-Scholes (G-BS) replication.

All parameters are module-level constants. Where the paper does not specify a
numerical value (grid sizes, time steps, Monte Carlo sample sizes), a comment
says so and records the value chosen for this replication.
"""

# --- Common G-BS model specification (paper Remark 3.1, Section 3.4) ---
SIGMA_LOW = 0.15   # lower volatility bound sigma_low; not specified in the paper, chosen as a representative base case
SIGMA_HIGH = 0.35  # upper volatility bound sigma_high; not specified in the paper, chosen as a representative base case
X0 = 100.0         # initial state x0 (paper: "analogous to an initial stock price"), not specified numerically in the paper

# Domain truncation D_n ~ [X_MIN, X_MAX] approximating D = (0, infinity) (Remark 3.1); not specified in the paper
X_MIN = 1e-3
X_MAX = 500.0

# --- exp_1: generator extraction via numerical differentiation (Section 3.1-3.2) ---
EXP1_N_X_PAPER = 2000          # spatial grid points for the PDE solve (paper requires N_x >= 500; not specified exactly)
EXP1_N_X_RUN = 800             # reduced grid used for the executed run to keep total runtime bounded
EXP1_T_VALUES = [0.004, 0.002, 0.001, 0.0005]  # sequence of small maturities t_n -> 0 for the generator limit; not specified in the paper
EXP1_STEPS_PER_SOLVE = 20       # implicit time steps per PDE solve; not specified in the paper
EXP1_BUMP_WIDTH_FACTOR = 0.1    # test-function cutoff width w = factor * x0 (paper's implementation notes suggest w ~ 0.1*x0)
EXP1_TEST_POINTS_X0 = [80.0, 100.0, 130.0]  # test points x0 in D; not specified in the paper
EXP1_TARGET_TRIPLES = [  # (r, p, H) targets for the generator limit; not specified in the paper
    (0.0, 0.0, 1.0),
    (0.0, 0.0, -1.0),
    (5.0, 0.5, 2.0),
    (5.0, 0.5, -2.0),
    (-3.0, -1.0, 0.5),
]
EXP1_SUBLINEARITY_TRIPLES = [  # fixed triples for (G1) sublinearity checks; not specified in the paper
    (1.0, 2.0, 3.0),
    (-2.0, 0.5, -1.0),
    (4.0, -1.5, 2.5),
    (0.5, 0.5, -0.5),
]
EXP1_SUBLINEARITY_ALPHAS = [0.5, 1.0, 2.0, 3.5]  # positive scaling factors alpha for the homogeneity check (G1)

# --- exp_2: support sets via convex duality (Section 3.2, Eq. 3.5) ---
EXP2_TEST_POINTS_X = [80.0, 100.0, 130.0]  # states x at which A(x) is constructed; not specified in the paper
EXP2_TEST_TRIPLES_U = [  # test points U=(r,p,H) for the dual-representation check; not specified in the paper
    (0.0, 0.0, 1.0),
    (0.0, 0.0, -1.0),
    (2.0, 1.0, 4.0),
    (-1.0, -2.0, -3.0),
]
EXP2_W_GRID_BOUND_PAPER = 10.0  # |r|,|p|,|H| <= bound for the discretised semi-infinite LP constraint (paper suggests [-10,10])
EXP2_W_GRID_POINTS_PAPER = 50   # grid points per axis for the discretised LP (paper suggests 50 points each)
EXP2_W_GRID_POINTS_RUN = 15     # reduced grid density used for the executed run (50^3 points is unnecessarily dense for an LP whose optimum lies on the boundary)

# --- exp_3: Monte Carlo robust valuation and PDE verification (Section 3.3-3.5) ---
EXP3_STRIKES = [80.0, 90.0, 100.0, 110.0, 120.0]  # test strikes for put/call payoffs; not specified in the paper
EXP3_MATURITY = 1.0            # maturity T for the robust-valuation comparison; not specified in the paper
EXP3_N_STEPS_PAPER = 1000      # Euler-Maruyama time steps (paper: N_s = 1000)
EXP3_N_PATHS_PAPER = 100000    # Monte Carlo paths (paper: N_mc = 100000)
EXP3_N_STEPS_RUN = 250         # reduced time steps used for the executed run
EXP3_N_PATHS_RUN = 20000       # reduced path count used for the executed run
EXP3_MC_SEED = 20240521        # fixed random seed for reproducibility; not specified in the paper
EXP3_PDE_N_X_PAPER = 1000      # PDE spatial grid points (paper: N_x = 1000)
EXP3_PDE_N_X_RUN = 600         # reduced PDE grid used for the executed run
EXP3_PDE_N_T_PAPER = 2000      # PDE time steps (paper: dt = T/2000)
EXP3_PDE_N_T_RUN = 400         # reduced PDE time steps used for the executed run
EXP3_THRESHOLDS = [85.0, 95.0, 100.0, 105.0, 115.0]  # candidate switching thresholds for state-dependent admissible volatility; not specified in the paper
EXP3_LYAPUNOV_X_GRID_POINTS = 2000  # grid points for verifying the Lyapunov condition on [X_MIN, X_MAX]; not specified in the paper

# --- exp_4: Dupire local volatility recovery (Section 3.4, Remark 3.1) ---
EXP4_STRIKES = [50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 110.0, 120.0, 130.0, 140.0, 150.0]  # paper: 11 strikes
EXP4_MATURITIES = [0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]  # paper: 7 maturities
EXP4_CURVATURE_REL_TOL = 0.05  # mask points whose K^2*d2P/dK2 falls below this fraction of its own maximum (ill-conditioned denominator); not specified in the paper


def sigma_low_state_dependent(x):
    """sigma_low(x) = 0.1 + 0.05*tanh(x/50), as given in the exp_4 data_requirements."""
    import numpy as np
    return 0.1 + 0.05 * np.tanh(np.asarray(x, dtype=float) / 50.0)


def sigma_high_state_dependent(x):
    """sigma_high(x) = 0.3 + 0.05*tanh(x/50), as given in the exp_4 data_requirements."""
    import numpy as np
    return 0.3 + 0.05 * np.tanh(np.asarray(x, dtype=float) / 50.0)


EXP4_STATE_DEP_N_X_PAPER = 1000  # PDE spatial grid points for the state-dependent surface generation; not specified in the paper
EXP4_STATE_DEP_N_X_RUN = 400     # reduced grid used for the executed run
EXP4_STATE_DEP_N_T_PER_UNIT_TIME_PAPER = 400  # PDE time steps per unit maturity; not specified in the paper
EXP4_STATE_DEP_N_T_PER_UNIT_TIME_RUN = 100     # reduced time-step density used for the executed run
