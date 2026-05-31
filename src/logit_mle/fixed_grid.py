"""
Fixed-grid nonparametric random coefficients (FKRB / HHO).

Second stage on a fitted RandomCoefficients model: holding the step-1 mean
utilities (delta, xi) fixed as data, recover a nonparametric mixing distribution
over a fixed grid of deviation vectors by constrained least squares (FKRB) /
nonnegative elastic net (HHO, 2022).

The grid-point predicted shares reuse ``random_coefficients._s_ijt`` with
``sigma`` acting as an indicator of which characteristics are gridded: a column
``g`` carries the grid deviation when ``sigma_g = 1`` and is held homogeneous
(no heterogeneity) when ``sigma_g = 0``.  This is how ``drop_cols`` is realized
without any new share math.

See ``docs/hho_design.md`` for the full design.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from .random_coefficients import _s_ijt


def grid_to_rc_inputs(beta_grid, drop_cols, G):
    """Map a kept-column grid to the ``(sigma_vec, nu_full)`` inputs ``_s_ijt`` expects.

    The grid over the gridded columns is placed into full G-space (zeros in the
    dropped columns), and ``sigma_vec`` flags which columns are gridded.

    Returns ``(sigma_vec, nu_full)`` such that
    ``_s_ijt(delta, sigma_vec, xi, x2, nu_full, avail)`` produces, for each grid
    point ``r``, the logit shares under the deviation
    ``sum_{g in keep} x_jg * beta_grid[r, g_idx]`` — i.e. the dropped columns
    contribute no heterogeneity (held homogeneous, absorbed in delta).

    Parameters
    ----------
    beta_grid : array, shape (R, G_grid)
        Deviation grid over the kept (gridded) columns, in order.
    drop_cols : tuple[int]
        Columns of the characteristic matrix held homogeneous (excluded from the
        grid).  ``G_grid = G - len(drop_cols)``.
    G : int
        Total number of characteristics in ``x2``.

    Returns
    -------
    sigma_vec : jnp.ndarray, shape (G,)
        1.0 on gridded columns, 0.0 on dropped columns.
    nu_full : jnp.ndarray, shape (R, G)
        ``beta_grid`` placed in the kept columns, 0.0 in the dropped columns.
    """
    beta_grid = jnp.asarray(beta_grid)
    R = beta_grid.shape[0]
    keep = [g for g in range(G) if g not in drop_cols]
    if len(keep) != beta_grid.shape[1]:
        raise ValueError(
            f"beta_grid has {beta_grid.shape[1]} columns but {len(keep)} are "
            f"gridded (G={G}, drop_cols={tuple(drop_cols)})."
        )
    keep_idx = jnp.array(keep, dtype=int)
    sigma_vec = jnp.zeros(G).at[keep_idx].set(1.0)
    nu_full = jnp.zeros((R, G)).at[:, keep_idx].set(beta_grid)
    return sigma_vec, nu_full


def build_design_matrix(
    delta_inside,
    xi,
    x2,
    beta_grid,
    availability_matrix,
    *,
    drop_cols=(),
):
    """Build the fixed-grid design matrix Z of grid-point predicted shares.

    Each column ``r`` of ``Z`` is the vector of predicted shares
    ``s_jt(beta_r)`` (stacked over product-markets) for grid point ``beta_r``,
    holding the step-1 mean utilities fixed.  A mixture ``Z @ theta`` with
    ``theta`` on the simplex is then the predicted aggregate share vector.

    Parameters
    ----------
    delta_inside : array, shape (J-1,)
        Step-1 inside-good mean utilities (outside good normalized to 0).
    xi : array, shape (T,)
        Step-1 per-market outside-good utility (zeros if no market FE; last
        market normalized to 0 when present).
    x2 : array, shape (J, G)
        Product characteristics (including the outside good row).
    beta_grid : array, shape (R, G_grid)
        Deviation grid over the gridded columns (see ``drop_cols``).
    availability_matrix : array, shape (J, T)
        Availability mask.
    drop_cols : tuple[int]
        Characteristics held homogeneous (excluded from the grid).

    Returns
    -------
    Z : jnp.ndarray, shape (J*T, R)
        Design matrix; column ``r`` is ``vec(s_jt(beta_r))`` flattened C-order
        over ``(j, t)`` (see note).
    S_rjt : jnp.ndarray, shape (R, J, T)
        The grid-point shares before reshaping (kept for diversion / reuse).

    Notes
    -----
    ``Z`` is ``S_rjt.reshape(R, J*T).T``; the ``(J, T)`` flattening is C-order,
    so the market index ``t`` varies fastest within the ``J*T`` axis.  The
    observed-share target must be flattened the same way:
    ``s_obs = (q_jt / q_jt.sum(0)).reshape(J*T)``.
    """
    x2 = jnp.asarray(x2)
    G = x2.shape[1]
    R = beta_grid.shape[0]
    J, T = availability_matrix.shape

    sigma_vec, nu_full = grid_to_rc_inputs(beta_grid, drop_cols, G)
    S_rjt = _s_ijt(
        jnp.asarray(delta_inside),
        sigma_vec,
        jnp.asarray(xi),
        x2,
        nu_full,
        jnp.asarray(availability_matrix),
    )  # (R, J, T)
    Z = S_rjt.reshape(R, J * T).T  # (J*T, R)
    return Z, S_rjt


# ── Step 2 solver: nonnegative elastic net (HHO) / FKRB ──────────

def _require_cvxpy():
    try:
        import cvxpy as cp
    except ImportError as e:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "cvxpy is required for the fixed-grid solver. "
            "Install it with: uv pip install 'logit_mle[hho]'"
        ) from e
    return cp


def make_qp_solver(Gram, b):
    """Build the nonnegative elastic-net QP once; return ``solve(mu, ...)``.

    Solves, in Gram form (constants dropped),

        min_theta  theta' Gram theta - 2 b' theta + mu * ||theta||^2
        s.t.       theta >= 0,   sum(theta) = 1

    i.e. HHO's nonnegative elastic net (their Eq. 10), with FKRB as the special
    case ``mu = 0``.  ``mu`` is a cvxpy ``Parameter`` so the problem canonicalizes
    once and re-solves fast (and warm-started) across a ``mu`` sweep — the intended
    use inside cross-validation.

    Parameters
    ----------
    Gram : array, shape (R, R)
        ``Z.T @ Z`` (PSD).  Symmetrized internally.
    b : array, shape (R,)
        ``Z.T @ s_obs``.

    Returns
    -------
    solve : callable
        ``solve(mu, *, solver=None, **opts) -> theta`` returning the
        simplex-projected weights (shape ``(R,)``).
    """
    cp = _require_cvxpy()
    Gram = np.asarray(Gram, dtype=float)
    Gram = 0.5 * (Gram + Gram.T)          # symmetrize away numerical asymmetry
    b = np.asarray(b, dtype=float)
    R = Gram.shape[0]

    theta = cp.Variable(R, nonneg=True)
    mu_p = cp.Parameter(nonneg=True)
    P = cp.psd_wrap(Gram)                 # trust PSD (Gram form)
    objective = cp.Minimize(
        cp.quad_form(theta, P) - 2.0 * (b @ theta) + mu_p * cp.sum_squares(theta)
    )
    prob = cp.Problem(objective, [cp.sum(theta) == 1])

    def solve(mu, *, solver=None, **opts):
        cp_solver = solver if solver is not None else cp.OSQP
        mu_p.value = float(mu)
        if cp_solver is cp.OSQP and not opts:
            opts = dict(eps_abs=1e-9, eps_rel=1e-9, max_iter=100_000)
        prob.solve(solver=cp_solver, warm_start=True, **opts)
        if prob.status not in ("optimal", "optimal_inaccurate"):
            raise RuntimeError(f"QP solve failed: status={prob.status!r} (mu={mu})")
        th = np.clip(np.asarray(theta.value, dtype=float), 0.0, None)
        total = th.sum()
        # project tiny solver-tolerance violations back onto the simplex
        return th / total if total > 0 else th

    return solve


def solve_qp(Gram, b, mu, *, backend="cvxpy", **opts):
    """One-off solve of the nonnegative elastic-net QP (see :func:`make_qp_solver`).

    For a ``mu`` sweep, build the solver once with :func:`make_qp_solver` instead.
    """
    if backend != "cvxpy":
        raise NotImplementedError(
            f"solver backend {backend!r} not implemented (cvxpy only for now)"
        )
    return make_qp_solver(Gram, b)(mu, **opts)


# ── Step 3: cross-validation over the ridge parameter mu ─────────

def make_market_folds(T, *, k=10, train_pct=0.9, random_state=2025):
    """Repeated random train/test splits over markets (PyCMS ``create_kfolds`` style).

    Each of ``k`` folds draws a ``train_pct`` fraction of the ``T`` markets
    without replacement (seed ``random_state + fold``); the complement is the
    test set.  Not a disjoint k-fold partition — independent random subsamples.

    Returns
    -------
    list[tuple[np.ndarray, np.ndarray]]
        ``[(train_markets, test_markets), ...]`` of sorted integer index arrays.
    """
    idx = np.arange(T, dtype=int)
    n_train = int(T * train_pct)
    if not (0 < n_train < T):
        raise ValueError(f"train_pct={train_pct} gives n_train={n_train} for T={T}")
    folds = []
    for fold in range(k):
        rng = np.random.default_rng(random_state + fold)
        train = np.sort(rng.choice(idx, size=n_train, replace=False))
        test = np.sort(np.setdiff1d(idx, train, assume_unique=True))
        folds.append((train, test))
    return folds


@dataclass
class CVResult:
    """Result of cross-validating the ridge parameter ``mu``.

    Attributes
    ----------
    mu_grid : np.ndarray
        Swept ridge values (ascending; includes ``0.0`` = FKRB).
    oos_mse_mean, oos_mse_std, oos_mse_se : np.ndarray
        Per-``mu`` out-of-sample share-MSE across folds (mean, sample std with
        ``ddof=1``, and standard error ``std / sqrt(k)``).
    n_folds : int
    mu_argmin : float
        ``mu`` minimizing mean OOS MSE.
    mu_1se : float
        Largest ``mu`` whose mean OOS MSE is within one SE of the minimum
        (HHO's one-standard-error rule — more regularization).
    mu_selected : float
        The chosen ``mu`` (``mu_1se`` if ``select='one_se'`` else ``mu_argmin``).
    select : str
    theta_hat : np.ndarray
        Weights refit on all markets at ``mu_selected`` (simplex).
    per_fold_oos_mse : np.ndarray
        ``(n_folds, len(mu_grid))`` raw per-fold OOS MSE.
    """
    mu_grid: np.ndarray
    oos_mse_mean: np.ndarray
    oos_mse_std: np.ndarray
    oos_mse_se: np.ndarray
    n_folds: int
    mu_argmin: float
    mu_1se: float
    mu_selected: float
    select: str
    theta_hat: np.ndarray
    per_fold_oos_mse: np.ndarray


def cross_validate_mu(
    delta_inside,
    xi,
    x2,
    beta_grid,
    availability_matrix,
    q_jt,
    *,
    drop_cols=(),
    mu_grid=None,
    k=10,
    train_pct=0.9,
    random_state=2025,
    select="one_se",
    solver_opts=None,
):
    """Cross-validate the ridge ``mu`` by out-of-sample share fit, folding over markets.

    Builds the design matrix once, then for each fold fits the nonnegative
    elastic net on the training markets (warm-started along the ascending ``mu``
    grid) and scores out-of-sample share MSE on the held-out markets.  Selects
    ``mu`` by the one-standard-error rule (default) or argmin, and refits the
    weights on all markets at the selected ``mu``.

    Observed shares are ``q_jt / q_jt.sum(0)`` (unweighted LS, per HHO).

    Returns
    -------
    CVResult
    """
    if mu_grid is None:
        mu_grid = np.r_[0.0, np.logspace(-6.0, 1.0, 50)]
    mu_grid = np.unique(np.asarray(mu_grid, dtype=float))      # sorted ascending
    solver_opts = solver_opts or {}

    availability_matrix = np.asarray(availability_matrix)
    J, T = availability_matrix.shape

    Z, _ = build_design_matrix(
        delta_inside, xi, x2, beta_grid, availability_matrix, drop_cols=drop_cols
    )
    Z = np.asarray(Z)                                          # (J*T, R)
    q_jt = np.asarray(q_jt, dtype=float)
    s_obs = (q_jt / q_jt.sum(axis=0, keepdims=True)).reshape(-1)   # (J*T,) C-order
    market_of_row = np.arange(J * T) % T                       # market index per Z row

    folds = make_market_folds(T, k=k, train_pct=train_pct, random_state=random_state)
    n_mu = mu_grid.size
    per_fold = np.empty((len(folds), n_mu))

    for f, (train, test) in enumerate(folds):
        tr = np.isin(market_of_row, train)
        te = np.isin(market_of_row, test)
        Ztr, ytr = Z[tr], s_obs[tr]
        Zte, yte = Z[te], s_obs[te]
        solve = make_qp_solver(Ztr.T @ Ztr, Ztr.T @ ytr)
        for m, mu in enumerate(mu_grid):                       # ascending -> warm-started
            theta = solve(mu, **solver_opts)
            resid = yte - Zte @ theta
            per_fold[f, m] = float(np.mean(resid ** 2))

    mean = per_fold.mean(axis=0)
    std = per_fold.std(axis=0, ddof=1) if len(folds) > 1 else np.zeros(n_mu)
    se = std / np.sqrt(len(folds))

    i_argmin = int(np.argmin(mean))
    mu_argmin = float(mu_grid[i_argmin])
    # one-SE: largest mu whose mean is within one SE of the best (most regularized)
    within = np.where(mean <= mean[i_argmin] + se[i_argmin])[0]
    mu_1se = float(mu_grid[int(within.max())])

    if select not in ("one_se", "argmin"):
        raise ValueError(f"select must be 'one_se' or 'argmin', got {select!r}")
    mu_selected = mu_1se if select == "one_se" else mu_argmin

    solve_full = make_qp_solver(Z.T @ Z, Z.T @ s_obs)
    theta_hat = solve_full(mu_selected, **solver_opts)

    return CVResult(
        mu_grid=mu_grid,
        oos_mse_mean=mean,
        oos_mse_std=std,
        oos_mse_se=se,
        n_folds=len(folds),
        mu_argmin=mu_argmin,
        mu_1se=mu_1se,
        mu_selected=mu_selected,
        select=select,
        theta_hat=theta_hat,
        per_fold_oos_mse=per_fold,
    )
