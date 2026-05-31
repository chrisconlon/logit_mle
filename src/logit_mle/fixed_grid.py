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

from .random_coefficients import RandomCoefficients, _s_ijt, _s_jt, _diversion_jk
from .quadrature import beta_grid_box


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


@jax.jit
def _project_simplex(v):
    """Euclidean projection onto {x >= 0, sum x = 1} (Duchi et al. 2008)."""
    R = v.shape[0]
    u = jnp.sort(v)[::-1]
    css = jnp.cumsum(u) - 1.0
    idx = jnp.arange(1, R + 1)
    rho = jnp.sum(u - css / idx > 0)
    tau = css[rho - 1] / rho
    return jnp.maximum(v - tau, 0.0)


def _power_lambda_max(Gram, n_iter=50):
    """Largest eigenvalue of a PSD matrix via power iteration (O(R^2) per step)."""
    R = Gram.shape[0]
    v0 = jnp.ones(R) / jnp.sqrt(R)
    v = jax.lax.fori_loop(
        0, n_iter, lambda i, v: (lambda w: w / jnp.linalg.norm(w))(Gram @ v), v0
    )
    return v @ (Gram @ v)


def make_pgd_solver(Gram, b, *, lmax_iter=50):
    """Pure-JAX accelerated projected gradient (FISTA) for the same QP as
    :func:`make_qp_solver`; returns ``solve(mu, tol=..., max_iter=...) -> theta``.

    No factorization (the gradient ``2(Gram theta + mu theta - b)`` is an O(R^2)
    matvec) and no cvxpy dependency.  Intended for large ``R`` on a CUDA GPU,
    where the matvecs and a ``vmap`` over ``mu`` parallelize; on CPU it is
    neutral-to-slower than warm-started OSQP (see ``docs/hho_design.md`` §11).
    Convergence is adaptive: stop when ``||theta_{k+1} - theta_k|| < tol`` or at
    ``max_iter``.  ``lambda_max(Gram)`` (for the step size) is computed once and
    reused across ``mu`` since ``lambda_max(Gram + mu I) = lambda_max(Gram) + mu``.
    """
    Gram = jnp.asarray(Gram, dtype=float)
    Gram = 0.5 * (Gram + Gram.T)
    b = jnp.asarray(b, dtype=float)
    R = Gram.shape[0]
    lam = _power_lambda_max(Gram, lmax_iter)

    @jax.jit
    def solve(mu, tol=1e-9, max_iter=100_000):
        mu = jnp.asarray(mu, dtype=float)
        step = 1.0 / (2.0 * (lam + mu))

        def grad(x):                       # 2 (P x - b),  P = Gram + mu I
            return 2.0 * (Gram @ x + mu * x - b)

        th0 = jnp.full(R, 1.0 / R)
        init = (th0, th0, 1.0, jnp.array(0), jnp.array(jnp.inf))

        def cond(c):
            _, _, _, k, gap = c
            return (k < max_iter) & (gap > tol)

        def body(c):
            th, y, t, k, _ = c
            th1 = _project_simplex(y - step * grad(y))
            t1 = 0.5 * (1.0 + jnp.sqrt(1.0 + 4.0 * t * t))
            y1 = th1 + ((t - 1.0) / t1) * (th1 - th)
            return (th1, y1, t1, k + 1, jnp.linalg.norm(th1 - th))

        th, _, _, _, _ = jax.lax.while_loop(cond, body, init)
        return th

    return solve


_SOLVER_MAKERS = {"cvxpy": make_qp_solver, "pgd": make_pgd_solver}


def solve_qp(Gram, b, mu, *, backend="cvxpy", **opts):
    """One-off solve of the nonnegative elastic-net QP.

    ``backend="cvxpy"`` (default, needs the ``[hho]`` extra) or ``"pgd"`` (pure
    JAX).  For a ``mu`` sweep, build the solver once with :func:`make_qp_solver`
    or :func:`make_pgd_solver` instead.
    """
    if backend not in _SOLVER_MAKERS:
        raise NotImplementedError(
            f"solver backend {backend!r} not in {sorted(_SOLVER_MAKERS)}"
        )
    return np.asarray(_SOLVER_MAKERS[backend](Gram, b)(mu, **opts))


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
    backend="cvxpy",
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
        # 25 points incl. mu=0 (FKRB). Kept modest: each solve re-factorizes an R×R
        # KKT (factorization-bound), so cost is linear in the number of mu points;
        # 25 is ample for one-SE selection.
        mu_grid = np.r_[0.0, np.logspace(-6.0, 1.0, 24)]
    mu_grid = np.unique(np.asarray(mu_grid, dtype=float))      # sorted ascending
    solver_opts = solver_opts or {}
    if backend not in _SOLVER_MAKERS:
        raise NotImplementedError(
            f"solver backend {backend!r} not in {sorted(_SOLVER_MAKERS)}"
        )
    maker = _SOLVER_MAKERS[backend]

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
        solve = maker(Ztr.T @ Ztr, Ztr.T @ ytr)
        for m, mu in enumerate(mu_grid):                       # ascending -> warm-started (cvxpy)
            theta = np.asarray(solve(mu, **solver_opts))
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

    solve_full = maker(Z.T @ Z, Z.T @ s_obs)
    theta_hat = np.asarray(solve_full(mu_selected, **solver_opts))

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


# ── Step 4: public estimator + result ────────────────────────────

class FixedGridRC:
    """Fixed-grid nonparametric random-coefficient estimator (FKRB / HHO).

    Second stage on a fitted ``RandomCoefficients``: holds the step-1 mean
    utilities ``(delta, xi)`` fixed and recovers a nonparametric mixing
    distribution over a fixed grid of deviation vectors by cross-validated
    nonnegative elastic net.  Construct from raw step-1 arrays, or from a fitted
    model via :meth:`from_fitted`.
    """

    def __init__(
        self,
        delta_hat,
        sigma_hat,
        xi_hat=None,
        *,
        x2,
        availability_matrix,
        q_jt,
        market_fe=False,
        beta_grid=None,
        drop_cols=(),
        grid_size=200,
        span=3.0,
        grid_seed=0,
        solver="cvxpy",
    ):
        """
        Parameters
        ----------
        delta_hat : array, shape (J-1,) or (J,)
            Step-1 mean utilities.  If length ``J`` the outside good (last) is
            dropped (it is normalized to 0).
        sigma_hat : array, shape (G,)
            Step-1 RC std devs; used only to size the default grid.
        xi_hat : array, shape (T,) or (T-1,), optional
            Step-1 per-market outside-good utility (last market normalized to 0).
            Ignored when ``market_fe=False`` (treated as zeros).
        x2 : array, shape (J, G)
        availability_matrix : array, shape (J, T)
        q_jt : array, shape (J, T)
            Observed purchase counts (-> observed shares ``q / q.sum(0)``).
        market_fe : bool
        beta_grid : array, shape (R, G_grid), optional
            Deviation grid over the gridded columns.  If None, built with
            :func:`beta_grid_box` from ``sigma_hat[keep]`` (``grid_size`` points,
            half-width ``span * sigma``).
        drop_cols : tuple[int]
            Characteristics held homogeneous (excluded from the grid).
        grid_size, span, grid_seed :
            Default-grid controls (used only when ``beta_grid is None``).
        solver : str
            Step-2 QP backend ("cvxpy").
        """
        self.x2 = np.asarray(x2, dtype=float)
        self.avail = np.asarray(availability_matrix)
        self.q_jt = np.asarray(q_jt, dtype=float)
        self.J, self.T = self.avail.shape
        self.G = self.x2.shape[1]
        self.market_fe = bool(market_fe)
        self.drop_cols = tuple(drop_cols)
        if solver not in _SOLVER_MAKERS:
            raise ValueError(f"solver must be one of {sorted(_SOLVER_MAKERS)}, got {solver!r}")
        self.solver = solver

        delta_hat = np.asarray(delta_hat, dtype=float)
        if delta_hat.shape[0] == self.J:          # drop normalized outside good
            delta_hat = delta_hat[: self.J - 1]
        elif delta_hat.shape[0] != self.J - 1:
            raise ValueError(
                f"delta_hat must have length J-1={self.J - 1} or J={self.J}, "
                f"got {delta_hat.shape[0]}"
            )
        self.delta_inside = delta_hat

        self.sigma_hat = np.asarray(sigma_hat, dtype=float)
        self.xi = self._normalize_xi(xi_hat)

        keep = [g for g in range(self.G) if g not in self.drop_cols]
        if beta_grid is None:
            beta_grid = beta_grid_box(
                self.sigma_hat[keep], grid_size, span=span, seed=grid_seed
            )
        self.beta_grid = np.asarray(beta_grid, dtype=float)

    def _normalize_xi(self, xi_hat):
        """Return ξ as a length-T vector (zeros if no market FE; last = 0)."""
        if not self.market_fe:
            return np.zeros(self.T)
        if xi_hat is None:
            return np.zeros(self.T)
        xi_hat = np.asarray(xi_hat, dtype=float)
        if xi_hat.shape[0] == self.T:
            return xi_hat
        if xi_hat.shape[0] == self.T - 1:
            return np.append(xi_hat, 0.0)
        raise ValueError(
            f"xi_hat must have length T={self.T} or T-1={self.T - 1}, "
            f"got {xi_hat.shape[0]}"
        )

    @classmethod
    def from_fitted(cls, rc_model, step1_result, *, beta_grid=None, drop_cols=(),
                    grid_size=200, span=3.0, grid_seed=0, solver="cvxpy"):
        """Build from a fitted ``RandomCoefficients`` and its ``OptimizeResult``."""
        p = rc_model._unpack_theta(np.asarray(step1_result.x))
        return cls(
            np.asarray(p["delta_inside"]),
            np.asarray(p["sigma"]),
            np.asarray(p["xi"]),
            x2=np.asarray(rc_model.x2),
            availability_matrix=np.asarray(rc_model.availability_matrix),
            q_jt=np.asarray(rc_model.q_jt),
            market_fe=rc_model.market_fe,
            beta_grid=beta_grid,
            drop_cols=drop_cols,
            grid_size=grid_size,
            span=span,
            grid_seed=grid_seed,
            solver=solver,
        )

    def fit(self, *, mu_grid=None, k=10, train_pct=0.9, random_state=2025,
            select="one_se", solver_opts=None):
        """Cross-validate ``mu`` and refit on all markets.  Returns a FixedGridResult."""
        cv = cross_validate_mu(
            self.delta_inside, self.xi, self.x2, self.beta_grid, self.avail, self.q_jt,
            drop_cols=self.drop_cols, mu_grid=mu_grid, k=k, train_pct=train_pct,
            random_state=random_state, select=select, backend=self.solver,
            solver_opts=solver_opts,
        )
        return FixedGridResult(
            delta_inside=self.delta_inside, xi=self.xi, x2=self.x2, avail=self.avail,
            q_jt=self.q_jt, beta_grid=self.beta_grid, theta_hat=cv.theta_hat,
            drop_cols=self.drop_cols, market_fe=self.market_fe, cv=cv,
        )


class FixedGridResult:
    """A fitted nonparametric mixing distribution ``{(beta_r, theta_r)}``.

    Downstream objects reuse the existing ``RandomCoefficients`` machinery with
    the grid as integration nodes and ``theta_hat`` as weights, so they are
    directly comparable to the normal-RC run.
    """

    def __init__(self, *, delta_inside, xi, x2, avail, q_jt, beta_grid, theta_hat,
                 drop_cols, market_fe, cv):
        self.delta_inside = np.asarray(delta_inside)
        self.xi = np.asarray(xi)
        self.x2 = np.asarray(x2)
        self.avail = np.asarray(avail)
        self.q_jt = np.asarray(q_jt)
        self.beta_grid = np.asarray(beta_grid)
        self.theta_hat = np.asarray(theta_hat)
        self.drop_cols = tuple(drop_cols)
        self.market_fe = bool(market_fe)
        self.cv = cv
        self.mu_selected = cv.mu_selected
        self.J, self.T = self.avail.shape
        self.G = self.x2.shape[1]
        self.sigma_vec, self.nu_full = grid_to_rc_inputs(
            self.beta_grid, self.drop_cols, self.G
        )
        self._rc_cache = None

    def f_beta(self):
        """The estimated mixing distribution: ``(beta_grid, theta_hat)``."""
        return self.beta_grid, self.theta_hat

    def shares(self):
        """Predicted market shares under the nonparametric f(beta). Shape (J, T)."""
        s = _s_jt(self.delta_inside, self.sigma_vec, self.xi, self.x2,
                  self.nu_full, self.theta_hat, self.avail)
        return np.asarray(s)

    def diversion_matrix(self):
        """Diversion ratios at the xi=0 baseline, full availability. Shape (J, J)."""
        avail_full = jnp.ones((self.J, self.T), dtype=bool)
        D = _diversion_jk(self.delta_inside, self.sigma_vec, self.x2,
                          self.nu_full, self.theta_hat, avail_full)
        return np.asarray(D)

    def to_random_coefficients(self):
        """Return ``(rc, theta_vec)``: an equivalent RandomCoefficients + its theta.

        The mixing distribution is represented as integration nodes ``nu_full``
        with weights ``theta_hat`` and ``sigma`` the gridded/dropped indicator.
        """
        if self._rc_cache is None:
            rc = RandomCoefficients(
                self.avail, self.q_jt, x2=self.x2,
                nu_i=np.asarray(self.nu_full), w_i=np.asarray(self.theta_hat),
                market_fe=self.market_fe,
            )
            parts = [np.asarray(self.delta_inside), np.asarray(self.sigma_vec)]
            if self.market_fe:
                parts.append(np.asarray(self.xi[:-1]))   # inside xi (last normalized)
            self._rc_cache = (rc, np.concatenate(parts))
        return self._rc_cache

    def elasticity_matrix(self, *, prices, price_coeff, price_col=None):
        """Price elasticity matrix under the nonparametric f(beta). Shape (J, J)."""
        rc, theta_vec = self.to_random_coefficients()
        return np.asarray(rc.elasticity_matrix(
            theta_vec, prices=prices, price_coeff=price_coeff, price_col=price_col
        ))
