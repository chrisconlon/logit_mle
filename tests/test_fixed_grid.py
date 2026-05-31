"""
Fixed-grid (FKRB / HHO) second stage: grid construction and design matrix.

Two things are validated here, before any solver is wired up:

1. ``beta_grid_box`` lays space-filling deviation points over the intended box
   (no inverse-normal transform, no weights).
2. The ``sigma_vec`` / ``nu_full`` construction used to build the design matrix
   reproduces the intended deviation ``sum_{g in keep} x_jg * beta_rg`` exactly
   — i.e. reusing ``_s_ijt`` with ``sigma`` as a gridded/dropped indicator is
   correct — and a simplex mixture of the grid-point shares equals the
   ``RandomCoefficients`` shares evaluated at ``(nu_i=nu_full, w_i=theta,
   sigma=sigma_vec)``.  This is the "(nodes, weights)" reuse the design relies on.
"""
import numpy as np
import pytest

from logit_mle import (
    RandomCoefficients, beta_grid_box, build_design_matrix, halton_draws,
    FixedGridRC, FixedGridResult,
)
from logit_mle.fixed_grid import (
    grid_to_rc_inputs, solve_qp, make_qp_solver, make_market_folds, cross_validate_mu,
)
from logit_mle.random_coefficients import _diversion_jk


# ── Helpers ──────────────────────────────────────────────────────

def make_problem(J_in=6, T=3, G=3, R=40, *, drop_cols=(),
                 nonzero_xi=False, partial_avail=False, seed=0):
    """A small fixed-grid problem: characteristics, step-1 means, a grid, weights."""
    rng = np.random.RandomState(seed)
    J = J_in + 1
    x2 = rng.randn(J, G)
    x2[J - 1] = 0.0                                   # outside good characteristics = 0
    delta_inside = rng.uniform(-3.0, -1.0, J_in)
    if nonzero_xi:
        xi = rng.uniform(-1.0, 1.0, T)
        xi[-1] = 0.0                                  # last market normalized (RC convention)
    else:
        xi = np.zeros(T)
    avail = np.ones((J, T), dtype=bool)
    if partial_avail:
        # drop a few inside goods in some markets; outside good stays available.
        avail[1, 0] = False
        avail[2, min(1, T - 1)] = False
    sigma_hat = 0.3 + 0.5 * np.abs(rng.randn(G))

    keep = [g for g in range(G) if g not in drop_cols]
    beta_grid = beta_grid_box(sigma_hat[keep], R, seed=seed)

    w = rng.rand(R)
    theta = w / w.sum()                               # weights on the simplex

    return dict(J=J, T=T, G=G, R=R, x2=x2, delta_inside=delta_inside, xi=xi,
                avail=avail, sigma_hat=sigma_hat, beta_grid=beta_grid,
                theta=theta, drop_cols=drop_cols, keep=keep)


def direct_grid_shares(delta_inside, xi, x2, beta_grid, drop_cols, avail):
    """Reference grid-point shares computed directly in numpy (no JAX reuse).

    Mirrors ``_v_ijt``: mean utility ``delta_jt`` plus deviation
    ``sum_{g in keep} x_jg * beta_rg``, then softmax over products.
    """
    J, T = avail.shape
    J_in = J - 1
    G = x2.shape[1]
    keep = [g for g in range(G) if g not in drop_cols]

    delta_jt = np.concatenate(
        [np.broadcast_to(delta_inside[:, None], (J_in, T)), xi[None, :]], axis=0
    )  # (J, T)
    dev_rj = beta_grid @ x2[:, keep].T                       # (R, J)
    V = delta_jt[None, :, :] + dev_rj[:, :, None]            # (R, J, T)
    V = np.where(avail[None, :, :], V, -1e20)
    V = V - V.max(axis=1, keepdims=True)
    e = np.exp(V)
    return e / e.sum(axis=1, keepdims=True)                  # (R, J, T)


# ── beta_grid_box ────────────────────────────────────────────────

def test_beta_grid_box_shape_and_box():
    sigma_hat = np.array([0.5, 0.8, 0.3, 1.0])
    span, n = 3.0, 500
    grid = beta_grid_box(sigma_hat, n, span=span, seed=0)

    assert grid.shape == (n, sigma_hat.shape[0])
    half = span * sigma_hat
    assert np.all(grid >= -half[None, :] - 1e-12)
    assert np.all(grid <= half[None, :] + 1e-12)


def test_beta_grid_box_reproducible_and_seed_varies():
    sigma_hat = np.array([0.5, 0.8])
    a = beta_grid_box(sigma_hat, 200, seed=7)
    b = beta_grid_box(sigma_hat, 200, seed=7)
    c = beta_grid_box(sigma_hat, 200, seed=8)
    np.testing.assert_array_equal(a, b)
    assert not np.allclose(a, c)


def test_beta_grid_box_roughly_centered():
    # Low-discrepancy coverage of a symmetric box -> column means near 0.
    sigma_hat = np.array([0.5, 1.0, 0.3])
    grid = beta_grid_box(sigma_hat, 4000, span=3.0, seed=1)
    half = 3.0 * sigma_hat
    assert np.all(np.abs(grid.mean(axis=0)) < 0.05 * half)


# ── grid_to_rc_inputs ───────────────────────────────────────────────

def test_grid_to_rc_inputs_drops_columns():
    G, R = 4, 10
    beta_grid = np.arange(R * 3, dtype=float).reshape(R, 3)   # 3 kept cols
    sigma_vec, nu_full = grid_to_rc_inputs(beta_grid, drop_cols=(0,), G=G)

    np.testing.assert_array_equal(np.asarray(sigma_vec), [0.0, 1.0, 1.0, 1.0])
    assert np.asarray(nu_full).shape == (R, G)
    np.testing.assert_array_equal(np.asarray(nu_full)[:, 0], 0.0)        # dropped col zeroed
    np.testing.assert_array_equal(np.asarray(nu_full)[:, 1:], beta_grid)  # kept cols carry grid


def test_grid_to_rc_inputs_dimension_mismatch_raises():
    with pytest.raises(ValueError):
        grid_to_rc_inputs(np.zeros((5, 3)), drop_cols=(0,), G=3)  # keep=2 != grid 3 cols


# ── design matrix: the construction reproduces the intended deviation ─

@pytest.mark.parametrize("drop_cols", [(), (0,), (1, 2)])
@pytest.mark.parametrize("nonzero_xi", [False, True])
@pytest.mark.parametrize("partial_avail", [False, True])
def test_design_matrix_matches_direct_softmax(drop_cols, nonzero_xi, partial_avail):
    p = make_problem(drop_cols=drop_cols, nonzero_xi=nonzero_xi,
                     partial_avail=partial_avail, seed=3)
    Z, S_rjt = build_design_matrix(
        p["delta_inside"], p["xi"], p["x2"], p["beta_grid"], p["avail"],
        drop_cols=drop_cols,
    )
    S_ref = direct_grid_shares(
        p["delta_inside"], p["xi"], p["x2"], p["beta_grid"], drop_cols, p["avail"]
    )
    np.testing.assert_allclose(np.asarray(S_rjt), S_ref, atol=1e-10, rtol=0)

    # grid-point shares sum to 1 over products
    np.testing.assert_allclose(np.asarray(S_rjt).sum(axis=1), 1.0, atol=1e-10)

    # Z is S_rjt reshaped to (J*T, R)
    R, J, T = S_rjt.shape
    np.testing.assert_allclose(
        np.asarray(Z), np.asarray(S_rjt).reshape(R, J * T).T, atol=0
    )


def test_dropped_column_carries_no_heterogeneity():
    # Two grids identical on kept cols but differing on a dropped col must give
    # identical design matrices (the dropped col is homogeneous).
    p = make_problem(G=3, drop_cols=(0,), seed=5)
    keep = p["keep"]
    Z1, _ = build_design_matrix(
        p["delta_inside"], p["xi"], p["x2"], p["beta_grid"], p["avail"], drop_cols=(0,)
    )
    # build a "full G" grid where the dropped col is arbitrary nonzero -- but
    # build_design_matrix only sees the kept-column grid, so this is implicitly
    # enforced; assert instead that gridding a *different* dropped-col value via
    # grid_to_rc_inputs leaves shares unchanged.
    sigma_vec, nu_full = grid_to_rc_inputs(p["beta_grid"], (0,), p["G"])
    nu_perturbed = np.asarray(nu_full).copy()
    nu_perturbed[:, 0] = 7.0                          # arbitrary value in dropped col
    from logit_mle.random_coefficients import _s_ijt
    S_a = np.asarray(_s_ijt(p["delta_inside"], sigma_vec, p["xi"], p["x2"],
                            nu_full, p["avail"]))
    S_b = np.asarray(_s_ijt(p["delta_inside"], sigma_vec, p["xi"], p["x2"],
                            nu_perturbed, p["avail"]))
    np.testing.assert_allclose(S_a, S_b, atol=1e-12)


# ── reuse identity: mixture == RandomCoefficients shares ──────────

@pytest.mark.parametrize("drop_cols", [(), (0,)])
def test_mixture_equals_rc_shares(drop_cols):
    p = make_problem(drop_cols=drop_cols, seed=11)
    Z, _ = build_design_matrix(
        p["delta_inside"], p["xi"], p["x2"], p["beta_grid"], p["avail"],
        drop_cols=drop_cols,
    )
    mixture = (np.asarray(Z) @ p["theta"]).reshape(p["J"], p["T"])

    # A RandomCoefficients with nodes = nu_full, weights = theta, sigma = sigma_vec
    # integrates to exactly the same mixture (the (nodes, weights) reuse).
    sigma_vec, nu_full = grid_to_rc_inputs(p["beta_grid"], drop_cols, p["G"])
    rc = RandomCoefficients(p["avail"], x2=p["x2"],
                            nu_i=np.asarray(nu_full), w_i=p["theta"])
    theta_vec = np.concatenate([p["delta_inside"], np.asarray(sigma_vec)])
    s_rc = np.asarray(rc._compute_shares(theta_vec, rc.availability_matrix))

    np.testing.assert_allclose(s_rc, mixture, atol=1e-12)


# ── Step-2 QP solver (cvxpy): FKRB / nonnegative elastic net ─────

def _gram(Z, s):
    return Z.T @ Z, Z.T @ s


def test_solve_qp_recovers_planted_fkrb():
    """FKRB special case (mu=0): exact recovery when s_obs = Z @ theta_true."""
    pytest.importorskip("cvxpy")
    rng = np.random.RandomState(0)
    JT, R = 60, 8
    Z = rng.rand(JT, R)
    theta_true = rng.rand(R)
    theta_true /= theta_true.sum()
    Gram, b = _gram(Z, Z @ theta_true)
    theta_hat = solve_qp(Gram, b, mu=0.0)
    np.testing.assert_allclose(theta_hat, theta_true, atol=1e-5)


def test_solve_qp_simplex_constraints():
    pytest.importorskip("cvxpy")
    rng = np.random.RandomState(1)
    Z, s = rng.rand(40, 6), rng.rand(40)
    Gram, b = _gram(Z, s)
    for mu in (0.0, 1e-3, 1.0):
        th = solve_qp(Gram, b, mu=mu)
        assert th.shape == (6,)
        assert np.all(th >= -1e-12)
        np.testing.assert_allclose(th.sum(), 1.0, atol=1e-9)


def test_solve_qp_ridge_shrinks_to_uniform():
    # As mu -> infinity the ridge dominates and minimizing ||theta||^2 on the
    # simplex gives the uniform distribution.
    pytest.importorskip("cvxpy")
    rng = np.random.RandomState(2)
    Z, s = rng.rand(40, 6), rng.rand(40)
    Gram, b = _gram(Z, s)
    th = solve_qp(Gram, b, mu=1e6)
    np.testing.assert_allclose(th, np.full(6, 1.0 / 6.0), atol=1e-3)


# ── Step-3 cross-validation over mu (fold over markets) ──────────

def test_make_market_folds():
    folds = make_market_folds(20, k=5, train_pct=0.9, random_state=1)
    assert len(folds) == 5
    for train, test in folds:
        assert len(train) == int(20 * 0.9)
        assert set(train.tolist()) & set(test.tolist()) == set()
        assert set(train.tolist()) | set(test.tolist()) == set(range(20))


def test_make_market_folds_reproducible():
    a = make_market_folds(20, random_state=3)
    b = make_market_folds(20, random_state=3)
    for (ta, _), (tb, _) in zip(a, b):
        np.testing.assert_array_equal(ta, tb)


def _planted_q_jt(p, drop_cols=()):
    """True shares from a simplex mixture of grid-point shares (sum to 1 per market)."""
    Z, _ = build_design_matrix(
        p["delta_inside"], p["xi"], p["x2"], p["beta_grid"], p["avail"],
        drop_cols=drop_cols,
    )
    return np.asarray((np.asarray(Z) @ p["theta"]).reshape(p["J"], p["T"]))


def test_cross_validate_mu_mechanics():
    pytest.importorskip("cvxpy")
    p = make_problem(J_in=8, T=24, G=3, R=25, nonzero_xi=True, seed=7)
    q = _planted_q_jt(p)
    res = cross_validate_mu(
        p["delta_inside"], p["xi"], p["x2"], p["beta_grid"], p["avail"], q,
        k=5, mu_grid=np.r_[0.0, np.logspace(-4, 0, 8)], random_state=1,
    )
    n = res.mu_grid.size
    assert res.oos_mse_mean.shape == (n,)
    assert res.oos_mse_se.shape == (n,)
    assert res.mu_1se >= res.mu_argmin           # one-SE is at least as regularized
    assert res.mu_selected == res.mu_1se         # default select="one_se"
    assert res.mu_selected in res.mu_grid
    assert np.all(res.theta_hat >= -1e-12)
    np.testing.assert_allclose(res.theta_hat.sum(), 1.0, atol=1e-9)


def test_cross_validate_mu_select_argmin():
    pytest.importorskip("cvxpy")
    p = make_problem(J_in=6, T=20, G=2, R=20, nonzero_xi=True, seed=2)
    q = _planted_q_jt(p)
    res = cross_validate_mu(
        p["delta_inside"], p["xi"], p["x2"], p["beta_grid"], p["avail"], q,
        k=5, mu_grid=np.r_[0.0, np.logspace(-4, 0, 6)], random_state=5,
        select="argmin",
    )
    assert res.mu_selected == res.mu_argmin


def test_cross_validate_mu_reproducible():
    pytest.importorskip("cvxpy")
    p = make_problem(J_in=6, T=20, G=2, R=20, nonzero_xi=True, seed=2)
    q = _planted_q_jt(p)
    kw = dict(k=5, mu_grid=np.r_[0.0, np.logspace(-4, 0, 6)], random_state=5)
    a = cross_validate_mu(p["delta_inside"], p["xi"], p["x2"], p["beta_grid"],
                          p["avail"], q, **kw)
    b = cross_validate_mu(p["delta_inside"], p["xi"], p["x2"], p["beta_grid"],
                          p["avail"], q, **kw)
    np.testing.assert_array_equal(a.per_fold_oos_mse, b.per_fold_oos_mse)
    assert a.mu_selected == b.mu_selected


# ── Step-4 public class: FixedGridRC / FixedGridResult ──────────

_SMALL_MU = np.r_[0.0, np.logspace(-4, 0, 5)]


def _fit_small(p, **fit_kw):
    fg = FixedGridRC(p["delta_inside"], p["sigma_hat"], p["xi"],
                     x2=p["x2"], availability_matrix=p["avail"], q_jt=_planted_q_jt(p),
                     market_fe=True, beta_grid=p["beta_grid"])
    return fg.fit(k=4, mu_grid=_SMALL_MU, random_state=1, **fit_kw)


def test_fixedgridrc_raw_runs():
    pytest.importorskip("cvxpy")
    p = make_problem(J_in=6, T=20, G=3, R=25, nonzero_xi=True, seed=4)
    res = _fit_small(p)
    assert isinstance(res, FixedGridResult)
    s = res.shares()
    assert s.shape == (p["J"], p["T"])
    np.testing.assert_allclose(s.sum(axis=0), 1.0, atol=1e-9)
    assert res.diversion_matrix().shape == (p["J"], p["J"])
    assert np.all(res.theta_hat >= -1e-12)
    np.testing.assert_allclose(res.theta_hat.sum(), 1.0, atol=1e-9)
    assert res.mu_selected in res.cv.mu_grid


def test_fixedgridresult_shares_match_mixture():
    pytest.importorskip("cvxpy")
    p = make_problem(J_in=6, T=18, G=3, R=20, nonzero_xi=True, seed=8)
    res = _fit_small(p)
    Z, _ = build_design_matrix(p["delta_inside"], p["xi"], p["x2"], p["beta_grid"],
                               p["avail"], drop_cols=())
    mixture = (np.asarray(Z) @ res.theta_hat).reshape(p["J"], p["T"])
    np.testing.assert_allclose(res.shares(), mixture, atol=1e-9)


def test_fixedgridresult_reuse_matches_random_coefficients():
    # shares() and diversion_matrix() equal the RandomCoefficients representation
    # (nu_i = nu_full, w_i = theta_hat, sigma = sigma_vec) -- the (nodes, weights) reuse.
    pytest.importorskip("cvxpy")
    p = make_problem(J_in=6, T=18, G=3, R=20, nonzero_xi=True, seed=8)
    res = _fit_small(p)
    rc, theta_vec = res.to_random_coefficients()
    np.testing.assert_allclose(
        res.shares(),
        np.asarray(rc._compute_shares(theta_vec, rc.availability_matrix)),
        atol=1e-10,
    )
    np.testing.assert_allclose(
        res.diversion_matrix(), np.asarray(rc.diversion_matrix(theta_vec)), atol=1e-10
    )


def test_from_fitted_runs():
    # Quick RC fit, then the fixed-grid second stage from the fitted model.
    pytest.importorskip("cvxpy")
    rng = np.random.RandomState(0)
    J_in, T, G = 5, 16, 2
    J = J_in + 1
    avail = np.ones((J, T), dtype=bool)
    x2 = rng.randn(J, G)
    x2[J - 1] = 0.0
    nu_i, w_i = halton_draws(G, 200, seed=0)

    rc0 = RandomCoefficients(avail, x2=x2, nu_i=nu_i, w_i=w_i, market_fe=False)
    theta_true = np.concatenate([rng.uniform(-2.0, -1.0, J_in), np.array([0.4, 0.6])])
    q = np.asarray(rc0._compute_shares(theta_true, avail))   # synthetic shares as counts

    rc = RandomCoefficients(avail, q, x2=x2, nu_i=nu_i, w_i=w_i, market_fe=False)
    step1 = rc.fit(seed=1, verbose=False)

    fg = FixedGridRC.from_fitted(rc, step1, grid_size=60, span=3.0)
    res = fg.fit(k=4, mu_grid=np.r_[0.0, np.logspace(-3, 0, 5)], random_state=1)
    assert res.shares().shape == (J, T)
    np.testing.assert_allclose(res.shares().sum(axis=0), 1.0, atol=1e-9)
    np.testing.assert_allclose(res.theta_hat.sum(), 1.0, atol=1e-9)


# ── Step-5 capstone: normal recovery ────────────────────────────

def test_normal_recovery_capstone():
    """When the DGP is normal-RC, the nonparametric fixed-grid diversion matches the
    fitted normal-RC diversion (and tracks the truth) -- going nonparametric does not
    distort substitution.

    Only product assortment (availability) varies across markets here -- the cleanest
    identifying variation, and the favorable case (characteristics fixed, no prices, no
    market FE).  Data are noiseless, so recovery is near-exact.  With sampling noise these
    estimators need many markets to converge (see docs/hho_design.md, identification note).
    """
    pytest.importorskip("cvxpy")
    rng = np.random.RandomState(0)
    J_in, T, G = 8, 60, 2
    J = J_in + 1

    x2 = rng.randn(J, G)
    x2[J - 1] = 0.0
    avail = rng.rand(J, T) < 0.85
    avail[J - 1, :] = True                                  # outside always available
    for t in range(T):                                     # >= 2 inside goods per market
        if avail[: J - 1, t].sum() < 2:
            avail[rng.choice(J - 1, 2, replace=False), t] = True

    # True normal-RC DGP -> observed shares (as counts) and true diversion.
    nu_i, w_i = halton_draws(G, 800, seed=1)
    truth = RandomCoefficients(avail, x2=x2, nu_i=nu_i, w_i=w_i, market_fe=False)
    theta_true = np.concatenate([rng.uniform(-2.5, -1.0, J_in), np.array([1.0, 0.7])])
    q = np.asarray(truth._compute_shares(theta_true, avail))
    D_true = np.asarray(truth.diversion_matrix(theta_true))

    # Step 1: fit normal RC.
    rc = RandomCoefficients(avail, q, x2=x2, nu_i=nu_i, w_i=w_i, market_fe=False)
    step1 = rc.fit(seed=2, verbose=False)
    D_normal = np.asarray(rc.diversion_matrix(step1.x))

    # Step 2: nonparametric fixed grid.
    fg = FixedGridRC.from_fitted(rc, step1, grid_size=150, span=4.0, grid_seed=0)
    res = fg.fit(k=5, mu_grid=np.r_[0.0, np.logspace(-8, -1, 20)],
                 random_state=3, select="argmin")
    D_np = res.diversion_matrix()

    off = ~np.eye(J, dtype=bool)

    def corr(a, b):
        return float(np.corrcoef(a[off], b[off])[0, 1])

    print(f"\ncorr(np, normal)={corr(D_np, D_normal):.4f}  "
          f"corr(np, true)={corr(D_np, D_true):.4f}  "
          f"corr(normal, true)={corr(D_normal, D_true):.4f}")
    print(f"MAD(np, normal)={np.mean(np.abs(D_np[off] - D_normal[off])):.5f}  "
          f"MAD(np, true)={np.mean(np.abs(D_np[off] - D_true[off])):.5f}")

    # Nonparametric matches the fitted normal diversion, and tracks the truth.
    # (Noiseless normal data + availability identification -> near-exact recovery;
    # thresholds keep wide margin over the observed corr~1.0, MAD~3e-5.)
    assert corr(D_np, D_normal) > 0.99
    assert np.mean(np.abs(D_np[off] - D_normal[off])) < 0.005
    assert corr(D_np, D_true) > 0.99


# ── Cross-solver: identification of weights vs. diversion ────────

def test_cross_solver_agreement():
    """OSQP vs Clarabel.  On a dense grid the FKRB weights theta are non-unique
    (Gram near-singular), so the solvers need *not* agree on theta at mu=0 -- but the
    ridge identifies theta (they agree at mu>0), and *diversion* agrees regardless.
    This encodes what is identified (substitution) vs what is not (the raw weights).
    """
    cp = pytest.importorskip("cvxpy")
    p = make_problem(J_in=6, T=24, G=3, R=60, nonzero_xi=True, seed=3)
    q = _planted_q_jt(p)
    Z, _ = build_design_matrix(p["delta_inside"], p["xi"], p["x2"], p["beta_grid"],
                               p["avail"])
    Z = np.asarray(Z)
    s_obs = (q / q.sum(axis=0, keepdims=True)).reshape(-1)
    Gram, b = Z.T @ Z, Z.T @ s_obs
    sigma_vec, nu_full = grid_to_rc_inputs(p["beta_grid"], (), p["G"])
    avail_full = np.ones((p["J"], 1), dtype=bool)
    off = ~np.eye(p["J"], dtype=bool)

    def diversion(theta):
        return np.asarray(
            _diversion_jk(p["delta_inside"], sigma_vec, p["x2"], nu_full, theta, avail_full)
        )

    for mu in (0.0, 1e-3):
        th_osqp = make_qp_solver(Gram, b)(mu, solver=cp.OSQP)
        th_clar = make_qp_solver(Gram, b)(mu, solver=cp.CLARABEL)
        D_osqp, D_clar = diversion(th_osqp), diversion(th_clar)
        # diversion (the identified object) agrees across solvers regardless of mu
        assert np.corrcoef(D_osqp[off], D_clar[off])[0, 1] > 0.999
        assert np.max(np.abs(D_osqp[off] - D_clar[off])) < 0.02
        if mu > 0:
            # regularization identifies the weights -> solvers agree on theta too
            assert np.max(np.abs(th_osqp - th_clar)) < 1e-2
