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

from logit_mle import RandomCoefficients, beta_grid_box, build_design_matrix
from logit_mle.fixed_grid import (
    grid_to_rc_inputs, solve_qp, make_market_folds, cross_validate_mu,
)


# ── Helpers ──────────────────────────────────────────────────────

def make_problem(J_in=6, T=3, G=3, R=40, *, drop_cols=(),
                 nonzero_xi=False, partial_avail=False, seed=0):
    """A small fixed-grid problem: characteristics, step-1 means, a grid, weights."""
    rng = np.random.RandomState(seed)
    J = J_in + 1
    x2 = rng.randn(J, G)
    x2[J - 1] = 0.0                                   # outside good characteristics = 0
    delta_inside = rng.uniform(-3.0, -1.0, J_in)
    xi = rng.uniform(-1.0, 1.0, T) if nonzero_xi else np.zeros(T)  # outside-good utility
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
