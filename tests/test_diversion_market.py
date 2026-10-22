"""
Tests for ``diversion_market``: evaluating diversion at an estimated market's xi.

By default diversion is evaluated at xi = 0, the reference market whose xi is
normalized. With ``diversion_market=t`` it is evaluated at xi_t. Because shares
depend only on inside utilities relative to the outside utility, D at xi_t equals
the default D with every delta shifted by -xi_t; and for a non-reference t a
common shift of every delta and every free xi leaves D unchanged, so the
diversion term of the augmented likelihood no longer pins the level.
``diversion_market="mean"`` evaluates at the mean of the free xi (every market
but the reference), which no single market's sales can move cheaply.
"""
import numpy as np
import jax
import jax.numpy as jnp
import pytest

from logit_mle import Logit, NestedLogit, RandomCoefficients

J_IN, T, G, I = 7, 6, 2, 300
J = J_IN + 1


def data(seed=0):
    rng = np.random.RandomState(seed)
    avail = rng.rand(J, T) < 0.8
    avail[-1] = True
    q = rng.randint(5, 80, (J, T)).astype(float) * avail
    return rng, avail, q


def build(kind, diversion_market=None, seed=0):
    """Return (model, theta, number of nonlinear parameters)."""
    rng, avail, q = data(seed)
    xi = rng.randn(T - 1) * 0.5
    delta = rng.randn(J_IN) * 0.5 - 1.5
    if kind == "logit":
        model = Logit(avail, q, market_fe=True, diversion_market=diversion_market)
        return model, jnp.array(np.concatenate([delta, xi])), 0
    if kind == "nested":
        ids = np.array([0, 0, 0, 1, 1, 2, 2])          # no single-product nest
        model = NestedLogit(avail, q, nesting_ids=ids, market_fe=True,
                            diversion_market=diversion_market)
        return model, jnp.array(np.concatenate([delta, [0.4], xi])), 1
    x2 = rng.randn(J, G)
    x2[-1] = 0.0
    nu, w = rng.randn(I, G), np.ones(I) / I
    model = RandomCoefficients(avail, q, x2=x2, nu_i=nu, w_i=w, market_fe=True,
                               diversion_market=diversion_market)
    return model, jnp.array(np.concatenate([delta, [0.6, 0.9], xi])), G


def shift(theta, k, c, delta=True, xi=True):
    th = np.array(theta)
    if delta:
        th[:J_IN] += c
    if xi:
        th[J_IN + k:] += c
    return jnp.array(th)


KINDS = ["logit", "nested", "rc"]


@pytest.mark.parametrize("kind", KINDS)
def test_default_and_reference_market_agree(kind):
    """None, T - 1 and -1 all evaluate at the reference market (xi = 0)."""
    m0, theta, _ = build(kind)
    D0 = np.array(m0.diversion_matrix(theta))
    for dm in (T - 1, -1):
        m, _, _ = build(kind, diversion_market=dm)
        np.testing.assert_allclose(np.array(m.diversion_matrix(theta)), D0, rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("t", [0, 3])
def test_equals_default_with_delta_shifted_by_xi_t(kind, t):
    m_t, theta, k = build(kind, diversion_market=t)
    m0, _, _ = build(kind)
    xi_t = float(theta[J_IN + k + t])
    D_t = np.array(m_t.diversion_matrix(theta))
    D_ref = np.array(m0.diversion_matrix(shift(theta, k, -xi_t, xi=False)))
    np.testing.assert_allclose(D_t, D_ref, rtol=1e-10, atol=1e-12)
    assert np.abs(D_t - np.array(m0.diversion_matrix(theta))).max() > 1e-4  # xi_t != 0 matters


@pytest.mark.parametrize("kind", KINDS)
def test_mean_equals_default_with_delta_shifted_by_mean_free_xi(kind):
    m_mean, theta, k = build(kind, diversion_market="mean")
    m0, _, _ = build(kind)
    xi_bar = float(np.mean(np.array(theta[J_IN + k:])))      # the free xi; the reference is not in theta
    np.testing.assert_allclose(np.array(m_mean.diversion_matrix(theta)),
                               np.array(m0.diversion_matrix(shift(theta, k, -xi_bar, xi=False))),
                               rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("dm", [2, "mean"])
def test_common_shift_leaves_diversion_unchanged(kind, dm):
    """Shift every delta and every free xi by c: D at a non-reference market, or at
    the mean of the free xi, is unchanged; D at the reference market (the default)
    moves."""
    m_t, theta, k = build(kind, diversion_market=dm)
    m0, _, _ = build(kind)
    shifted = shift(theta, k, 0.7)
    np.testing.assert_allclose(np.array(m_t.diversion_matrix(shifted)),
                               np.array(m_t.diversion_matrix(theta)), rtol=1e-10, atol=1e-12)
    assert np.abs(np.array(m0.diversion_matrix(shifted))
                  - np.array(m0.diversion_matrix(theta))).max() > 1e-3


@pytest.mark.parametrize("kind", KINDS)
def test_xi_t_enters_the_gradient(kind):
    m_t, theta, k = build(kind, diversion_market=1)
    g = jax.grad(lambda th: jnp.sum(m_t._compute_diversion(th)[:-1, :-1] ** 2))(theta)
    assert abs(float(g[J_IN + k + 1])) > 1e-8
    others = np.delete(np.array(g[J_IN + k:]), 1)
    assert np.all(others == 0.0)


@pytest.mark.parametrize("kind", KINDS)
def test_mean_spreads_the_gradient_equally_over_free_xi(kind):
    m, theta, k = build(kind, diversion_market="mean")
    g = jax.grad(lambda th: jnp.sum(m._compute_diversion(th)[:-1, :-1] ** 2))(theta)
    g_xi = np.array(g[J_IN + k:])
    assert np.abs(g_xi).min() > 1e-10
    np.testing.assert_allclose(g_xi, g_xi[0], rtol=1e-10)


@pytest.mark.parametrize("dm", [None, 2, "mean"])
def test_nested_logit_fit_objective_matches_log_likelihood(dm):
    """NestedLogit.fit builds its own objective in gamma space; at the optimum it
    must equal the shared log_likelihood, diversion term included."""
    m_true, theta, _ = build("nested", diversion_market=dm, seed=1)
    D = np.array(m_true.diversion_matrix(theta))
    np.fill_diagonal(D, 0.0)
    D = D / D.sum(axis=1, keepdims=True)
    rng, avail, q = data(1)
    ids = np.array([0, 0, 0, 1, 1, 2, 2])
    model = NestedLogit(avail, q, nesting_ids=ids, market_fe=True, diversion_data=D,
                        diversion_market=dm)
    r = model.fit(aug_div=True, penalty=1e-2, diversion_rows=(0, 3), ftol=1e-10, verbose=False)
    ll = float(model.log_likelihood(jnp.array(r.x), aug_div=True, penalty=1e-2, diversion_rows=(0, 3)))
    np.testing.assert_allclose(r.fun, ll, rtol=1e-9)


def test_mean_with_one_market_is_the_reference():
    rng = np.random.RandomState(3)
    avail = np.ones((J, 1), dtype=bool)
    theta = jnp.array(rng.randn(J_IN) * 0.5 - 1.5)
    D_mean = Logit(avail, diversion_market="mean").diversion_matrix(theta)
    np.testing.assert_allclose(np.array(D_mean), np.array(Logit(avail).diversion_matrix(theta)), rtol=1e-14)


def test_out_of_range_market_raises():
    _, avail, q = data()
    with pytest.raises(ValueError):
        Logit(avail, q, market_fe=True, diversion_market="median")
    with pytest.raises(ValueError):
        Logit(avail, q, market_fe=True, diversion_market=T)
    with pytest.raises(ValueError):
        Logit(avail, q, market_fe=True, diversion_market=-T - 1)
