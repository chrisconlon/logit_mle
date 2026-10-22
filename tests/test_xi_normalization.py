"""
Tests for ``xi_normalization``: "last" (xi_T = 0, the default) and "mean_zero"
(the T market effects average to zero, so xi = 0 is the average market).

Shares depend only on delta - xi, so an estimate under "last" re-expressed in
mean-zero form (subtract the mean of all T market effects from delta and xi)
must reproduce every market's shares under "mean_zero", and the mean-zero
diversion must equal the "last" diversion evaluated at the average market.
"""
import numpy as np
import jax
import jax.numpy as jnp
import pytest

from logit_mle import Logit, NestedLogit, RandomCoefficients
from logit_mle.base import complete_xi

J_IN, T, G, I = 7, 6, 2, 300
J = J_IN + 1
KINDS = ["logit", "nested", "rc"]


def data(seed=0):
    rng = np.random.RandomState(seed)
    avail = rng.rand(J, T) < 0.8
    avail[-1] = True
    q = rng.randint(5, 80, (J, T)).astype(float) * avail
    return rng, avail, q


def build(kind, xi_normalization="last", seed=0, diversion_data=None):
    """Return (model, theta, number of nonlinear parameters)."""
    rng, avail, q = data(seed)
    xi = rng.randn(T - 1) * 0.5
    delta = rng.randn(J_IN) * 0.5 - 1.5
    kw = dict(market_fe=True, diversion_data=diversion_data, xi_normalization=xi_normalization)
    if kind == "logit":
        return Logit(avail, q, **kw), jnp.array(np.concatenate([delta, xi])), 0
    if kind == "nested":
        ids = np.array([0, 0, 0, 1, 1, 2, 2])
        return (NestedLogit(avail, q, nesting_ids=ids, **kw),
                jnp.array(np.concatenate([delta, [0.4], xi])), 1)
    x2 = rng.randn(J, G)
    x2[-1] = 0.0
    nu, w = rng.randn(I, G), np.ones(I) / I
    return (RandomCoefficients(avail, q, x2=x2, nu_i=nu, w_i=w, **kw),
            jnp.array(np.concatenate([delta, [0.6, 0.9], xi])), G)


def to_mean_zero(theta, k):
    """Re-express a "last"-normalized theta in mean-zero form."""
    th = np.array(theta)
    xi_full = np.append(th[J_IN + k:], 0.0)
    xi_bar = xi_full.mean()
    th[:J_IN] -= xi_bar
    th[J_IN + k:] -= xi_bar
    return jnp.array(th), xi_bar


def test_complete_xi():
    free = jnp.array([0.3, -1.2, 0.5])
    np.testing.assert_allclose(np.array(complete_xi(free, "last")), [0.3, -1.2, 0.5, 0.0])
    full = np.array(complete_xi(free, "mean_zero"))
    np.testing.assert_allclose(full[:3], [0.3, -1.2, 0.5])
    assert abs(full.sum()) < 1e-15
    for norm in ("last", "mean_zero"):                       # one market: no free xi
        np.testing.assert_allclose(np.array(complete_xi(jnp.zeros(0), norm)), [0.0])


@pytest.mark.parametrize("kind", KINDS)
def test_mean_zero_reproduces_last_shares(kind):
    m_last, theta, k = build(kind)
    m_mz, _, _ = build(kind, "mean_zero")
    theta_mz, _ = to_mean_zero(theta, k)
    np.testing.assert_allclose(np.array(m_mz.shares(theta_mz)), np.array(m_last.shares(theta)),
                               rtol=1e-11, atol=1e-14)
    assert abs(float(jnp.sum(m_mz._unpack_theta(theta_mz)["xi"]))) < 1e-12
    np.testing.assert_allclose(float(m_mz.log_likelihood(theta_mz)), float(m_last.log_likelihood(theta)),
                               rtol=1e-12)


@pytest.mark.parametrize("kind", KINDS)
def test_mean_zero_diversion_is_at_the_average_market(kind):
    m_last, theta, k = build(kind)
    m_mz, _, _ = build(kind, "mean_zero")
    theta_mz, xi_bar = to_mean_zero(theta, k)
    th_avg = np.array(theta)
    th_avg[:J_IN] -= xi_bar                                  # "last" model, delta measured from the average market
    np.testing.assert_allclose(np.array(m_mz.diversion_matrix(theta_mz)),
                               np.array(m_last.diversion_matrix(jnp.array(th_avg))), rtol=1e-10, atol=1e-12)
    assert np.abs(np.array(m_mz.diversion_matrix(theta_mz))
                  - np.array(m_last.diversion_matrix(theta))).max() > 1e-4


@pytest.mark.parametrize("kind", KINDS)
def test_diversion_does_not_respond_to_any_market_effect(kind):
    """Under both normalizations the diversion depends on delta (and rho or
    sigma) only. Under "mean_zero" that means no single market can move the
    point where diversion is evaluated: shifting the level requires every
    market's xi to move with delta, which the sum-to-zero constraint forbids."""
    for norm in ("last", "mean_zero"):
        m, theta, k = build(kind, norm)
        g = jax.grad(lambda th: jnp.sum(m._compute_diversion(th)[:-1, :-1] ** 2))(theta)
        assert np.all(np.array(g[J_IN + k:]) == 0.0)
    m, theta, k = build(kind, "mean_zero")
    shifted = np.array(theta)
    shifted[:J_IN] += 0.5
    shifted[J_IN + k:] += 0.5                               # every free xi moves with delta ...
    full = np.array(m._unpack_theta(jnp.array(shifted))["xi"])
    assert abs(full[-1] - float(-np.sum(shifted[J_IN + k:]))) < 1e-12   # ... so the last one must move against them
    assert np.abs(np.array(m.shares(jnp.array(shifted)))[:, -1] - np.array(m.shares(theta))[:, -1]).max() > 1e-3


@pytest.mark.parametrize("norm", ["last", "mean_zero"])
def test_nested_logit_fit_objective_matches_log_likelihood(norm):
    """NestedLogit.fit builds its own gamma-space objective; at the optimum it
    must equal the shared log_likelihood, diversion term included."""
    m_true, theta, _ = build("nested", seed=1)
    D = np.array(m_true.diversion_matrix(theta))
    np.fill_diagonal(D, 0.0)
    D = D / D.sum(axis=1, keepdims=True)
    model, _, _ = build("nested", norm, seed=1, diversion_data=D)
    r = model.fit(aug_div=True, penalty=1e-2, diversion_rows=(0, 3), ftol=1e-10, verbose=False)
    ll = float(model.log_likelihood(jnp.array(r.x), aug_div=True, penalty=1e-2, diversion_rows=(0, 3)))
    np.testing.assert_allclose(r.fun, ll, rtol=1e-9)
    if norm == "mean_zero":
        assert abs(float(jnp.sum(model._unpack_theta(jnp.array(r.x))["xi"]))) < 1e-10


def test_invalid_normalization_raises():
    _, avail, q = data()
    with pytest.raises(ValueError):
        Logit(avail, q, market_fe=True, xi_normalization="first")
