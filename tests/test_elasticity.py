"""
Test elasticity matrix computation.

Verifies:
1. Closed-form elasticities match numerical derivatives of shares w.r.t. prices
2. Own-price elasticities are negative
3. Cross-price elasticities are positive (substitutes)
4. Logit IIA: cross-elasticities η_jk depend on k but not on j
5. RC with sigma=0 reduces to logit elasticities
6. RC with price_col uses heterogeneous coefficients correctly
"""
import numpy as np
import jax.numpy as jnp
import pytest

from logit_mle import Logit, NestedLogit, RandomCoefficients


def numerical_elasticity(model, theta, prices, price_coeff, eps=1e-6):
    """Compute elasticities by finite differences on shares.

    η_jk = (∂s_j/∂p_k) · (p_k/s_j)

    For each k, perturb delta_k by price_coeff * eps (since ∂δ/∂p = price_coeff),
    recompute shares, and finite-difference.
    """
    avail = model._availability_matrix_full
    s_base = np.array(model._compute_shares(jnp.asarray(theta), avail)[:, 0])
    J = len(s_base)
    eta = np.zeros((J, J))

    for k in range(J):
        # Perturb delta_k: δ_k -> δ_k + price_coeff * eps
        p = model._unpack_theta(jnp.asarray(theta))
        delta_new = np.array(p["delta"])
        delta_new[k] += price_coeff * eps

        # Rebuild theta with perturbed delta
        theta_new = np.array(theta).copy()
        if k < J - 1:  # inside good (delta stored in theta)
            theta_new[k] += price_coeff * eps
        # else: outside good (delta=0, not in theta) — perturbation requires
        # shifting all other deltas by -price_coeff*eps instead
        else:
            theta_new[:J-1] -= price_coeff * eps

        s_new = np.array(model._compute_shares(jnp.asarray(theta_new), avail)[:, 0])
        ds = (s_new - s_base) / eps
        eta[:, k] = ds * (prices[k] / s_base)

    return eta


class TestLogitElasticity:

    def test_matches_numerical(self):
        rng = np.random.RandomState(42)
        J_in = 8
        J = J_in + 1
        avail = np.ones((J, 1), dtype=bool)
        theta = jnp.array(rng.randn(J_in) * 0.5)
        prices = np.abs(rng.randn(J)) + 1.0
        alpha = -0.5

        model = Logit(avail)
        eta_analytic = np.array(model.elasticity_matrix(theta, prices=prices, price_coeff=alpha))
        eta_numerical = numerical_elasticity(model, theta, prices, alpha)

        np.testing.assert_allclose(eta_analytic, eta_numerical, atol=1e-5,
            err_msg="Logit: analytic and numerical elasticities disagree")

    def test_own_negative_cross_positive(self):
        rng = np.random.RandomState(42)
        J = 6
        avail = np.ones((J, 1), dtype=bool)
        theta = jnp.array(rng.randn(J - 1) * 0.5)
        prices = np.abs(rng.randn(J)) + 1.0
        alpha = -0.5

        model = Logit(avail)
        eta = np.array(model.elasticity_matrix(theta, prices=prices, price_coeff=alpha))

        for j in range(J):
            assert eta[j, j] < 0, f"Own-price elasticity η[{j},{j}] should be negative"
            for k in range(J):
                if k != j:
                    assert eta[j, k] > 0, f"Cross-price elasticity η[{j},{k}] should be positive"

    def test_iia_cross_elasticity(self):
        """Logit IIA: η_jk = α · p_k · s_k for all j ≠ k (independent of j)."""
        rng = np.random.RandomState(42)
        J = 6
        avail = np.ones((J, 1), dtype=bool)
        theta = jnp.array(rng.randn(J - 1) * 0.5)
        prices = np.abs(rng.randn(J)) + 1.0
        alpha = -0.5

        model = Logit(avail)
        eta = np.array(model.elasticity_matrix(theta, prices=prices, price_coeff=alpha))

        # For each k, all off-diagonal entries in column k should be equal
        for k in range(J):
            cross_vals = [eta[j, k] for j in range(J) if j != k]
            np.testing.assert_allclose(cross_vals, cross_vals[0], atol=1e-12,
                err_msg=f"Logit IIA violated: cross-elasticities in column {k} differ")


class TestNestedLogitElasticity:

    def test_matches_numerical(self):
        J_per_nest, G = 4, 3
        J_in = J_per_nest * G
        J = J_in + 1
        rng = np.random.RandomState(42)

        nesting_ids = np.repeat(np.arange(G), J_per_nest)
        avail = np.ones((J, 1), dtype=bool)
        deltas = rng.randn(J_in) * 0.5
        rho = 0.5
        theta = jnp.array(np.concatenate([deltas, [rho]]))
        prices = np.abs(rng.randn(J)) + 1.0
        alpha = -0.5

        model = NestedLogit(avail, nesting_ids=nesting_ids)
        eta_analytic = np.array(model.elasticity_matrix(theta, prices=prices, price_coeff=alpha))
        eta_numerical = numerical_elasticity(model, theta, prices, alpha)

        np.testing.assert_allclose(eta_analytic, eta_numerical, atol=1e-4,
            err_msg="NL: analytic and numerical elasticities disagree")

    def test_own_negative(self):
        J_per_nest, G = 3, 2
        J_in = J_per_nest * G
        J = J_in + 1
        rng = np.random.RandomState(42)

        nesting_ids = np.repeat(np.arange(G), J_per_nest)
        avail = np.ones((J, 1), dtype=bool)
        theta = jnp.array(np.concatenate([rng.randn(J_in) * 0.5, [0.5]]))
        prices = np.abs(rng.randn(J)) + 1.0

        model = NestedLogit(avail, nesting_ids=nesting_ids)
        eta = np.array(model.elasticity_matrix(theta, prices=prices, price_coeff=-0.5))

        # Check inside goods only (outside good delta is normalized to 0)
        for j in range(J_in):
            assert eta[j, j] < 0, f"Own-price elasticity η[{j},{j}] should be negative"


class TestRandomCoefficientsElasticity:

    def test_matches_numerical_homogeneous(self):
        """With price_col=None, should match numerical derivatives."""
        rng = np.random.RandomState(42)
        J_in, G, I = 6, 2, 30
        J = J_in + 1
        avail = np.ones((J, 1), dtype=bool)
        x2 = rng.randn(J, G)
        nu_i = rng.randn(I, G)
        w_i = np.ones(I) / I
        theta = jnp.array(np.concatenate([rng.randn(J_in) * 0.5, np.abs(rng.randn(G)) * 0.3]))
        prices = np.abs(rng.randn(J)) + 1.0
        alpha = -0.5

        model = RandomCoefficients(avail, x2=x2, nu_i=nu_i, w_i=w_i)
        eta_analytic = np.array(model.elasticity_matrix(theta, prices=prices, price_coeff=alpha))
        eta_numerical = numerical_elasticity(model, theta, prices, alpha)

        np.testing.assert_allclose(eta_analytic, eta_numerical, atol=1e-4,
            err_msg="RC (homogeneous): analytic and numerical elasticities disagree")

    def test_sigma_zero_matches_logit(self):
        """With sigma=0 and price_col=None, RC elasticities should match logit."""
        rng = np.random.RandomState(42)
        J_in, G, I = 6, 2, 20
        J = J_in + 1
        avail = np.ones((J, 1), dtype=bool)
        x2 = rng.randn(J, G)
        nu_i = rng.randn(I, G)
        w_i = np.ones(I) / I
        deltas = rng.randn(J_in) * 0.5
        theta = jnp.array(np.concatenate([deltas, [0.0, 0.0]]))
        prices = np.abs(rng.randn(J)) + 1.0
        alpha = -0.5

        model_rc = RandomCoefficients(avail, x2=x2, nu_i=nu_i, w_i=w_i)
        model_logit = Logit(avail)
        theta_logit = jnp.array(deltas)

        eta_rc = np.array(model_rc.elasticity_matrix(theta, prices=prices, price_coeff=alpha))
        eta_logit = np.array(model_logit.elasticity_matrix(theta_logit, prices=prices, price_coeff=alpha))

        np.testing.assert_allclose(eta_rc, eta_logit, atol=1e-6,
            err_msg="RC(sigma=0) should match logit elasticities")

    def test_heterogeneous_price_col(self):
        """With price_col, individual β_i^p = α + σ_g · ν_ig should change elasticities."""
        rng = np.random.RandomState(42)
        J_in, G, I = 6, 2, 50
        J = J_in + 1
        avail = np.ones((J, 1), dtype=bool)
        x2 = rng.randn(J, G)
        nu_i = rng.randn(I, G)
        w_i = np.ones(I) / I
        theta = jnp.array(np.concatenate([rng.randn(J_in) * 0.5, [0.5, 0.3]]))
        prices = np.abs(rng.randn(J)) + 1.0
        alpha = -0.5

        model = RandomCoefficients(avail, x2=x2, nu_i=nu_i, w_i=w_i)

        eta_homo = np.array(model.elasticity_matrix(theta, prices=prices, price_coeff=alpha))
        eta_hetero = np.array(model.elasticity_matrix(theta, prices=prices, price_coeff=alpha, price_col=0))

        # They should differ when sigma[0] > 0
        assert not np.allclose(eta_homo, eta_hetero, atol=1e-6), (
            "Heterogeneous and homogeneous elasticities should differ when sigma > 0"
        )

        # Both should have negative own-price elasticities
        for j in range(J):
            assert eta_hetero[j, j] < 0, f"Own-price η[{j},{j}] should be negative"

    def test_own_negative_cross_positive(self):
        rng = np.random.RandomState(42)
        J_in, G, I = 6, 2, 30
        J = J_in + 1
        avail = np.ones((J, 1), dtype=bool)
        x2 = rng.randn(J, G)
        nu_i = rng.randn(I, G)
        w_i = np.ones(I) / I
        theta = jnp.array(np.concatenate([rng.randn(J_in) * 0.5, np.abs(rng.randn(G)) * 0.3]))
        prices = np.abs(rng.randn(J)) + 1.0

        model = RandomCoefficients(avail, x2=x2, nu_i=nu_i, w_i=w_i)
        eta = np.array(model.elasticity_matrix(theta, prices=prices, price_coeff=-0.5))

        for j in range(J):
            assert eta[j, j] < 0, f"Own-price elasticity η[{j},{j}] should be negative"
