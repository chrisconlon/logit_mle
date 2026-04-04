"""
Test diversion matrices by comparing two independent computations:

1. **Product removal**: Remove product j from the choice set, recompute shares,
   and compute D_jk = (s_k' - s_k) / s_j  (the definition of diversion).

2. **Analytic formula**: Each model's closed-form or integration-based formula.

These must agree to numerical precision for all three model types.
"""
import numpy as np
import jax
import jax.numpy as jnp
import pytest

from logit_mle import Logit, NestedLogit, RandomCoefficients


# ── Helpers ──────────────────────────────────────────────────────

def diversion_by_removal(model, theta, J):
    """Compute diversion by literally removing each product and recomputing shares.

    For each j, set availability[j, :] = False, recompute shares, and compute
    D_jk = (s_k_new - s_k_base) / s_j_base.

    Uses full availability (single market) so all markets are identical.
    """
    avail_full = jnp.ones((J, 1), dtype=bool)
    s_base = model._compute_shares(jnp.asarray(theta), avail_full)[:, 0]  # (J,)

    D = np.zeros((J, J))
    for j in range(J):
        avail_j = avail_full.at[j, :].set(False)
        s_new = model._compute_shares(jnp.asarray(theta), avail_j)[:, 0]
        D[j, :] = np.array((s_new - s_base) / s_base[j])
        D[j, j] = 0.0

    return D


def make_logit(J_in=10, seed=42):
    """Create a Logit model with J_in inside goods + 1 outside good."""
    rng = np.random.RandomState(seed)
    J = J_in + 1
    avail = np.ones((J, 1), dtype=bool)
    theta = jnp.array(rng.randn(J_in) * 0.5)
    model = Logit(avail)
    return model, theta, J


def make_nested_logit(J_per_nest=4, G=3, rho_cardell=0.5, seed=42):
    """Create a NestedLogit with G nests of J_per_nest products each."""
    rng = np.random.RandomState(seed)
    J_in = J_per_nest * G
    J = J_in + 1

    nesting_ids = np.repeat(np.arange(G), J_per_nest)  # (J_in,)
    avail = np.ones((J, 1), dtype=bool)

    deltas = rng.randn(J_in) * 0.5
    sigma_train = 1.0 - rho_cardell
    theta = jnp.array(np.concatenate([deltas, [sigma_train]]))

    model = NestedLogit(avail, nesting_ids=nesting_ids)
    return model, theta, J


def make_rc(J_in=8, G=2, I=50, seed=42):
    """Create a RandomCoefficients model with G characteristics and I quad nodes."""
    rng = np.random.RandomState(seed)
    J = J_in + 1

    avail = np.ones((J, 1), dtype=bool)
    x2 = rng.randn(J, G)
    nu_i = rng.randn(I, G)
    w_i = np.ones(I) / I  # uniform weights for simplicity

    deltas = rng.randn(J_in) * 0.5
    sigmas = np.abs(rng.randn(G)) * 0.3
    theta = jnp.array(np.concatenate([deltas, sigmas]))

    model = RandomCoefficients(avail, x2=x2, nu_i=nu_i, w_i=w_i)
    return model, theta, J


# ── Tests ────────────────────────────────────────────────────────

class TestLogitDiversion:

    def test_removal_matches_formula(self):
        model, theta, J = make_logit()
        D_formula = np.array(model.diversion_matrix(theta))
        D_removal = diversion_by_removal(model, theta, J)

        np.fill_diagonal(D_formula, 0.0)
        np.testing.assert_allclose(D_removal, D_formula, atol=1e-10,
            err_msg="Logit: product removal and closed-form disagree")

    def test_rows_sum_to_one(self):
        model, theta, J = make_logit()
        D = np.array(model.diversion_matrix(theta))
        np.fill_diagonal(D, 0.0)
        row_sums = D.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-10,
            err_msg="Logit: diversion rows should sum to 1")

    def test_iia_formula(self):
        """D_jk = s_k / (1 - s_j) for all j != k."""
        model, theta, J = make_logit()
        s = np.array(model.shares(theta)[:, 0])
        D = np.array(model.diversion_matrix(theta))

        for j in range(J):
            for k in range(J):
                if k == j:
                    continue
                expected = s[k] / (1.0 - s[j])
                np.testing.assert_allclose(D[j, k], expected, atol=1e-10,
                    err_msg=f"Logit IIA: D[{j},{k}] wrong")


class TestNestedLogitDiversion:

    @pytest.mark.parametrize("rho", [0.25, 0.5, 0.75])
    def test_removal_matches_formula(self, rho):
        model, theta, J = make_nested_logit(rho_cardell=rho)
        D_formula = np.array(model.diversion_matrix(theta))
        D_removal = diversion_by_removal(model, theta, J)

        np.fill_diagonal(D_formula, 0.0)
        np.testing.assert_allclose(D_removal, D_formula, atol=1e-6,
            err_msg=f"NL (rho={rho}): removal and formula disagree")

    @pytest.mark.parametrize("rho", [0.25, 0.5, 0.75])
    def test_vmap_matches_formula(self, rho):
        model, theta, J = make_nested_logit(rho_cardell=rho)
        D_formula = np.array(model.compute_diversion_matrix_from_formula(theta))
        D_vmap = np.array(model.compute_diversion_matrix_vmap(theta))

        np.fill_diagonal(D_formula, 0.0)
        np.fill_diagonal(D_vmap, 0.0)
        np.testing.assert_allclose(D_vmap, D_formula, atol=1e-6,
            err_msg=f"NL (rho={rho}): vmap and formula disagree")

    @pytest.mark.parametrize("rho", [0.0, 0.25, 0.5, 0.75])
    def test_rows_sum_to_one(self, rho):
        model, theta, J = make_nested_logit(rho_cardell=rho)
        D = np.array(model.diversion_matrix(theta))
        np.fill_diagonal(D, 0.0)
        row_sums = D.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-6,
            err_msg=f"NL (rho={rho}): rows should sum to 1")

    def test_reduces_to_logit_at_rho_zero(self):
        """When rho=0 (sigma=1), NL diversion = logit diversion = s_k/(1-s_j)."""
        model, theta, J = make_nested_logit(rho_cardell=0.0)
        s = np.array(model.shares(theta)[:, 0])
        D = np.array(model.diversion_matrix(theta))
        np.fill_diagonal(D, 0.0)

        for j in range(J):
            for k in range(J):
                if k == j:
                    continue
                expected = s[k] / (1.0 - s[j])
                np.testing.assert_allclose(D[j, k], expected, atol=1e-6,
                    err_msg=f"NL(rho=0): D[{j},{k}] should match logit IIA")

    @pytest.mark.parametrize("rho", [0.25, 0.5, 0.75])
    def test_within_nest_exceeds_cross_nest(self, rho):
        """Within-nest diversion rate (D_jk/s_k) > cross-nest rate."""
        model, theta, J = make_nested_logit(rho_cardell=rho, J_per_nest=4, G=3)
        D = np.array(model.diversion_matrix(theta))
        s = np.array(model.shares(theta)[:, 0])
        np.fill_diagonal(D, 0.0)

        J_in = J - 1
        nesting_ids = np.repeat(np.arange(3), 4)
        g_of = np.concatenate([nesting_ids, [3]])  # outside good in nest 3

        for j in range(J_in):
            within = [D[j, k] / s[k] for k in range(J) if k != j and g_of[k] == g_of[j]]
            cross = [D[j, k] / s[k] for k in range(J) if k != j and g_of[k] != g_of[j]]
            if within and cross:
                assert max(within) > max(cross), (
                    f"Row {j}: within-nest rate should exceed cross-nest rate"
                )


class TestRandomCoefficientsDiversion:

    def test_removal_matches_formula(self):
        model, theta, J = make_rc()
        D_formula = np.array(model.diversion_matrix(theta))
        D_removal = diversion_by_removal(model, theta, J)

        np.fill_diagonal(D_formula, 0.0)
        np.testing.assert_allclose(D_removal, D_formula, atol=1e-6,
            err_msg="RC: product removal and integration formula disagree")

    def test_rows_sum_to_one(self):
        model, theta, J = make_rc()
        D = np.array(model.diversion_matrix(theta))
        np.fill_diagonal(D, 0.0)
        row_sums = D.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-6,
            err_msg="RC: diversion rows should sum to 1")

    def test_uniform_sigma_reduces_to_logit(self):
        """When sigma=0, RC should give logit diversion."""
        rng = np.random.RandomState(99)
        J_in, G, I = 6, 2, 20
        J = J_in + 1

        avail = np.ones((J, 1), dtype=bool)
        x2 = rng.randn(J, G)
        nu_i = rng.randn(I, G)
        w_i = np.ones(I) / I

        deltas = rng.randn(J_in) * 0.5
        sigmas = np.zeros(G)  # no heterogeneity
        theta = jnp.array(np.concatenate([deltas, sigmas]))

        model = RandomCoefficients(avail, x2=x2, nu_i=nu_i, w_i=w_i)
        D_rc = np.array(model.diversion_matrix(theta))
        np.fill_diagonal(D_rc, 0.0)

        # Should match logit IIA
        s = np.array(model.shares(theta)[:, 0])
        for j in range(J):
            for k in range(J):
                if k == j:
                    continue
                expected = s[k] / (1.0 - s[j])
                np.testing.assert_allclose(D_rc[j, k], expected, atol=1e-6,
                    err_msg=f"RC(sigma=0): D[{j},{k}] should match logit IIA")
