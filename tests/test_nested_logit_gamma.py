"""
Tests targeting the (gamma = delta/(1-rho)) internal reparameterization
in NestedLogit.

Three properties we want to lock in:

1. **Multi-seed robustness.** Under the gamma substitution, ``fit()`` should
   converge to the same basin from many random starting values, with no
   NaN/non-finite objectives. This is the core reason for the substitution.

2. **gamma <-> delta round-trip.** ``gamma_from_theta`` and ``theta_from_gamma``
   should be exact inverses for arbitrary ``(delta, rho, xi)`` vectors.

3. **rho = 0 boundary reduces to plain Logit.** When the nesting parameter is
   exactly zero (Berry/Cardell), NL shares must equal plain logit shares with
   the same delta.
"""
import numpy as np
import jax.numpy as jnp
import pytest

from logit_mle import Logit, NestedLogit


# ── Shared helpers ──────────────────────────────────────────────

def simulate_choices(s_jt, I, seed=42):
    rng = np.random.RandomState(seed)
    J, T = s_jt.shape
    q_jt = np.zeros((J, T), dtype=np.float64)
    for t in range(T):
        q_jt[:, t] = rng.multinomial(I, s_jt[:, t])
    return q_jt


def make_partial_availability(J, T, frac_available=0.8, seed=123):
    rng = np.random.RandomState(seed)
    avail = np.zeros((J, T), dtype=bool)
    for t in range(T):
        n = max(3, int(frac_available * (J - 1)))
        idx = rng.choice(J - 1, n, replace=False)
        avail[idx, t] = True
    avail[J - 1, :] = True
    return avail


# ── 1. Multi-seed robustness ────────────────────────────────────

class TestMultiSeedRobustness:

    def _setup(self):
        """Synthesize a small NL fitting problem with known truth."""
        rng = np.random.RandomState(2026)
        J_per_nest, G = 4, 3
        J_in = J_per_nest * G
        J = J_in + 1
        T, I = 40, 8000

        delta_true = rng.randn(J_in) * 1.0
        rho_true = 0.45
        nesting_ids = np.repeat(np.arange(G), J_per_nest)
        avail = make_partial_availability(J, T, frac_available=0.8)

        theta_true = jnp.array(np.concatenate([delta_true, [rho_true]]))
        model_truth = NestedLogit(avail, nesting_ids=nesting_ids)
        s_jt = np.array(model_truth.shares(theta_true))
        q_jt = simulate_choices(s_jt, I)

        return avail, q_jt, nesting_ids, J_in, rho_true

    def test_fit_robust_across_many_seeds(self):
        """Every seed should converge to the same basin with finite objective.

        This is the headline test for the gamma substitution. Before the
        refactor, several seeds (e.g. 2025) hit NaN; the gamma version
        converges from any reasonable starting value.
        """
        avail, q_jt, nesting_ids, J_in, rho_true = self._setup()

        seeds = [2025, 42, 314, 161, 2024, 777, 999, 123]
        rho_hats, objectives = [], []

        for seed in seeds:
            model = NestedLogit(avail, q_jt, nesting_ids=nesting_ids)
            res = model.fit(seed=seed, verbose=False)
            assert np.isfinite(res.fun), (
                f"seed={seed}: objective is NaN/Inf -- gamma reparam should "
                f"prevent this"
            )
            rho_hat = float(res.x[J_in])
            rho_hats.append(rho_hat)
            objectives.append(float(res.fun))

        # All seeds should land at essentially the same objective
        obj_arr = np.array(objectives)
        obj_spread = obj_arr.max() - obj_arr.min()
        assert obj_spread < 1.0, (
            f"Objective spread across seeds is {obj_spread:.4f} -- "
            f"seeds are landing in different basins"
        )

        # All seeds should agree on rho to within ~0.05
        rho_arr = np.array(rho_hats)
        rho_spread = rho_arr.max() - rho_arr.min()
        assert rho_spread < 0.05, (
            f"rho spread across seeds is {rho_spread:.4f}: {rho_hats}"
        )

        # And the recovered rho should be close to truth
        np.testing.assert_allclose(
            rho_arr.mean(), rho_true, atol=0.08,
            err_msg=f"recovered rho = {rho_arr.mean():.4f}, true = {rho_true}"
        )


# ── 2. Round-trip gamma <-> delta helpers ────────────────────────

class TestGammaDeltaRoundTrip:

    @pytest.mark.parametrize("rho", [0.0, 0.1, 0.3, 0.5, 0.8, 0.95])
    def test_round_trip_no_fe(self, rho):
        rng = np.random.RandomState(7)
        J_per_nest, G = 3, 3
        J_in = J_per_nest * G
        J = J_in + 1
        nesting_ids = np.repeat(np.arange(G), J_per_nest)
        avail = np.ones((J, 1), dtype=bool)

        delta = rng.randn(J_in) * 0.7
        theta = jnp.array(np.concatenate([delta, [rho]]))

        model = NestedLogit(avail, nesting_ids=nesting_ids)

        gamma_theta = model.gamma_from_theta(theta)
        theta_back = model.theta_from_gamma(gamma_theta)

        np.testing.assert_allclose(np.array(theta_back), np.array(theta),
                                   atol=1e-12,
                                   err_msg=f"round-trip mismatch at rho={rho}")

        # Check the relationship gamma = delta/(1-rho)
        gamma_inside = np.array(gamma_theta[:J_in])
        np.testing.assert_allclose(gamma_inside, delta / (1.0 - rho), atol=1e-12)

    @pytest.mark.parametrize("rho", [0.0, 0.25, 0.6, 0.9])
    def test_round_trip_with_fe(self, rho):
        rng = np.random.RandomState(11)
        J_per_nest, G = 3, 2
        J_in = J_per_nest * G
        J = J_in + 1
        T = 5
        nesting_ids = np.repeat(np.arange(G), J_per_nest)
        avail = np.ones((J, T), dtype=bool)

        delta = rng.randn(J_in) * 0.5
        xi = rng.randn(T - 1) * 0.4
        theta = jnp.array(np.concatenate([delta, [rho], xi]))

        model = NestedLogit(avail, nesting_ids=nesting_ids, market_fe=True)

        gamma_theta = model.gamma_from_theta(theta)
        theta_back = model.theta_from_gamma(gamma_theta)

        np.testing.assert_allclose(np.array(theta_back), np.array(theta),
                                   atol=1e-12,
                                   err_msg=f"round-trip mismatch at rho={rho}")

        # Verify gamma_xi = xi / (1 - rho)
        gamma_xi_inside = np.array(gamma_theta[J_in + 1:])
        np.testing.assert_allclose(gamma_xi_inside, xi / (1.0 - rho), atol=1e-12)

    def test_shares_invariant_to_round_trip(self):
        """Shares should be the same whether we pass theta directly or
        round-trip it through gamma form."""
        rng = np.random.RandomState(13)
        J_per_nest, G = 4, 3
        J_in = J_per_nest * G
        J = J_in + 1
        nesting_ids = np.repeat(np.arange(G), J_per_nest)
        avail = np.ones((J, 1), dtype=bool)

        delta = rng.randn(J_in) * 0.8
        rho = 0.4
        theta = jnp.array(np.concatenate([delta, [rho]]))

        model = NestedLogit(avail, nesting_ids=nesting_ids)
        s_direct = np.array(model.shares(theta))

        gamma_theta = model.gamma_from_theta(theta)
        theta_back = model.theta_from_gamma(gamma_theta)
        s_round = np.array(model.shares(theta_back))

        np.testing.assert_allclose(s_round, s_direct, atol=1e-14)


# ── 3. rho = 0 boundary reduces to plain logit ───────────────────

class TestRhoZeroBoundary:

    def test_shares_match_logit_at_rho_zero(self):
        """NL with rho = 0 should give the same shares as plain Logit
        for the same delta."""
        rng = np.random.RandomState(17)
        J_per_nest, G = 3, 4
        J_in = J_per_nest * G
        J = J_in + 1
        T = 6
        nesting_ids = np.repeat(np.arange(G), J_per_nest)
        avail = make_partial_availability(J, T, frac_available=0.7, seed=99)

        delta = rng.randn(J_in) * 0.6
        theta_nl = jnp.array(np.concatenate([delta, [0.0]]))
        theta_logit = jnp.array(delta)

        model_nl = NestedLogit(avail, nesting_ids=nesting_ids)
        model_logit = Logit(avail)

        s_nl = np.array(model_nl.shares(theta_nl))
        s_logit = np.array(model_logit.shares(theta_logit))

        np.testing.assert_allclose(
            s_nl, s_logit, atol=1e-12,
            err_msg="NL at rho=0 should give plain logit shares"
        )

    def test_diversion_match_logit_at_rho_zero(self):
        """NL diversion at rho = 0 should match plain Logit IIA diversion."""
        rng = np.random.RandomState(19)
        J_per_nest, G = 4, 3
        J_in = J_per_nest * G
        J = J_in + 1
        nesting_ids = np.repeat(np.arange(G), J_per_nest)
        avail = np.ones((J, 1), dtype=bool)

        delta = rng.randn(J_in) * 0.5
        theta_nl = jnp.array(np.concatenate([delta, [0.0]]))
        theta_logit = jnp.array(delta)

        model_nl = NestedLogit(avail, nesting_ids=nesting_ids)
        model_logit = Logit(avail)

        D_nl = np.array(model_nl.diversion_matrix(theta_nl))
        D_logit = np.array(model_logit.diversion_matrix(theta_logit))

        # Both diagonals are formula-defined and not necessarily equal; zero
        # them out before comparing.
        np.fill_diagonal(D_nl, 0.0)
        np.fill_diagonal(D_logit, 0.0)

        np.testing.assert_allclose(
            D_nl, D_logit, atol=1e-10,
            err_msg="NL diversion at rho=0 should match logit IIA"
        )

    def test_jacobian_invariant_to_xi_with_market_fe(self):
        """With market_fe=True, the Jacobian should be evaluated at ξ=0,
        so changing the ξ entries of theta should not change the Jacobian."""
        rng = np.random.RandomState(37)
        J_per_nest, G = 3, 3
        J_in = J_per_nest * G
        J = J_in + 1
        T = 6
        nesting_ids = np.repeat(np.arange(G), J_per_nest)
        avail = np.ones((J, T), dtype=bool)

        delta = rng.randn(J_in) * 0.5
        rho = 0.4
        xi_a = rng.randn(T - 1) * 0.7
        xi_b = rng.randn(T - 1) * 0.7

        theta_a = jnp.array(np.concatenate([delta, [rho], xi_a]))
        theta_b = jnp.array(np.concatenate([delta, [rho], xi_b]))

        model = NestedLogit(avail, nesting_ids=nesting_ids, market_fe=True)
        jac_a = np.array(model.jacobian(theta_a))
        jac_b = np.array(model.jacobian(theta_b))

        np.testing.assert_allclose(
            jac_a, jac_b, atol=1e-12,
            err_msg="NL Jacobian should be invariant to xi (ξ=0 baseline)"
        )

    def test_fit_handles_rho_at_lower_boundary(self):
        """fit() should not blow up when the truth is exactly rho = 0."""
        rng = np.random.RandomState(23)
        J_per_nest, G = 4, 3
        J_in = J_per_nest * G
        J = J_in + 1
        T, I = 30, 8000

        delta_true = rng.randn(J_in) * 1.0
        rho_true = 0.0  # boundary
        nesting_ids = np.repeat(np.arange(G), J_per_nest)
        avail = make_partial_availability(J, T, frac_available=0.8, seed=29)

        theta_true = jnp.array(np.concatenate([delta_true, [rho_true]]))
        model_truth = NestedLogit(avail, nesting_ids=nesting_ids)
        s_jt = np.array(model_truth.shares(theta_true))
        q_jt = simulate_choices(s_jt, I, seed=31)

        model = NestedLogit(avail, q_jt, nesting_ids=nesting_ids)
        res = model.fit(seed=2025, verbose=False)

        assert np.isfinite(res.fun), "objective should be finite at rho=0 truth"

        rho_hat = float(res.x[J_in])
        # Should land near the boundary, not bounce away to a wrong basin
        assert rho_hat < 0.15, (
            f"rho_hat = {rho_hat:.4f}, expected near 0"
        )
