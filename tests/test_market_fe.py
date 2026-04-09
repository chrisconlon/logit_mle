"""
Parameter recovery tests for Logit and Nested Logit with market fixed effects.

For each model:
1. Fix true parameters (delta, xi, and rho for NL)
2. Compute true shares under partial availability
3. Simulate multinomial choice data from true shares
4. Estimate via MLE
5. Check that estimated parameters recover the truth
"""
import numpy as np
import jax.numpy as jnp
import pytest

from logit_mle import Logit, NestedLogit


def simulate_choices(s_jt, I, seed=42):
    """Simulate multinomial choices from shares. Returns q_jt (J, T)."""
    rng = np.random.RandomState(seed)
    J, T = s_jt.shape
    q_jt = np.zeros((J, T), dtype=np.float64)
    for t in range(T):
        choices = rng.multinomial(I, s_jt[:, t])
        q_jt[:, t] = choices
    return q_jt


def make_partial_availability(J, T, frac_available=0.7, seed=123):
    """Random partial availability matrix. Outside good always available."""
    rng = np.random.RandomState(seed)
    avail = np.zeros((J, T), dtype=bool)
    for t in range(T):
        n_avail = max(3, int(frac_available * (J - 1)))
        idx = rng.choice(J - 1, n_avail, replace=False)
        avail[idx, t] = True
    avail[J - 1, :] = True  # outside good always available
    return avail


class TestLogitMarketFE:

    def test_recovery_no_fe(self):
        """Baseline: Logit without market FE recovers delta."""
        rng = np.random.RandomState(42)
        J, T, I = 10, 50, 5000

        delta_true = rng.randn(J - 1) * 1.5
        avail = np.ones((J, T), dtype=bool)

        model_true = Logit(avail)
        theta_true = jnp.array(delta_true)
        s_jt = np.array(model_true.shares(theta_true))
        q_jt = simulate_choices(s_jt, I)

        model = Logit(avail, q_jt)
        result = model.fit(seed=2025, verbose=False)

        np.testing.assert_allclose(result.x, delta_true, atol=0.1,
            err_msg="Logit without FE: delta not recovered")

    def test_recovery_with_fe(self):
        """Logit with market_fe=True recovers both delta and xi."""
        rng = np.random.RandomState(42)
        J, T, I = 10, 30, 10000

        delta_true = rng.randn(J - 1) * 1.5
        xi_true = rng.randn(T - 1) * 0.5  # moderate market variation

        avail = make_partial_availability(J, T, frac_available=0.8, seed=99)

        # True theta = (delta[J-1], xi[T-1])
        theta_true = jnp.array(np.concatenate([delta_true, xi_true]))

        model_true = Logit(avail, market_fe=True)
        s_jt = np.array(model_true.shares(theta_true))
        q_jt = simulate_choices(s_jt, I)

        model = Logit(avail, q_jt, market_fe=True)
        result = model.fit(seed=2025, verbose=False)

        delta_hat = result.x[:J - 1]
        xi_hat = result.x[J - 1:]

        print(f"\nLogit market_fe recovery (J={J}, T={T}, I={I}):")
        print(f"  delta: max|err|={np.max(np.abs(delta_hat - delta_true)):.4f}, "
              f"RMSE={np.sqrt(np.mean((delta_hat - delta_true)**2)):.4f}")
        print(f"  xi:    max|err|={np.max(np.abs(xi_hat - xi_true)):.4f}, "
              f"RMSE={np.sqrt(np.mean((xi_hat - xi_true)**2)):.4f}")

        np.testing.assert_allclose(delta_hat, delta_true, atol=0.15,
            err_msg="Logit market_fe: delta not recovered")
        np.testing.assert_allclose(xi_hat, xi_true, atol=0.25,
            err_msg="Logit market_fe: xi not recovered")

    def test_fe_changes_shares(self):
        """Verify that xi actually affects shares (not inert)."""
        J, T = 8, 5
        rng = np.random.RandomState(42)
        avail = np.ones((J, T), dtype=bool)

        delta = rng.randn(J - 1) * 1.0
        xi = np.array([0.5, -0.5, 1.0, -1.0])  # T-1 = 4

        theta_with_xi = jnp.array(np.concatenate([delta, xi]))
        theta_no_xi = jnp.array(delta)

        model_fe = Logit(avail, market_fe=True)
        model_plain = Logit(avail)

        s_fe = np.array(model_fe.shares(theta_with_xi))
        s_plain = np.array(model_plain.shares(theta_no_xi))

        # Shares should differ across markets when xi != 0
        assert not np.allclose(s_fe, s_plain, atol=1e-6), \
            "market_fe shares should differ from plain logit when xi != 0"

        # In the last market (xi_T = 0), shares should match plain logit
        np.testing.assert_allclose(s_fe[:, -1], s_plain[:, -1], atol=1e-10,
            err_msg="Last market (xi=0) should match plain logit")

        # Outside good share should be higher when xi > 0
        # (positive xi = more attractive outside option)
        for t in range(T - 1):
            if xi[t] > 0:
                assert s_fe[J - 1, t] > s_plain[J - 1, t], \
                    f"Market {t}: xi>0 should increase outside good share"

    def test_diversion_invariant_to_xi(self):
        """Diversion should be the same regardless of xi (evaluated at xi=0)."""
        J, T = 8, 5
        rng = np.random.RandomState(42)
        avail = np.ones((J, T), dtype=bool)

        delta = rng.randn(J - 1) * 1.0
        xi = rng.randn(T - 1) * 0.5

        theta_fe = jnp.array(np.concatenate([delta, xi]))
        theta_plain = jnp.array(delta)

        model_fe = Logit(avail, market_fe=True)
        model_plain = Logit(avail)

        D_fe = np.array(model_fe.diversion_matrix(theta_fe))
        D_plain = np.array(model_plain.diversion_matrix(theta_plain))

        np.testing.assert_allclose(D_fe, D_plain, atol=1e-10,
            err_msg="Diversion should be invariant to xi")


class TestNestedLogitMarketFE:

    def test_recovery_no_fe(self):
        """Baseline: NL without market FE recovers delta and sigma."""
        rng = np.random.RandomState(42)
        J_per_nest, G_nests = 4, 3
        J_in = J_per_nest * G_nests
        J = J_in + 1
        T, I = 50, 10000

        delta_true = rng.randn(J_in) * 1.0
        rho_true = 0.4  # Berry/Cardell rho
        nesting_ids = np.repeat(np.arange(G_nests), J_per_nest)

        avail = make_partial_availability(J, T, frac_available=0.8, seed=77)

        theta_true = jnp.array(np.concatenate([delta_true, [rho_true]]))

        model_true = NestedLogit(avail, nesting_ids=nesting_ids)
        s_jt = np.array(model_true.shares(theta_true))
        q_jt = simulate_choices(s_jt, I)

        model = NestedLogit(avail, q_jt, nesting_ids=nesting_ids)
        result = model.fit(seed=2025, verbose=False)

        delta_hat = result.x[:J_in]
        rho_hat = result.x[J_in]

        print(f"\nNL no-FE recovery (J={J}, T={T}, I={I}):")
        print(f"  delta: max|err|={np.max(np.abs(delta_hat - delta_true)):.4f}")
        print(f"  rho:   true={rho_true:.4f}, hat={rho_hat:.4f}")

        np.testing.assert_allclose(delta_hat, delta_true, atol=0.15,
            err_msg="NL: delta not recovered")
        np.testing.assert_allclose(rho_hat, rho_true, atol=0.1,
            err_msg="NL: rho not recovered")

    def test_recovery_with_fe(self):
        """NL with market_fe=True recovers delta, rho, and xi."""
        rng = np.random.RandomState(42)
        J_per_nest, G_nests = 3, 3
        J_in = J_per_nest * G_nests
        J = J_in + 1
        T, I = 30, 10000

        delta_true = rng.randn(J_in) * 1.0
        rho_true = 0.5
        xi_true = rng.randn(T - 1) * 0.3
        nesting_ids = np.repeat(np.arange(G_nests), J_per_nest)

        avail = make_partial_availability(J, T, frac_available=0.8, seed=77)

        # theta = (delta[J-1], rho, xi[T-1])
        theta_true = jnp.array(np.concatenate([delta_true, [rho_true], xi_true]))

        model_true = NestedLogit(avail, nesting_ids=nesting_ids, market_fe=True)
        s_jt = np.array(model_true.shares(theta_true))
        q_jt = simulate_choices(s_jt, I)

        model = NestedLogit(avail, q_jt, nesting_ids=nesting_ids, market_fe=True)
        result = model.fit(seed=2025, verbose=False)

        delta_hat = result.x[:J_in]
        rho_hat = result.x[J_in]
        xi_hat = result.x[J_in + 1:]

        print(f"\nNL market_fe recovery (J={J}, T={T}, I={I}):")
        print(f"  delta: max|err|={np.max(np.abs(delta_hat - delta_true)):.4f}, "
              f"RMSE={np.sqrt(np.mean((delta_hat - delta_true)**2)):.4f}")
        print(f"  rho:   true={rho_true:.4f}, hat={rho_hat:.4f}, "
              f"|err|={abs(rho_hat - rho_true):.4f}")
        print(f"  xi:    max|err|={np.max(np.abs(xi_hat - xi_true)):.4f}, "
              f"RMSE={np.sqrt(np.mean((xi_hat - xi_true)**2)):.4f}")

        np.testing.assert_allclose(delta_hat, delta_true, atol=0.2,
            err_msg="NL market_fe: delta not recovered")
        np.testing.assert_allclose(rho_hat, rho_true, atol=0.15,
            err_msg="NL market_fe: rho not recovered")
        np.testing.assert_allclose(xi_hat, xi_true, atol=0.2,
            err_msg="NL market_fe: xi not recovered")

    def test_fe_changes_shares(self):
        """xi affects shares (outside good more attractive when xi > 0)."""
        J_per_nest, G_nests = 3, 2
        J_in = J_per_nest * G_nests
        J = J_in + 1
        T = 4
        rng = np.random.RandomState(42)
        avail = np.ones((J, T), dtype=bool)
        nesting_ids = np.repeat(np.arange(G_nests), J_per_nest)

        delta = rng.randn(J_in) * 0.5
        rho = 0.5
        xi = np.array([0.5, -0.5, 1.0])  # T-1 = 3

        theta_fe = jnp.array(np.concatenate([delta, [rho], xi]))
        theta_plain = jnp.array(np.concatenate([delta, [rho]]))

        model_fe = NestedLogit(avail, nesting_ids=nesting_ids, market_fe=True)
        model_plain = NestedLogit(avail, nesting_ids=nesting_ids)

        s_fe = np.array(model_fe.shares(theta_fe))
        s_plain = np.array(model_plain.shares(theta_plain))

        # Last market (xi=0) should match
        np.testing.assert_allclose(s_fe[:, -1], s_plain[:, -1], atol=1e-8,
            err_msg="Last market (xi=0) should match plain NL")

        # Other markets should differ
        assert not np.allclose(s_fe[:, 0], s_plain[:, 0], atol=1e-6)

    def test_diversion_invariant_to_xi(self):
        """Diversion at xi=0 should match plain NL diversion."""
        J_per_nest, G_nests = 3, 2
        J_in = J_per_nest * G_nests
        J = J_in + 1
        T = 4
        rng = np.random.RandomState(42)
        avail = np.ones((J, T), dtype=bool)
        nesting_ids = np.repeat(np.arange(G_nests), J_per_nest)

        delta = rng.randn(J_in) * 0.5
        rho = 0.5
        xi = rng.randn(T - 1) * 0.5

        theta_fe = jnp.array(np.concatenate([delta, [rho], xi]))
        theta_plain = jnp.array(np.concatenate([delta, [rho]]))

        model_fe = NestedLogit(avail, nesting_ids=nesting_ids, market_fe=True)
        model_plain = NestedLogit(avail, nesting_ids=nesting_ids)

        D_fe = np.array(model_fe.diversion_matrix(theta_fe))
        D_plain = np.array(model_plain.diversion_matrix(theta_plain))

        np.testing.assert_allclose(D_fe, D_plain, atol=1e-10,
            err_msg="NL diversion should be invariant to xi")
