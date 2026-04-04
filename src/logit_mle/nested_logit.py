"""
Nested Logit discrete choice model.

Parameters:
  - delta[J-1]: mean utility for inside goods (outside good normalized to 0)
  - sigma:      Train's nesting parameter in (0, 1); Berry/Cardell rho = 1 - sigma
  - xi[T-1]:    outside good's mean utility per market (only if market_fe=True;
                last market normalized to 0)

Constructor takes nesting_ids (J-1,) integer vector for inside goods.
The outside good is automatically placed in its own singleton nest.

Diversion: closed-form formula from the paper appendix (tau_in/tau_out),
evaluated at xi=0 (structural diversion).
A vmap product-removal method is also available for testing.
"""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from .base import DiscreteChoiceModel

UTILITY_PUNISH = -1e20


# ── JAX computation functions (importable for tests) ─────────────
#
# These work with a "core" theta = (delta[J-1], sigma) that has no xi.
# The outside good's utility is always 0 in these functions.
# The class handles xi by building a delta vector where delta[J-1] = xi_t.

def compute_v_gjt(theta, nest_matrix, availability_matrix):
    """Deterministic utility per nest. Shape (G, J, T).

    theta = (delta[J-1], sigma). nest_matrix is (G, J) bool.
    """
    delta = jnp.concatenate([theta[:-1], jnp.array([0.0])])
    G, J = nest_matrix.shape
    T = availability_matrix.shape[1]

    delta_reshape = delta.reshape(1, J)
    vgj_nest = jnp.where(nest_matrix, delta_reshape, UTILITY_PUNISH)

    vgjt_nest = vgj_nest.reshape(G, J, 1)
    V_gjt = jnp.where(availability_matrix.reshape(1, J, T), vgjt_nest, UTILITY_PUNISH)
    return V_gjt


def _compute_v_gjt_with_delta(delta_full, sigma, nest_matrix, availability_matrix):
    """Like compute_v_gjt but takes full delta vector directly (for market-varying delta)."""
    G, J = nest_matrix.shape
    T = availability_matrix.shape[1]

    delta_reshape = delta_full.reshape(1, J)
    vgj_nest = jnp.where(nest_matrix, delta_reshape, UTILITY_PUNISH)

    vgjt_nest = vgj_nest.reshape(G, J, 1)
    V_gjt = jnp.where(availability_matrix.reshape(1, J, T), vgjt_nest, UTILITY_PUNISH)
    return V_gjt


def compute_iv_gt(theta, V_gjt):
    """Inclusive value per nest. Shape (G, T)."""
    rho = theta[-1]
    return jax.nn.logsumexp(V_gjt / rho, axis=1)


def compute_s_gjt(theta, V_gjt):
    """Conditional share of j given nest g. Shape (G, J, T)."""
    rho = theta[-1]
    return jax.nn.softmax(V_gjt / rho, axis=1)


def compute_s_gt(theta, IV_gt):
    """Nest share. Shape (G, T)."""
    rho = theta[-1]
    return jax.nn.softmax(IV_gt * rho, axis=0)


def compute_s_jt(theta, S_gjt, S_gt):
    """Unconditional market shares. Shape (J, T)."""
    G, T = S_gt.shape
    S_jt = S_gjt * S_gt.reshape(G, 1, T)
    return S_jt.sum(axis=0)


def _shares_pipeline(theta, nest_matrix, availability_matrix):
    """Full share computation pipeline. Returns (S_gjt, S_gt, S_jt)."""
    V_gjt = compute_v_gjt(theta, nest_matrix, availability_matrix)
    IV_gt = compute_iv_gt(theta, V_gjt)
    S_gjt = compute_s_gjt(theta, V_gjt)
    S_gt = compute_s_gt(theta, IV_gt)
    S_jt = compute_s_jt(theta, S_gjt, S_gt)
    return S_gjt, S_gt, S_jt


def _shares_pipeline_delta(delta_full, sigma, nest_matrix, availability_matrix):
    """Share pipeline taking delta vector directly (for market-varying outside good)."""
    V_gjt = _compute_v_gjt_with_delta(delta_full, sigma, nest_matrix, availability_matrix)
    IV_gt = jax.nn.logsumexp(V_gjt / sigma, axis=1)
    S_gjt = jax.nn.softmax(V_gjt / sigma, axis=1)
    S_gt = jax.nn.softmax(IV_gt * sigma, axis=0)
    S_jt = (S_gjt * S_gt.reshape(S_gt.shape[0], 1, S_gt.shape[1])).sum(axis=0)
    return S_gjt, S_gt, S_jt


def _shares_mfe(delta_inside, sigma, xi, nest_matrix, availability_matrix):
    """Shares with market-varying outside good utility. Shape (J, T).

    Computes shares market-by-market with delta[J-1] = xi_t.
    """
    J_in = delta_inside.shape[0]
    T = availability_matrix.shape[1]

    def shares_at_t(t):
        delta_full = jnp.concatenate([delta_inside, xi[t:t+1]])
        avail_t = availability_matrix[:, t:t+1]
        _, _, s_jt = _shares_pipeline_delta(delta_full, sigma, nest_matrix, avail_t)
        return s_jt[:, 0]

    return jax.vmap(shares_at_t)(jnp.arange(T)).T  # (J, T)


def _diversion_from_formula(theta, nest_matrix, availability_matrix_full):
    """Closed-form diversion for nested logit at xi=0. Shape (J, J).

    Uses Berry/Cardell rho = 1 - sigma notation internally.
    Computed under full availability at t=0.
    """
    S_gjt, S_gt, S_jt = _shares_pipeline(theta, nest_matrix, availability_matrix_full)

    s_gj = S_gjt[:, :, 0]  # (G, J)
    s_g = S_gt[:, 0]       # (G,)
    s_j = S_jt[:, 0]       # (J,)

    G, J = s_gj.shape
    J_in = J - 1
    inside = jnp.arange(J_in)

    rho = 1.0 - theta[-1]  # Berry/Cardell

    # Nest assignment for each product
    g_of_all = jnp.argmax(nest_matrix.astype(jnp.int32), axis=0)  # (J,)
    g_of_in = g_of_all[:J_in]

    # Conditional share s_{j|g(j)} and nest share s_{g(j)} for inside goods
    sgj = s_gj[g_of_in, inside]
    sg = s_g[g_of_in]

    # K_j, a_j, b_j terms
    DELTA = (1.0 - sg) + sg * (1.0 - sgj) ** (1.0 - rho)
    aj = 1.0 / DELTA - 1.0
    cj = 1.0 / (DELTA * (1.0 - sgj) ** rho) - 1.0
    bj = cj - aj

    # Same-nest indicator
    M = (g_of_in[:, None] == g_of_all[None, :]).astype(s_j.dtype)  # (J_in, J)

    # Ratio s_k / s_j
    ratio = s_j[None, :] / s_j[:J_in][:, None]  # (J_in, J)

    # Inside rows + outside row
    D_in = ratio * (aj[:, None] + bj[:, None] * M)  # (J_in, J)
    D_0 = (s_j / (1.0 - s_j[-1]))[None, :]         # (1, J)
    D = jnp.vstack([D_in, D_0])                     # (J, J)

    return D


def _diversion_vmap(theta, nest_matrix, availability_matrix_full):
    """Simulation-based diversion via vmap product removal at xi=0. Shape (J, J, T).

    Slower than the formula but useful for testing.
    """
    J, T = availability_matrix_full.shape

    V_gjt = compute_v_gjt(theta, nest_matrix, availability_matrix_full)
    IV_gt = compute_iv_gt(theta, V_gjt)
    S_gjt = compute_s_gjt(theta, V_gjt)
    S_gt = compute_s_gt(theta, IV_gt)
    S_jt = compute_s_jt(theta, S_gjt, S_gt)

    def shares_with_j_removed(j):
        new_avail = availability_matrix_full.at[j, :].set(False)
        V = compute_v_gjt(theta, nest_matrix, new_avail)
        IV = compute_iv_gt(theta, V)
        Sg = compute_s_gjt(theta, V)
        Sn = compute_s_gt(theta, IV)
        return compute_s_jt(theta, Sg, Sn)

    S_rjt = jax.vmap(shares_with_j_removed)(jnp.arange(J))

    numer = S_rjt - S_jt[None, :, :]
    denom = S_jt[:, None, :]
    D_jkt = numer / denom

    row_idx, col_idx = jnp.diag_indices(J)
    D_jkt = D_jkt.at[row_idx, col_idx, :].set(0.0)
    D_jkt = jnp.nan_to_num(D_jkt, nan=0.0)

    return D_jkt


# ── NestedLogit class ────────────────────────────────────────────

class NestedLogit(DiscreteChoiceModel):
    """Nested logit model.

    Parameters
    ----------
    availability_matrix : (J, T) bool
    q_jt : (J, T) quantities, optional
    nesting_ids : (J-1,) integer array
        Nest assignment for each inside good. The outside good (last row)
        is automatically placed in its own singleton nest.
    market_fe : bool
        If True, estimate market-level outside-good utility xi[T-1].
    diversion_data : (J, J) float, optional
    """

    def __init__(
        self,
        availability_matrix,
        q_jt=None,
        *,
        nesting_ids,
        market_fe=False,
        diversion_data=None,
    ):
        super().__init__(availability_matrix, q_jt, diversion_data=diversion_data)
        nesting_ids = np.asarray(nesting_ids)
        self.nest_matrix = self._build_nest_matrix(nesting_ids)
        self.G = self.nest_matrix.shape[0]
        self.market_fe = market_fe
        self._print_info()

    def _build_nest_matrix(self, nesting_ids):
        """Build (G+1, J) bool nest matrix from (J-1,) inside-good ids."""
        J_in = self.J - 1
        assert len(nesting_ids) == J_in, (
            f"nesting_ids has length {len(nesting_ids)}, expected {J_in}"
        )
        unique_nests = np.unique(nesting_ids)
        G = len(unique_nests)

        # Map nest labels to contiguous 0..G-1
        nest_map = {int(n): i for i, n in enumerate(unique_nests)}

        mat = np.zeros((G + 1, self.J), dtype=bool)
        for j, nid in enumerate(nesting_ids):
            mat[nest_map[int(nid)], j] = True
        mat[G, self.J - 1] = True  # outside good in singleton nest

        return jnp.array(mat)

    def _print_info(self):
        print("NESTED LOGIT")
        w = 18
        print(f"{'J (Products):':<{w}} {self.J}")
        print(f"{'T (Markets):':<{w}} {self.T}")
        print(f"{'G (Nests):':<{w}} {self.G}")

    def _unpack_theta(self, theta):
        delta_inside = theta[:self.J - 1]
        delta = jnp.concatenate([delta_inside, jnp.array([0.0])])
        sigma = theta[self.J - 1]
        if self.market_fe:
            xi = jnp.concatenate([theta[self.J:], jnp.array([0.0])])
        else:
            xi = jnp.zeros(self.T)
        return {"delta": delta, "delta_inside": delta_inside, "sigma": sigma, "xi": xi}

    def _core_theta(self, theta):
        """Build (delta[J-1], sigma) for the free functions (no xi)."""
        return jnp.concatenate([theta[:self.J - 1], theta[self.J - 1:self.J]])

    def _compute_shares(self, theta, avail):
        p = self._unpack_theta(theta)
        if self.market_fe:
            return _shares_mfe(p["delta_inside"], p["sigma"], p["xi"],
                               self.nest_matrix, avail)
        else:
            core = self._core_theta(theta)
            _, _, S_jt = _shares_pipeline(core, self.nest_matrix, avail)
            return S_jt

    def _compute_diversion(self, theta):
        """Closed-form diversion at xi=0 under full availability. Shape (J, J)."""
        core = self._core_theta(theta)
        return _diversion_from_formula(
            core, self.nest_matrix, self._availability_matrix_full
        )

    def _make_x0(self, rng):
        delta = rng.uniform(-10, 10, self.J - 1)
        sigma = rng.uniform(0.01, 0.99)
        if self.market_fe:
            xi = rng.uniform(-1, 1, self.T - 1)
            return jnp.array(np.concatenate([delta, [sigma], xi]))
        return jnp.array(np.concatenate([delta, [sigma]]))

    def _theta_bounds(self):
        delta_b = [(-30, 30)] * (self.J - 1)
        sigma_b = [(0.001, 0.999)]
        if self.market_fe:
            xi_b = [(-30, 30)] * (self.T - 1)
            return delta_b + sigma_b + xi_b
        return delta_b + sigma_b

    def _compute_jacobian(self, theta):
        """∂s_j/∂δ_k at xi=0 under full availability, market 0. Shape (J, J)."""
        p = self._unpack_theta(theta)
        sigma = p["sigma"]
        nest_matrix = self.nest_matrix
        avail = self._availability_matrix_full

        def shares_of_delta(delta_full):
            G, J = nest_matrix.shape
            T = avail.shape[1]
            delta_reshape = delta_full.reshape(1, J)
            vgj_nest = jnp.where(nest_matrix, delta_reshape, UTILITY_PUNISH)
            vgjt_nest = vgj_nest.reshape(G, J, 1)
            V_gjt = jnp.where(avail.reshape(1, J, T), vgjt_nest, UTILITY_PUNISH)

            IV_gt = jax.nn.logsumexp(V_gjt / sigma, axis=1)
            S_gjt = jax.nn.softmax(V_gjt / sigma, axis=1)
            S_gt = jax.nn.softmax(IV_gt * sigma, axis=0)
            S_jt = (S_gjt * S_gt.reshape(G, 1, T)).sum(axis=0)
            return S_jt[:, 0]

        delta = jnp.concatenate([theta[:self.J - 1], jnp.array([0.0])])
        return jax.jacobian(shares_of_delta)(delta)

    # ── NL-specific public methods ───────────────────────────────

    def compute_model_shares(self, theta):
        """Full share decomposition. Returns (S_gjt, S_gt, S_jt).

        - S_gjt: (G, J, T) -- share of j conditional on nest g
        - S_gt:  (G, T)    -- nest shares
        - S_jt:  (J, T)    -- unconditional shares
        """
        theta = jnp.asarray(theta)
        if self.market_fe:
            p = self._unpack_theta(theta)
            # For decomposition, compute market-by-market
            # Return full availability version at xi=0 for simplicity
            core = self._core_theta(theta)
            return _shares_pipeline(core, self.nest_matrix, self.availability_matrix)
        else:
            core = self._core_theta(theta)
            return _shares_pipeline(core, self.nest_matrix, self.availability_matrix)

    def compute_diversion_matrix_from_formula(self, theta):
        """Closed-form diversion at xi=0. Shape (J, J). Alias for diversion_matrix()."""
        theta = jnp.asarray(theta)
        core = self._core_theta(theta)
        return _diversion_from_formula(
            core, self.nest_matrix, self._availability_matrix_full
        )

    def compute_diversion_matrix_vmap(self, theta):
        """Simulation-based diversion via product removal at xi=0. Shape (J, J).

        Uses market 0 (all markets equivalent under full availability at xi=0).
        Slower than the formula -- mainly for testing.
        """
        theta = jnp.asarray(theta)
        core = self._core_theta(theta)
        D_jkt = _diversion_vmap(
            core, self.nest_matrix, self._availability_matrix_full
        )
        return D_jkt[:, :, 0]
