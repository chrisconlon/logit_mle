"""
Random coefficients discrete choice model.

Covers both RCN (market_fe=False) and RCC (market_fe=True).

Parameters:
  - delta[J-1]: mean utility (outside good normalized to 0)
  - sigma[G]:   std dev of taste heterogeneity per characteristic
  - xi[T-1]:    market fixed effects (only if market_fe=True; last market normalized to 0)

Diversion: D_jk = E_i[ s_ik/(1-s_ij) * s_ij/s_j ] integrated over sparse grid.
"""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from .base import DiscreteChoiceModel

UTILITY_PUNISH = -1e20


# ── JAX computation functions ────────────────────────────────────

@jax.jit
def _v_ijt(delta, sigma, xi, x_jg, nu_i, availability_matrix):
    """Individual-level utility. Shape (I, J, T)."""
    I = nu_i.shape[0]
    J = x_jg.shape[0]
    G = x_jg.shape[1]
    T = availability_matrix.shape[1]

    # Random coefficient interaction: sum_g x_jg * nu_ig * sigma_g -> (I, J)
    x_nu_sigma = jnp.sum(
        sigma.reshape(1, 1, G) * nu_i.reshape(I, 1, G) * x_jg.reshape(1, J, G),
        axis=-1,
    )

    # V_ij = delta_j + random part
    V_ij = delta.reshape(1, J) + x_nu_sigma  # (I, J)

    # Add market FEs: V_ijt = V_ij + xi_t
    V_ijt = V_ij[:, :, None] + xi[None, None, :]  # (I, J, T)

    # Mask unavailable products
    V_ijt = jnp.where(availability_matrix[None, :, :], V_ijt, UTILITY_PUNISH)

    return V_ijt


@jax.jit
def _s_ijt(delta, sigma, xi, x_jg, nu_i, availability_matrix):
    """Individual choice probabilities. Shape (I, J, T)."""
    V_ijt = _v_ijt(delta, sigma, xi, x_jg, nu_i, availability_matrix)
    return jax.nn.softmax(V_ijt, axis=1)


@jax.jit
def _s_jt(delta, sigma, xi, x_jg, nu_i, w_i, availability_matrix):
    """Market shares (integrated over individuals). Shape (J, T)."""
    S_ijt = _s_ijt(delta, sigma, xi, x_jg, nu_i, availability_matrix)
    I = w_i.shape[0]
    return jnp.sum(w_i.reshape(I, 1, 1) * S_ijt, axis=0)


@jax.jit
def _diversion_jk(delta, sigma, xi, x_jg, nu_i, w_i, availability_matrix_full):
    """Diversion D_jk = E_i[ s_ik/(1-s_ij) * s_ij/s_j ]. Shape (J, J)."""
    S_ijt = _s_ijt(delta, sigma, xi, x_jg, nu_i, availability_matrix_full)
    I = w_i.shape[0]
    S_jt = jnp.sum(w_i.reshape(I, 1, 1) * S_ijt, axis=0)

    # Use market 0 (all markets identical under full availability)
    S_ij = S_ijt[:, :, 0]  # (I, J)
    S_j = S_jt[:, 0]       # (J,)

    J = S_j.shape[0]

    s_ij_re = S_ij.reshape(I, J, 1)
    s_ik_re = S_ij.reshape(I, 1, J)
    s_j_re = S_j.reshape(1, J, 1)

    # Individual-level diversion
    D_ijk = (s_ik_re / (1.0 - s_ij_re)) * (s_ij_re / s_j_re)

    # Integrate over individuals
    D_jk = jnp.sum(w_i.reshape(I, 1, 1) * D_ijk, axis=0)  # (J, J)

    return D_jk


# ── RandomCoefficients class ────────────────────────────────────

class RandomCoefficients(DiscreteChoiceModel):
    """Random coefficients logit (RCN when market_fe=False, RCC when market_fe=True)."""

    def __init__(
        self,
        availability_matrix,
        q_jt=None,
        *,
        x2,
        nu_i,
        w_i,
        market_fe=False,
        diversion_data=None,
    ):
        super().__init__(availability_matrix, q_jt, diversion_data=diversion_data)
        self.x2 = jnp.array(x2)
        self.nu_i = jnp.array(nu_i)
        self.w_i = jnp.array(w_i)
        self.market_fe = market_fe
        self.G = self.x2.shape[1]
        self.I = self.nu_i.shape[0]
        self._print_info()

    def _print_info(self):
        label = "RCC" if self.market_fe else "RCN"
        print(label)
        w = 18
        print(f"{'I (Individuals):':<{w}} {self.I}")
        print(f"{'J (Products):':<{w}} {self.J}")
        print(f"{'T (Markets):':<{w}} {self.T}")
        print(f"{'G (Random Coeffs):':<{w}} {self.G}")

    def _unpack_theta(self, theta):
        delta = jnp.concatenate([theta[: self.J - 1], jnp.array([0.0])])
        sigma = theta[self.J - 1 : self.J - 1 + self.G]
        if self.market_fe:
            xi = jnp.concatenate([theta[self.J - 1 + self.G :], jnp.array([0.0])])
        else:
            xi = jnp.zeros(self.T)
        return {"delta": delta, "sigma": sigma, "xi": xi}

    def _compute_shares(self, theta, avail):
        p = self._unpack_theta(theta)
        return _s_jt(p["delta"], p["sigma"], p["xi"],
                      self.x2, self.nu_i, self.w_i, avail)

    def _compute_diversion(self, theta):
        p = self._unpack_theta(theta)
        return _diversion_jk(p["delta"], p["sigma"], p["xi"],
                              self.x2, self.nu_i, self.w_i,
                              self._availability_matrix_full)

    def _make_x0(self, rng):
        delta = rng.uniform(-5, -3, self.J - 1)
        sigma = rng.uniform(0, 1, self.G)
        if self.market_fe:
            xi = rng.uniform(-10, 10, self.T - 1)
            return jnp.array(np.concatenate([delta, sigma, xi]))
        else:
            return jnp.array(np.concatenate([delta, sigma]))

    def _theta_bounds(self):
        delta_b = [(-100, -1)] * (self.J - 1)
        sigma_b = [(0, 20)] * self.G
        if self.market_fe:
            xi_b = [(-100, 30)] * (self.T - 1)
            return delta_b + sigma_b + xi_b
        else:
            return delta_b + sigma_b

    def _compute_jacobian(self, theta):
        """∂s_j/∂δ_k = E_i[w_i · s_ik · (1_{j=k} - s_ij)]. Shape (J, J)."""
        p = self._unpack_theta(theta)
        S_ijt = _s_ijt(p["delta"], p["sigma"], p["xi"],
                        self.x2, self.nu_i, self._availability_matrix_full)
        S_ij = S_ijt[:, :, 0]  # (I, J)
        I, J = S_ij.shape
        # E_i[w_i * (diag(s_i) - s_i s_i^T)]
        # = sum_i w_i * diag(s_i) - sum_i w_i * outer(s_i, s_i)
        w = self.w_i  # (I,)
        weighted_diag = jnp.sum(w[:, None] * S_ij, axis=0)       # (J,)
        weighted_outer = jnp.einsum("i,ij,ik->jk", w, S_ij, S_ij)  # (J, J)
        return jnp.diag(weighted_diag) - weighted_outer

    def _compute_elasticity(self, theta, prices, price_coeff, price_col):
        """Price elasticity with heterogeneous price coefficients.

        β_i^p = price_coeff + σ_{price_col} · ν_i[price_col]

        η_jk = (p_k / s_j) · E_i[w_i · β_i^p · s_ik · (1_{j=k} - s_ij)]
        """
        p = self._unpack_theta(theta)
        S_ijt = _s_ijt(p["delta"], p["sigma"], p["xi"],
                        self.x2, self.nu_i, self._availability_matrix_full)
        S_ij = S_ijt[:, :, 0]  # (I, J)
        s_j = jnp.sum(self.w_i[:, None] * S_ij, axis=0)  # (J,)

        # Individual-level price coefficient
        if price_col is not None:
            beta_i = price_coeff + p["sigma"][price_col] * self.nu_i[:, price_col]  # (I,)
        else:
            beta_i = jnp.full(self.I, price_coeff)  # (I,)

        # E_i[w_i · β_i · (diag(s_i) - s_i s_i^T)]
        wb = self.w_i * beta_i  # (I,)
        weighted_diag = jnp.sum(wb[:, None] * S_ij, axis=0)          # (J,)
        weighted_outer = jnp.einsum("i,ij,ik->jk", wb, S_ij, S_ij)  # (J, J)
        jac_price = jnp.diag(weighted_diag) - weighted_outer         # (J, J)

        # η_jk = p_k · jac_price_jk / s_j
        return prices[None, :] * jac_price / s_j[:, None]
