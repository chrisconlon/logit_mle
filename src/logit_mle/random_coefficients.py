"""
Random coefficients discrete choice model.

Covers both RCN (market_fe=False) and RCC (market_fe=True).

Parameters:
  - delta[J-1]: mean utility (outside good normalized to 0)
  - sigma[G]:   std dev of taste heterogeneity per characteristic
  - xi[T-1]:    outside good's mean utility per market (only if market_fe=True;
                last market normalized to 0)

Diversion: D_jk = E_i[ s_ik/(1-s_ij) * s_ij/s_j ] integrated over sparse grid,
evaluated at the xi=0 baseline.
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
def _v_ijt(delta_inside, sigma, xi, x_jg, nu_i, availability_matrix):
    """Individual-level utility. Shape (I, J, T).

    delta_inside : (J-1,)  mean utility for inside goods
    sigma        : (G,)    std dev of random coefficients
    xi           : (T,)    outside good's utility per market
    x_jg         : (J, G)  product characteristics (including outside good)
    nu_i         : (I, G)  quadrature nodes
    """
    I = nu_i.shape[0]
    J = x_jg.shape[0]
    G = x_jg.shape[1]
    T = availability_matrix.shape[1]
    J_in = J - 1

    # Random coefficient interaction: sum_g x_jg * nu_ig * sigma_g -> (I, J)
    x_nu_sigma = jnp.sum(
        sigma.reshape(1, 1, G) * nu_i.reshape(I, 1, G) * x_jg.reshape(1, J, G),
        axis=-1,
    )

    # Mean utility: delta_j for inside goods, xi_t for outside good
    # Build (J, T) delta matrix
    delta_jt = jnp.concatenate([
        jnp.broadcast_to(delta_inside[:, None], (J_in, T)),
        xi[None, :],
    ], axis=0)  # (J, T)

    # V_ijt = delta_jt + RC interaction
    V_ijt = delta_jt[None, :, :] + x_nu_sigma[:, :, None]  # (I, J, T)

    # Mask unavailable products
    V_ijt = jnp.where(availability_matrix[None, :, :], V_ijt, UTILITY_PUNISH)

    return V_ijt


@jax.jit
def _s_ijt(delta_inside, sigma, xi, x_jg, nu_i, availability_matrix):
    """Individual choice probabilities. Shape (I, J, T)."""
    V_ijt = _v_ijt(delta_inside, sigma, xi, x_jg, nu_i, availability_matrix)
    return jax.nn.softmax(V_ijt, axis=1)


@jax.jit
def _s_jt(delta_inside, sigma, xi, x_jg, nu_i, w_i, availability_matrix):
    """Market shares (integrated over individuals). Shape (J, T)."""
    S_ijt = _s_ijt(delta_inside, sigma, xi, x_jg, nu_i, availability_matrix)
    I = w_i.shape[0]
    return jnp.sum(w_i.reshape(I, 1, 1) * S_ijt, axis=0)


@jax.jit
def _diversion_jk(delta_inside, sigma, x_jg, nu_i, w_i, availability_matrix_full):
    """Diversion D_jk at xi=0. Shape (J, J)."""
    T = availability_matrix_full.shape[1]
    xi_zero = jnp.zeros(T)
    S_ijt = _s_ijt(delta_inside, sigma, xi_zero, x_jg, nu_i, availability_matrix_full)
    I = w_i.shape[0]
    S_jt = jnp.sum(w_i.reshape(I, 1, 1) * S_ijt, axis=0)

    # Use market 0 (all markets identical at xi=0 under full availability)
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
        delta_inside = theta[:self.J - 1]
        delta = jnp.concatenate([delta_inside, jnp.array([0.0])])
        sigma = theta[self.J - 1 : self.J - 1 + self.G]
        if self.market_fe:
            xi = jnp.concatenate([theta[self.J - 1 + self.G :], jnp.array([0.0])])
        else:
            xi = jnp.zeros(self.T)
        return {"delta": delta, "delta_inside": delta_inside, "sigma": sigma, "xi": xi}

    def _compute_shares(self, theta, avail):
        p = self._unpack_theta(theta)
        return _s_jt(p["delta_inside"], p["sigma"], p["xi"],
                      self.x2, self.nu_i, self.w_i, avail)

    def _compute_diversion(self, theta):
        p = self._unpack_theta(theta)
        return _diversion_jk(p["delta_inside"], p["sigma"],
                              self.x2, self.nu_i, self.w_i,
                              self._availability_matrix_full)

    def _make_x0(self, rng):
        delta = rng.uniform(-5, -3, self.J - 1)
        sigma = rng.uniform(0, 1, self.G)
        if self.market_fe:
            xi = rng.uniform(-1, 1, self.T - 1)
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

    def _theta_with_zero_xi(self, theta):
        """Replace ξ entries in theta with zero (ξ=0 baseline evaluation)."""
        if not self.market_fe:
            return theta
        # theta layout: (delta[J-1], sigma[G], xi[T-1])
        return jnp.concatenate(
            [theta[:self.J - 1 + self.G], jnp.zeros(self.T - 1)]
        )

    def _compute_jacobian(self, theta):
        """∂s_j/∂δ_k at xi=0. Shape (J, J)."""
        p = self._unpack_theta(theta)
        xi_zero = jnp.zeros(self.T)
        S_ijt = _s_ijt(p["delta_inside"], p["sigma"], xi_zero,
                        self.x2, self.nu_i, self._availability_matrix_full)
        S_ij = S_ijt[:, :, 0]  # (I, J)
        w = self.w_i
        weighted_diag = jnp.sum(w[:, None] * S_ij, axis=0)
        weighted_outer = jnp.einsum("i,ij,ik->jk", w, S_ij, S_ij)
        return jnp.diag(weighted_diag) - weighted_outer

    def _compute_elasticity(self, theta, prices, price_coeff, price_col):
        """Price elasticity at xi=0 with heterogeneous price coefficients.

        beta_i^p = price_coeff + sigma[price_col] * nu_i[price_col]
        """
        p = self._unpack_theta(theta)
        xi_zero = jnp.zeros(self.T)
        S_ijt = _s_ijt(p["delta_inside"], p["sigma"], xi_zero,
                        self.x2, self.nu_i, self._availability_matrix_full)
        S_ij = S_ijt[:, :, 0]
        s_j = jnp.sum(self.w_i[:, None] * S_ij, axis=0)

        if price_col is not None:
            beta_i = price_coeff + p["sigma"][price_col] * self.nu_i[:, price_col]
        else:
            beta_i = jnp.full(self.I, price_coeff)

        wb = self.w_i * beta_i
        weighted_diag = jnp.sum(wb[:, None] * S_ij, axis=0)
        weighted_outer = jnp.einsum("i,ij,ik->jk", wb, S_ij, S_ij)
        jac_price = jnp.diag(weighted_diag) - weighted_outer

        return prices[None, :] * jac_price / s_j[:, None]
