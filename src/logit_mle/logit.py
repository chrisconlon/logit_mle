"""
Logit discrete choice model.

Parameters:
  - delta[J-1]: mean utility for inside goods (outside good normalized to 0)
  - xi[T-1]:    outside good's mean utility per market (only if market_fe=True;
                last market normalized to 0)

Diversion: closed-form IIA formula D_jk = s_k / (1 - s_j), evaluated at xi=0.
"""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from .base import DiscreteChoiceModel


# ── JAX computation functions ────────────────────────────────────

@jax.jit
def _s_jt(delta_inside, xi, availability_matrix):
    """Market shares. Shape (J, T).

    delta_inside : (J-1,) mean utility for inside goods
    xi           : (T,)   outside good's utility per market
    """
    J_in = delta_inside.shape[0]
    T = availability_matrix.shape[1]

    # (J, T) utility: inside goods constant across t, outside good = xi_t
    delta_jt = jnp.concatenate([
        jnp.broadcast_to(delta_inside[:, None], (J_in, T)),
        xi[None, :],
    ], axis=0)

    vjt = jnp.where(availability_matrix, delta_jt, -jnp.inf)
    return jax.nn.softmax(vjt, axis=0)


@jax.jit
def _diversion_jk(delta_inside, availability_matrix_full):
    """Closed-form logit diversion at xi=0. Shape (J, J)."""
    T = availability_matrix_full.shape[1]
    xi_zero = jnp.zeros(T)
    sjt = _s_jt(delta_inside, xi_zero, availability_matrix_full)
    s_j = sjt[:, 0]  # all markets identical at xi=0 under full availability
    J = s_j.shape[0]

    D_jk = s_j[None, :] / (1.0 - s_j[:, None])
    D_jk = D_jk.at[jnp.diag_indices(J)].set(0.0)
    return D_jk


# ── Logit class ──────────────────────────────────────────────────

class Logit(DiscreteChoiceModel):

    def __init__(self, availability_matrix, q_jt=None, *, market_fe=False,
                 diversion_data=None):
        super().__init__(availability_matrix, q_jt, diversion_data=diversion_data)
        self.market_fe = market_fe

    def _unpack_theta(self, theta):
        delta_inside = theta[:self.J - 1]
        delta = jnp.concatenate([delta_inside, jnp.array([0.0])])
        if self.market_fe:
            xi = jnp.concatenate([theta[self.J - 1:], jnp.array([0.0])])
        else:
            xi = jnp.zeros(self.T)
        return {"delta": delta, "delta_inside": delta_inside, "xi": xi}

    def _compute_shares(self, theta, avail):
        p = self._unpack_theta(theta)
        return _s_jt(p["delta_inside"], p["xi"], avail)

    def _compute_diversion(self, theta):
        p = self._unpack_theta(theta)
        return _diversion_jk(p["delta_inside"], self._availability_matrix_full)

    def _make_x0(self, rng):
        delta = rng.uniform(-10, -1, self.J - 1)
        if self.market_fe:
            xi = rng.uniform(-1, 1, self.T - 1)
            return jnp.array(np.concatenate([delta, xi]))
        return jnp.array(delta)

    def _theta_bounds(self):
        bounds = [(-30, 30)] * (self.J - 1)
        if self.market_fe:
            bounds += [(-30, 30)] * (self.T - 1)
        return bounds

    def _compute_jacobian(self, theta):
        """Closed-form Jacobian at xi=0: ∂s_j/∂δ_k = s_k(1_{j=k} - s_j)."""
        p = self._unpack_theta(theta)
        xi_zero = jnp.zeros(self.T)
        s = _s_jt(p["delta_inside"], xi_zero, self._availability_matrix_full)[:, 0]
        return jnp.diag(s) - jnp.outer(s, s)

    def _compute_elasticity(self, theta, prices, price_coeff, price_col):
        """Logit price elasticity at xi=0."""
        p = self._unpack_theta(theta)
        xi_zero = jnp.zeros(self.T)
        s = _s_jt(p["delta_inside"], xi_zero, self._availability_matrix_full)[:, 0]
        jac = jnp.diag(s) - jnp.outer(s, s)
        return price_coeff * prices[None, :] * jac / s[:, None]
