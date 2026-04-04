"""
Logit discrete choice model.

Parameters: delta[J-1] (mean utility for inside goods; outside good normalized to 0).
Diversion: closed-form D_jk = s_k / (1 - s_j) under full availability.
"""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from .base import DiscreteChoiceModel


# ── JAX computation functions ────────────────────────────────────

@jax.jit
def _v_jt(theta, availability_matrix):
    """Deterministic utility. Shape (J, T)."""
    delta = jnp.concatenate([theta, jnp.array([0.0])])
    return jnp.where(availability_matrix, delta[:, None], -jnp.inf)


@jax.jit
def _s_jt(theta, availability_matrix):
    """Market shares. Shape (J, T)."""
    vjt = _v_jt(theta, availability_matrix)
    return jax.nn.softmax(vjt, axis=0)


@jax.jit
def _diversion_jk(theta, availability_matrix_full):
    """Closed-form logit diversion under full availability, market 0. Shape (J, J)."""
    sjt = _s_jt(theta, availability_matrix_full)
    s_j = sjt[:, 0]  # (J,) — all markets identical under full availability
    J = s_j.shape[0]

    # D_jk = s_k / (1 - s_j)
    D_jk = s_j[None, :] / (1.0 - s_j[:, None])  # (J, J)
    D_jk = D_jk.at[jnp.diag_indices(J)].set(0.0)
    return D_jk


# ── Logit class ──────────────────────────────────────────────────

class Logit(DiscreteChoiceModel):

    def __init__(self, availability_matrix, q_jt=None, *, diversion_data=None):
        super().__init__(availability_matrix, q_jt, diversion_data=diversion_data)

    def _compute_shares(self, theta, avail):
        return _s_jt(theta, avail)

    def _compute_diversion(self, theta):
        return _diversion_jk(theta, self._availability_matrix_full)

    def _unpack_theta(self, theta):
        delta = jnp.concatenate([theta, jnp.array([0.0])])
        return {"delta": delta}

    def _make_x0(self, rng):
        return jnp.array(rng.uniform(-10, -1, self.J - 1))

    def _theta_bounds(self):
        return [(-30, 30)] * (self.J - 1)

    def _compute_jacobian(self, theta):
        """Closed-form Jacobian: ∂s_j/∂δ_k = s_k(1_{j=k} - s_j). Shape (J, J)."""
        s = _s_jt(theta, self._availability_matrix_full)[:, 0]  # (J,)
        return jnp.diag(s) - jnp.outer(s, s)

    def _compute_elasticity(self, theta, prices, price_coeff, price_col):
        """Logit price elasticity: η_jk = α · p_k · s_k · (1_{j=k} - s_j) / s_j."""
        s = _s_jt(theta, self._availability_matrix_full)[:, 0]
        jac = jnp.diag(s) - jnp.outer(s, s)
        return price_coeff * prices[None, :] * jac / s[:, None]
