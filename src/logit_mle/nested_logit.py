"""
Nested Logit discrete choice model.

Public API parameters:
  - delta[J-1]: mean utility for inside goods (outside good normalized to 0)
  - rho:        Berry/Cardell nesting parameter in [0, 1).
                rho = 0 collapses to plain logit; rho -> 1 is full within-nest
                correlation.
  - xi[T-1]:    outside good's mean utility per market (only if market_fe=True;
                last market normalized to 0)

Constructor takes nesting_ids (J-1,) integer vector for inside goods.
The outside good is automatically placed in its own singleton nest.

Share formulas (Berry/Cardell):
  s_{j|g} = exp(delta_j / (1 - rho)) / D_g
  s_g     = D_g^(1 - rho) / sum_g' D_g'^(1 - rho)
  D_g     = sum_{k in J_g} exp(delta_k / (1 - rho))

Internally, the optimizer reparameterizes via gamma = delta / (1 - rho), which
removes 1/(1 - rho) from the inner exponentials and decouples the rho/delta
cross-partial. This makes the (negative) log-likelihood much smoother and the
optimization markedly more robust to starting values. The substitution is
transparent to the user: theta is passed and returned in (delta, rho, xi)
form everywhere.

Diversion: closed-form formula from the paper appendix (tau_in/tau_out),
evaluated at the xi=0 baseline. A vmap product-removal method is
also available for testing.
"""
from __future__ import annotations

import numpy as np
import scipy as sp
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from .base import DiscreteChoiceModel

UTILITY_PUNISH = -1e20


# ── JAX share / diversion functions in (gamma, rho) form ─────────
#
# These take gamma = delta / (1 - rho) and rho directly. The outside-good
# entries of gamma are passed in via gamma_xi (a length-T vector when there
# are market fixed effects, or a length-T vector of zeros otherwise).


def _shares_from_gamma(gamma_inside, rho, gamma_xi, nest_matrix, availability_matrix):
    """Shares (J, T) under (gamma, rho) parameterization.

    gamma_inside : (J-1,) inside-good gamma values
    rho          : scalar nesting parameter (Berry/Cardell)
    gamma_xi     : (T,) outside-good gamma values per market
                   (set last entry to 0 to apply normalization)
    """
    J_in = gamma_inside.shape[0]
    J = J_in + 1
    T = availability_matrix.shape[1]
    G = nest_matrix.shape[0]

    gamma_inside_full = jnp.broadcast_to(gamma_inside[:, None], (J_in, T))
    gamma_full = jnp.concatenate([gamma_inside_full, gamma_xi[None, :]], axis=0)

    nm = nest_matrix[:, :, None]                          # (G, J, 1)
    av = availability_matrix.astype(bool)[None, :, :]     # (1, J, T)
    g_jt = gamma_full[None, :, :]                         # (1, J, T)
    V_gjt = jnp.where(nm & av, g_jt, UTILITY_PUNISH)

    IV_gt = jax.nn.logsumexp(V_gjt, axis=1)            # log D_g, (G, T)
    s_gjt = jax.nn.softmax(V_gjt, axis=1)              # conditional within nest
    s_gt = jax.nn.softmax(IV_gt * (1.0 - rho), axis=0) # nest share
    s_jt = (s_gjt * s_gt[:, None, :]).sum(axis=0)
    return s_jt


def _shares_decomposition_from_gamma(gamma_inside, rho, gamma_xi,
                                     nest_matrix, availability_matrix):
    """Like _shares_from_gamma but also returns (S_gjt, S_gt, S_jt)."""
    J_in = gamma_inside.shape[0]
    T = availability_matrix.shape[1]
    G = nest_matrix.shape[0]

    gamma_inside_full = jnp.broadcast_to(gamma_inside[:, None], (J_in, T))
    gamma_full = jnp.concatenate([gamma_inside_full, gamma_xi[None, :]], axis=0)

    nm = nest_matrix[:, :, None]
    av = availability_matrix.astype(bool)[None, :, :]
    g_jt = gamma_full[None, :, :]
    V_gjt = jnp.where(nm & av, g_jt, UTILITY_PUNISH)

    IV_gt = jax.nn.logsumexp(V_gjt, axis=1)
    s_gjt = jax.nn.softmax(V_gjt, axis=1)
    s_gt = jax.nn.softmax(IV_gt * (1.0 - rho), axis=0)
    s_jt = (s_gjt * s_gt[:, None, :]).sum(axis=0)
    return s_gjt, s_gt, s_jt


def _diversion_from_gamma(gamma_inside, rho, nest_matrix, availability_matrix_full):
    """Closed-form diversion at xi=0 under full availability. Shape (J, J)."""
    J_in = gamma_inside.shape[0]
    J = J_in + 1
    T = availability_matrix_full.shape[1]

    gamma_xi_zero = jnp.zeros(T)
    s_gjt, s_gt, s_jt = _shares_decomposition_from_gamma(
        gamma_inside, rho, gamma_xi_zero, nest_matrix, availability_matrix_full
    )

    s_gj = s_gjt[:, :, 0]   # (G, J)
    s_g = s_gt[:, 0]        # (G,)
    s_j = s_jt[:, 0]        # (J,)

    inside = jnp.arange(J_in)
    g_of_all = jnp.argmax(nest_matrix.astype(jnp.int32), axis=0)
    g_of_in = g_of_all[:J_in]

    sgj = s_gj[g_of_in, inside]
    sg = s_g[g_of_in]

    # K_j = (1 - s_g) + s_g * (1 - s_{j|g})^{1-rho}
    DELTA = (1.0 - sg) + sg * (1.0 - sgj) ** (1.0 - rho)
    aj = 1.0 / DELTA - 1.0                                # tau_out - 1
    cj = 1.0 / (DELTA * (1.0 - sgj) ** rho) - 1.0         # tau_in - 1
    bj = cj - aj

    M = (g_of_in[:, None] == g_of_all[None, :]).astype(s_j.dtype)  # (J_in, J)
    ratio = s_j[None, :] / s_j[:J_in][:, None]                     # (J_in, J)

    D_in = ratio * (aj[:, None] + bj[:, None] * M)                 # (J_in, J)
    D_0 = (s_j / (1.0 - s_j[-1]))[None, :]                         # (1, J)
    D = jnp.vstack([D_in, D_0])
    return D


def _diversion_vmap_from_gamma(gamma_inside, rho, nest_matrix, availability_matrix_full):
    """Simulation-based diversion via vmap product removal at xi=0.

    Slower than the closed form -- mainly for testing.
    """
    J, T = availability_matrix_full.shape
    gamma_xi_zero = jnp.zeros(T)

    s_jt = _shares_from_gamma(
        gamma_inside, rho, gamma_xi_zero, nest_matrix, availability_matrix_full
    )

    def shares_with_j_removed(j):
        new_avail = availability_matrix_full.at[j, :].set(False)
        return _shares_from_gamma(
            gamma_inside, rho, gamma_xi_zero, nest_matrix, new_avail
        )

    S_rjt = jax.vmap(shares_with_j_removed)(jnp.arange(J))

    numer = S_rjt - s_jt[None, :, :]
    denom = s_jt[:, None, :]
    D_jkt = numer / denom

    row_idx, col_idx = jnp.diag_indices(J)
    D_jkt = D_jkt.at[row_idx, col_idx, :].set(0.0)
    D_jkt = jnp.nan_to_num(D_jkt, nan=0.0)
    return D_jkt


# ── NestedLogit class ────────────────────────────────────────────

class NestedLogit(DiscreteChoiceModel):
    """Nested logit model.

    Public theta layout: ``(delta[J-1], rho, xi[T-1])`` if ``market_fe=True``,
    else ``(delta[J-1], rho)``.

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

    # ── Theta packing in user (delta) form ───────────────────────

    def _unpack_theta(self, theta):
        """Split user-form theta = (delta[J-1], rho, xi[T-1])."""
        delta_inside = theta[:self.J - 1]
        delta = jnp.concatenate([delta_inside, jnp.array([0.0])])
        rho = theta[self.J - 1]
        if self.market_fe:
            xi_inside = theta[self.J:]
            xi = jnp.concatenate([xi_inside, jnp.array([0.0])])
        else:
            xi = jnp.zeros(self.T)
        return {
            "delta": delta,
            "delta_inside": delta_inside,
            "rho": rho,
            "xi": xi,
        }

    def _delta_theta_to_gamma(self, theta):
        """Convert user-form (delta, rho, xi) -> internal (gamma, rho, gamma_xi).

        gamma = delta / (1 - rho); gamma_xi = xi / (1 - rho).
        """
        p = self._unpack_theta(theta)
        rho = p["rho"]
        gamma_inside = p["delta_inside"] / (1.0 - rho)
        if self.market_fe:
            xi_inside = theta[self.J:]
            gamma_xi_inside = xi_inside / (1.0 - rho)
            return jnp.concatenate(
                [gamma_inside, jnp.array([rho]), gamma_xi_inside]
            )
        return jnp.concatenate([gamma_inside, jnp.array([rho])])

    def _gamma_theta_to_delta(self, theta_gamma):
        """Convert internal (gamma, rho, gamma_xi) -> user-form (delta, rho, xi)."""
        gamma_inside = theta_gamma[:self.J - 1]
        rho = theta_gamma[self.J - 1]
        delta_inside = gamma_inside * (1.0 - rho)
        if self.market_fe:
            gamma_xi_inside = theta_gamma[self.J:]
            xi_inside = gamma_xi_inside * (1.0 - rho)
            return jnp.concatenate([delta_inside, jnp.array([rho]), xi_inside])
        return jnp.concatenate([delta_inside, jnp.array([rho])])

    # ── Share / diversion (user-form theta) ──────────────────────

    def _gamma_components(self, theta):
        """Return (gamma_inside, rho, gamma_xi) given user-form theta."""
        p = self._unpack_theta(theta)
        rho = p["rho"]
        gamma_inside = p["delta_inside"] / (1.0 - rho)
        gamma_xi = p["xi"] / (1.0 - rho)  # length T, last entry is 0
        return gamma_inside, rho, gamma_xi

    def _compute_shares(self, theta, avail):
        gamma_inside, rho, gamma_xi = self._gamma_components(theta)
        return _shares_from_gamma(
            gamma_inside, rho, gamma_xi, self.nest_matrix, avail
        )

    def _compute_diversion(self, theta):
        """Closed-form diversion at xi=0 under full availability. Shape (J, J)."""
        gamma_inside, rho, _ = self._gamma_components(theta)
        return _diversion_from_gamma(
            gamma_inside, rho, self.nest_matrix, self._availability_matrix_full
        )

    # ── Starting values & bounds (user-form / delta-form) ────────

    def _make_x0(self, rng):
        """Starting values in user (delta) form."""
        delta = rng.uniform(-3, 3, self.J - 1)
        rho = rng.uniform(0.05, 0.5)
        if self.market_fe:
            xi = rng.uniform(-0.5, 0.5, self.T - 1)
            return jnp.array(np.concatenate([delta, [rho], xi]))
        return jnp.array(np.concatenate([delta, [rho]]))

    def _theta_bounds(self):
        """Bounds in user (delta) form. Used by base.fit() if it is ever called."""
        delta_b = [(-30, 30)] * (self.J - 1)
        rho_b = [(0.0, 0.999)]
        if self.market_fe:
            xi_b = [(-30, 30)] * (self.T - 1)
            return delta_b + rho_b + xi_b
        return delta_b + rho_b

    # ── Internal gamma-space starting values & bounds ────────────

    def _make_x0_gamma(self, rng):
        """Starting values in internal (gamma) form."""
        gamma = rng.uniform(-3, 3, self.J - 1)
        rho = rng.uniform(0.05, 0.5)
        if self.market_fe:
            gamma_xi = rng.uniform(-0.5, 0.5, self.T - 1)
            return np.concatenate([gamma, [rho], gamma_xi])
        return np.concatenate([gamma, [rho]])

    def _gamma_bounds(self):
        gamma_b = [(-50, 50)] * (self.J - 1)
        rho_b = [(0.0, 0.999)]
        if self.market_fe:
            gamma_xi_b = [(-50, 50)] * (self.T - 1)
            return gamma_b + rho_b + gamma_xi_b
        return gamma_b + rho_b

    # ── Custom fit() that optimizes in gamma-space ───────────────

    def fit(
        self,
        *,
        aug_div=False,
        penalty=1e-4,
        diversion_rows=None,
        ftol=1e-14,
        seed=2025,
        verbose=True,
    ):
        """Maximum likelihood with internal gamma reparameterization.

        Optimizes over gamma = delta / (1 - rho) for numerical stability,
        then converts the solution back to (delta, rho, xi) before returning.
        The returned ``result.x`` is in user (delta) form.
        """
        if self.q_jt is None:
            raise ValueError("q_jt is required for fit()")
        if aug_div and self._diversion_data is None:
            raise ValueError("aug_div=True requires diversion_data")

        rng = np.random.RandomState(seed)
        x0_gamma = self._make_x0_gamma(rng)
        bounds_gamma = self._gamma_bounds()

        # Capture closure data for clean JIT
        avail = self.availability_matrix
        avail_full = self._availability_matrix_full
        q_jt = self.q_jt
        nest_matrix = self.nest_matrix
        J_in = self.J - 1
        T = self.T
        market_fe = self.market_fe
        div_data = self._diversion_data

        if aug_div:
            if diversion_rows is None:
                rows = jnp.arange(self.J)
            else:
                rows = jnp.array(diversion_rows)
        else:
            rows = None

        def objective_gamma(theta_gamma):
            gamma_inside = theta_gamma[:J_in]
            rho = theta_gamma[J_in]
            if market_fe:
                gamma_xi_inside = theta_gamma[J_in + 1:]
                gamma_xi = jnp.concatenate(
                    [gamma_xi_inside, jnp.array([0.0])]
                )
            else:
                gamma_xi = jnp.zeros(T)

            s_jt = _shares_from_gamma(
                gamma_inside, rho, gamma_xi, nest_matrix, avail
            )
            s_jt_safe = jnp.where(s_jt == 0, 1.0, s_jt)
            model_ll = jnp.sum(q_jt * jnp.log(s_jt_safe))

            if not aug_div:
                return -model_ll

            D_jk = _diversion_from_gamma(
                gamma_inside, rho, nest_matrix, avail_full
            )
            D_jk = D_jk.at[jnp.diag_indices_from(D_jk)].set(0.0)
            D_jk_safe = jnp.where(D_jk == 0, 1.0, D_jk)
            div_ll = jnp.sum(
                div_data[rows, :] * jnp.log(D_jk_safe[rows, :])
            )
            return -(penalty * model_ll + div_ll)

        jit_obj = jax.jit(objective_gamma)
        jit_grad = jax.jit(jax.grad(objective_gamma))

        if verbose:
            def callback(x):
                ll = jit_obj(x)
                gn = jnp.linalg.norm(jit_grad(x))
                print(f"Likelihood: {ll:.6f}  ||grad||: {gn:.6e}")
        else:
            callback = None

        if verbose:
            print(f"Starting MLE (NestedLogit, dim={len(x0_gamma)}, "
                  f"aug_div={aug_div}, gamma-space)")
            ll0 = jit_obj(x0_gamma)
            gn0 = jnp.linalg.norm(jit_grad(x0_gamma))
            print(f"Likelihood at x0: {ll0:.6f}  ||grad||: {gn0:.6e}")

        result = sp.optimize.minimize(
            jit_obj,
            x0_gamma,
            method="L-BFGS-B",
            jac=jit_grad,
            bounds=bounds_gamma,
            callback=callback,
            options={"disp": verbose, "ftol": ftol, "maxfun": 1_000_000},
        )

        # Convert result.x from gamma form back to delta form
        result.x = np.asarray(self._gamma_theta_to_delta(jnp.array(result.x)))
        return result

    def _theta_with_zero_xi(self, theta):
        """Replace ξ entries in theta with zero (ξ=0 baseline evaluation)."""
        if not self.market_fe:
            return theta
        # theta layout: (delta[J-1], rho, xi[T-1])
        return jnp.concatenate([theta[:self.J], jnp.zeros(self.T - 1)])

    # NestedLogit inherits the base class _compute_jacobian (autodiff through
    # _compute_shares with the constant-shift invariance for the outside-good
    # column, and ξ zeroed via the _theta_with_zero_xi hook above). The internal
    # gamma reparameterization is handled inside _compute_shares; the chain rule
    # produces the correct ∂s_j/∂δ_k.

    # ── NL-specific public methods ───────────────────────────────

    def compute_model_shares(self, theta):
        """Full share decomposition. Returns (S_gjt, S_gt, S_jt).

        - S_gjt: (G, J, T) -- share of j conditional on nest g
        - S_gt:  (G, T)    -- nest shares
        - S_jt:  (J, T)    -- unconditional shares
        """
        theta = jnp.asarray(theta)
        gamma_inside, rho, gamma_xi = self._gamma_components(theta)
        return _shares_decomposition_from_gamma(
            gamma_inside, rho, gamma_xi, self.nest_matrix, self.availability_matrix
        )

    def compute_diversion_matrix_from_formula(self, theta):
        """Closed-form diversion at xi=0. Alias for diversion_matrix()."""
        return self.diversion_matrix(theta)

    def compute_diversion_matrix_vmap(self, theta):
        """Simulation-based diversion via product removal at xi=0. Shape (J, J).

        Uses market 0 (all markets equivalent under full availability at xi=0).
        Slower than the formula -- mainly for testing.
        """
        theta = jnp.asarray(theta)
        gamma_inside, rho, _ = self._gamma_components(theta)
        D_jkt = _diversion_vmap_from_gamma(
            gamma_inside, rho, self.nest_matrix, self._availability_matrix_full
        )
        return D_jkt[:, :, 0]

    # ── Public conversion helpers ────────────────────────────────

    def gamma_from_theta(self, theta):
        """Return the internal gamma-form theta corresponding to user theta.

        gamma = delta / (1 - rho); gamma_xi = xi / (1 - rho).
        Useful for diagnostics or for warm-starting internal optimizers.
        """
        return self._delta_theta_to_gamma(jnp.asarray(theta))

    def theta_from_gamma(self, theta_gamma):
        """Return the user-form theta corresponding to a gamma-form theta."""
        return self._gamma_theta_to_delta(jnp.asarray(theta_gamma))
