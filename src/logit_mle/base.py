"""
Base class for discrete choice models.

All models share the same optimization loop (L-BFGS-B), likelihood structure,
and augmented-likelihood pattern. Subclasses override utility/share computation,
parameter packing, and diversion formulas.
"""
from __future__ import annotations

import numpy as np
import scipy as sp
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


class DiscreteChoiceModel:
    """Base class for Logit, NestedLogit, and RandomCoefficients models."""

    def __init__(
        self,
        availability_matrix,
        q_jt=None,
        *,
        diversion_data=None,
    ):
        self.availability_matrix = jnp.array(availability_matrix)
        self.J, self.T = self.availability_matrix.shape
        self._availability_matrix_full = jnp.ones((self.J, self.T))
        self.q_jt = jnp.array(q_jt) if q_jt is not None else None
        self._diversion_data = (
            jnp.array(diversion_data) if diversion_data is not None else None
        )

    # ── Public API ───────────────────────────────────────────────

    def shares(self, theta):
        """Market shares s_jt under the stored availability matrix. Shape (J, T)."""
        theta = jnp.asarray(theta)
        return self._compute_shares(theta, self.availability_matrix)

    def diversion_matrix(self, theta):
        """Diversion ratios D_jk under full availability. Shape (J, J)."""
        theta = jnp.asarray(theta)
        return self._compute_diversion(theta)

    def log_likelihood(self, theta, *, aug_div=False, penalty=1e-4,
                       diversion_rows=None):
        """Negative log-likelihood (scalar). Optionally augmented with diversion."""
        theta = jnp.asarray(theta)
        return self._objective(theta, aug_div, penalty, diversion_rows)

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
        """Estimate parameters by maximum likelihood via L-BFGS-B.

        Parameters
        ----------
        aug_div : bool
            If True, augment the likelihood with a diversion component.
        penalty : float
            Weight on the model likelihood in the augmented objective.
        diversion_rows : tuple[int, ...] or None
            Row indices of diversion_data to use. None = all rows.
        ftol : float
            Function tolerance for L-BFGS-B.
        seed : int
            Random seed for starting values.
        verbose : bool
            Print iteration callback.

        Returns
        -------
        scipy.optimize.OptimizeResult
        """
        if self.q_jt is None:
            raise ValueError("q_jt is required for fit()")
        if aug_div and self._diversion_data is None:
            raise ValueError("aug_div=True requires diversion_data")

        rng = np.random.RandomState(seed)
        x0 = self._make_x0(rng)
        bounds = self._theta_bounds()

        # Build objective as a closure over data (not self) for clean JIT
        avail = self.availability_matrix
        avail_full = self._availability_matrix_full
        q_jt = self.q_jt
        div_data = self._diversion_data

        # Resolve diversion_rows
        if aug_div:
            if diversion_rows is None:
                rows = jnp.arange(self.J)
            else:
                rows = jnp.array(diversion_rows)
        else:
            rows = None

        def objective(theta):
            return self._objective(theta, aug_div, penalty, diversion_rows)

        jit_obj = jax.jit(objective)
        jit_grad = jax.jit(jax.grad(objective))

        if verbose:
            def callback(x):
                ll = jit_obj(x)
                gn = jnp.linalg.norm(jit_grad(x))
                print(f"Likelihood: {ll:.6f}  ||grad||: {gn:.6e}")
        else:
            callback = None

        if verbose:
            print(f"Starting MLE ({self.__class__.__name__}, "
                  f"dim={len(x0)}, aug_div={aug_div})")
            ll0 = jit_obj(x0)
            gn0 = jnp.linalg.norm(jit_grad(x0))
            print(f"Likelihood at x0: {ll0:.6f}  ||grad||: {gn0:.6e}")

        result = sp.optimize.minimize(
            jit_obj,
            x0,
            method="L-BFGS-B",
            jac=jit_grad,
            bounds=bounds,
            callback=callback,
            options={"disp": verbose, "ftol": ftol, "maxfun": 1_000_000},
        )

        return result

    def jacobian(self, theta):
        """Jacobian of shares w.r.t. mean utilities: ∂s_j/∂δ_k. Shape (J, J).

        Computed under full availability at market t=0.
        Default implementation uses JAX autodiff; subclasses may override
        with closed-form expressions.
        """
        theta = jnp.asarray(theta)
        return self._compute_jacobian(theta)

    def elasticity_matrix(self, theta, *, prices, price_coeff, price_col=None):
        """Price elasticity matrix η_jk = (∂s_j/∂p_k) · (p_k / s_j). Shape (J, J).

        Parameters
        ----------
        theta : array
            Estimated parameter vector.
        prices : array, shape (J,)
            Price of each product (including outside good).
        price_coeff : float
            Mean price coefficient (α, typically negative). For all model
            types, this is the average marginal utility of income.
        price_col : int, optional
            For RandomCoefficients only: column index of price in x2.
            When provided, the individual-level price coefficient is
            β_i^p = price_coeff + σ_{price_col} · ν_i[price_col],
            and the elasticity integrates over heterogeneous price
            sensitivity. When None, uses homogeneous price_coeff for
            all individuals (equivalent to Logit/NL behavior).
        """
        theta = jnp.asarray(theta)
        prices = jnp.asarray(prices)
        return self._compute_elasticity(theta, prices, price_coeff, price_col)

    # ── Internal objective ───────────────────────────────────────

    def _objective(self, theta, aug_div, penalty, diversion_rows):
        """Negative (augmented) log-likelihood."""
        s_jt = self._compute_shares(theta, self.availability_matrix)
        s_jt_safe = jnp.where(s_jt == 0, 1.0, s_jt)
        model_ll = jnp.sum(self.q_jt * jnp.log(s_jt_safe))

        if not aug_div:
            return -model_ll

        D_jk = self._compute_diversion(theta)
        D_jk = D_jk.at[jnp.diag_indices_from(D_jk)].set(0.0)
        D_jk_safe = jnp.where(D_jk == 0, 1.0, D_jk)

        if diversion_rows is None:
            rows = jnp.arange(self.J)
        else:
            rows = jnp.array(diversion_rows)

        div_ll = jnp.sum(
            self._diversion_data[rows, :] * jnp.log(D_jk_safe[rows, :])
        )

        return -(penalty * model_ll + div_ll)

    # ── Jacobian / elasticity internals ────────────────────────────

    def _compute_jacobian(self, theta):
        """∂s_j/∂δ_k under full availability, market 0. Shape (J, J).

        Default uses JAX autodiff through the share function.
        Subclasses may override with closed-form expressions.
        """
        avail = self._availability_matrix_full

        def shares_col0(delta_full):
            # Rebuild theta with modified delta (first J-1 entries of theta)
            new_theta = jnp.concatenate([delta_full[:-1], theta[self.J - 1:]])
            return self._compute_shares(new_theta, avail)[:, 0]

        p = self._unpack_theta(theta)
        return jax.jacobian(shares_col0)(p["delta"])

    def _compute_elasticity(self, theta, prices, price_coeff, price_col):
        """Price elasticity matrix. Shape (J, J).

        Default implementation: η_jk = price_coeff · p_k · (∂s_j/∂δ_k) / s_j.
        Subclasses with heterogeneous price coefficients override this.
        """
        jac = self._compute_jacobian(theta)  # (J, J)
        s = self._compute_shares(theta, self._availability_matrix_full)[:, 0]  # (J,)

        # η_jk = price_coeff · p_k · jac_jk / s_j
        eta = price_coeff * prices[None, :] * jac / s[:, None]
        return eta

    # ── Subclass hooks (must override) ───────────────────────────

    def _compute_shares(self, theta, avail):
        """Compute s_jt given availability matrix. Shape (J, T)."""
        raise NotImplementedError

    def _compute_diversion(self, theta):
        """Compute D_jk under full availability. Shape (J, J)."""
        raise NotImplementedError

    def _make_x0(self, rng):
        """Generate starting values for theta."""
        raise NotImplementedError

    def _theta_bounds(self):
        """Return list of (lo, hi) bounds for L-BFGS-B."""
        raise NotImplementedError

    def _unpack_theta(self, theta):
        """Split flat theta vector into named parameter dict."""
        raise NotImplementedError
