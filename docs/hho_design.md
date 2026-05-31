# Design doc: HHO fixed-grid nonparametric random coefficients

**Status:** proposal
**Scope:** add a second-stage estimator that replaces the parametric (independent-normal)
random-coefficient distribution in `RandomCoefficients` with a *nonparametric* distribution
estimated on a fixed grid, following Fox–Kim–Ryan–Bajari (FKRB, 2011) and the
nonnegative elastic-net generalization of Heiss–Hetzenecker–Osterhaus (HHO, 2022).

## 1. Goal

`RandomCoefficients` currently assumes \(\beta_i \sim N(0, \mathrm{diag}(\sigma^2))\):
in `_v_ijt`, the individual deviation is \(\sum_g \sigma_g\,\nu_{ig}\,x_{jg}\) with
\(\nu_i\sim N(0,1)\). The mean taste sits in \(\delta_{jt}\); the shape of the deviation
distribution is forced to be independent normal.

This estimator relaxes that. It recovers a nonparametric mixing distribution
\(\hat F(\beta) = \{(\beta_r,\theta_r)\}_{r=1}^R\) over a fixed grid of deviation vectors,
holding the step-1 mean utilities \(\hat\delta_{jt}\) fixed as data. The deliverable is a
distribution object that plugs straight back into the existing share / diversion /
elasticity machinery for downstream counterfactuals.

It answers a concrete question a practitioner (or referee) will ask: *is the
independent-normal assumption, rather than the data, driving the estimated substitution
patterns?* Refit nonparametrically, recompute diversion, compare.

## 2. What HHO estimate (one paragraph)

Fix a grid of deviation vectors \(\{\beta_1,\dots,\beta_R\}\subset\mathbb R^G\). Each grid
point implies a logit choice probability per product-market; the predicted share is the
mixture \(\hat s_{jt}(\theta)=\sum_r \theta_r\,s_{jt}(\beta_r)\), **linear in the weights
\(\theta\)**. FKRB estimate \(\theta\) by constrained least squares on the simplex. HHO add
a ridge (quadratic) term, giving a *nonnegative elastic net*:

```text
min_theta  || s_obs - Z theta ||^2  +  mu * ||theta||^2
s.t.       theta >= 0,   sum(theta) = 1
```

where \(Z\) is the \((JT)\times R\) matrix of grid-point predicted shares. On the simplex the
\(\ell_1\) norm is constant (\(\sum\theta=1\)), so the simplex constraint *is* the LASSO
piece and the ridge term \(\mu\|\theta\|^2\) is HHO's addition. The ridge does double duty:
it (i) regularizes the near-singular \(Z^\top Z\) that arises when the grid is dense (columns
for neighboring \(\beta_r\) are nearly collinear), and (ii) groups correlated grid points
instead of FKRB's "pick one at random, zero the rest." FKRB is the special case \(\mu\to 0\).
The penalty form above is HHO Eq. (10); the constraint form \(\|\theta\|^2\le t\) is Eq. (9),
equivalent under a monotone \(t\leftrightarrow\mu\) map.

References: HHO (2022, *J. Econometrics* 229:299–321); FKRB (2011, *Quant. Econ.* 2:381–418);
two-step plug-in for fixed coefficients per HHO §5, following Houde & Myers (2019) and
Fox et al. (2016).

## 3. Why this maps cleanly onto `logit_mle`

Three pieces of the existing code do almost all the work:

1. **Step 1 is `RandomCoefficients.fit`.** It already returns \(\hat\delta\) (per product),
   \(\hat\sigma\) (per characteristic), and \(\hat\xi\) (per market, if `market_fe=True`).
   We use \(\hat\delta,\hat\xi\) as the fixed mean-utility offset and \(\hat\sigma\) only to
   *locate* the grid. (Houde–Myers: a mis-specified-normal mixed logit recovers the means /
   \(\hat\delta\) consistently even when \(f(\beta)\) is wrong, which is what licenses
   holding them fixed.)

2. **Building \(Z\) is `_s_ijt` with `sigma = ones`.** The grid-point share
   \(s_{jt}(\beta_r)=\mathrm{softmax}_j(\hat\delta_{jt}+\sum_g x_{jg}\beta_{rg})\) is exactly
   what `_s_ijt(delta_inside, sigma, xi, x2, nu_i, avail)` computes if we set
   `sigma = jnp.ones(G)` and `nu_i = beta_grid` (shape `(R, G)`). It returns `(R, J, T)`,
   which is \(Z\) up to a reshape. No new utility/share math.

3. **The output is a `(nodes, weights)` rule.** \(\hat F(\beta)=\{(\beta_r,\theta_r)\}\) is
   precisely the `(nu_i, w_i)` integration interface, with `sigma = ones`. So diversion,
   elasticity, and shares under the nonparametric distribution are computed by the existing
   `_diversion_jk` / `_compute_elasticity` / `_s_jt` functions with `nu_i = beta_grid`,
   `w_i = theta_hat`, `sigma = ones`. **No new downstream math, and results are directly
   comparable to the normal-RC run.**

The only genuinely new code is grid construction and the constrained-LS solve.

## 4. Architecture

A standalone estimator `FixedGridRC` (module `src/logit_mle/fixed_grid.py`). It does **not**
inherit `DiscreteChoiceModel`'s `fit()` — that loop is L-BFGS over a likelihood, whereas this
is constrained least squares — but it lives in the same package and reuses the module-level
JAX functions in `random_coefficients.py`.

```python
class FixedGridRC:
    """Second-stage fixed-grid estimator. Construct from raw step-1 arrays, or
    from a fitted RandomCoefficients via `FixedGridRC.from_fitted(...)`."""

    def __init__(self, delta_hat, sigma_hat, xi_hat=None, *,
                 x2, availability_matrix, q_jt, market_fe=False,
                 beta_grid=None, drop_cols=(),
                 grid_size=200, span=3.0, grid_seed=0, solver="cvxpy"):
        """
        delta_hat           : (J-1,) step-1 inside-good mean utilities (offset, held fixed)
        sigma_hat           : (G,)   step-1 RC std devs (used only to locate the grid)
        xi_hat              : (T,) or (T-1,) market FE; ignored/zeros if market_fe=False
        x2                  : (J, G) product characteristics
        availability_matrix : (J, T)
        q_jt                : (J, T) observed purchase counts  ->  observed shares
        market_fe           : bool
        beta_grid           : (R, G_grid) grid over the *gridded* columns only; if None,
                              built from sigma_hat[keep] (Section 6)
        drop_cols           : tuple[int] columns of x2 held homogeneous (no grid dim;
                              deviation fixed at 0) -- e.g. the price column
        grid_size,span,grid_seed : default-grid controls (only when beta_grid is None)
        solver              : "cvxpy" (default); behind the _solve_qp interface (Section 5.4)
        """

    @classmethod
    def from_fitted(cls, rc_model, step1_result, *,
                    beta_grid=None, drop_cols=(), solver="cvxpy"):
        """Pull (delta_hat, sigma_hat, xi_hat) and the data off a fitted
        RandomCoefficients + its OptimizeResult, then delegate to __init__."""
        p = rc_model._unpack_theta(step1_result.x)   # {delta_inside, sigma, xi, ...}
        return cls(p["delta_inside"], p["sigma"], p["xi"],
                   x2=rc_model.x2, availability_matrix=rc_model.availability_matrix,
                   q_jt=rc_model.q_jt, market_fe=rc_model.market_fe,
                   beta_grid=beta_grid, drop_cols=drop_cols, solver=solver)

    def fit(self, *, mu_grid=None, k=10, train_pct=0.9, random_state=2025,
            select="one_se", solver_opts=None) -> "FixedGridResult":
        """Cross-validate mu (folding over markets), refit on all markets, return the result."""
```

`FixedGridResult` stores `delta_hat, xi_hat, beta_grid, theta_hat, mu_hat, drop_cols, cv_curve`
and exposes the downstream API by delegating to the existing JAX functions with
`sigma = sigma_vec` (1 on gridded cols, 0 on dropped), `nu_i = nu_full` (grid placed with
zeros on dropped cols), `w_i = theta_hat`:

```python
class FixedGridResult:
    def shares(self) -> Array:            # (J, T)   via _s_jt
    def diversion_matrix(self) -> Array:  # (J, J)   via _diversion_jk
    def elasticity_matrix(self, *, prices, price_coeff, price_col=None) -> Array
    def f_beta(self) -> tuple[Array, Array]   # (beta_grid, theta_hat): the estimated mixing dist
```

Design intent: a `FixedGridResult` is interchangeable with a `RandomCoefficients` evaluated
at `(nu_i=nu_full, w_i=theta_hat, sigma=sigma_vec)`. A convenience `to_random_coefficients()`
can return such an `RandomCoefficients` so existing notebooks/tests work unchanged.

## 5. Algorithm

### 5.1 Step 1 (reuse)

```python
rc = RandomCoefficients(avail, q_jt, x2=x2, nu_i=nu_i, w_i=w_i, market_fe=market_fe)
step1 = rc.fit(seed=...)
delta_inside_hat, sigma_hat, xi_inside_hat = unpack(step1.x)   # rc._unpack_theta
```

Holds \(\hat\delta,\hat\xi\) fixed for everything below; we do **not** update them (HHO's
recommended two-step; the iterative update that re-fits \(\delta\) is their Appendix B /
Train 2008, which they flag as very slow and do not use).

### 5.2 Grid (Section 6)

`beta_grid` shape `(R, G)`, deviations in \(\beta\)-space centered at 0.

### 5.3 Build \(Z\) (JAX)

```python
# drop_cols are held homogeneous: sigma_vec = 1 on gridded cols, 0 on dropped;
# the grid is placed into full G-space with zeros on the dropped columns.
keep      = [g for g in range(G) if g not in drop_cols]
sigma_vec = jnp.zeros(G).at[jnp.array(keep)].set(1.0)                  # (G,)
nu_full   = jnp.zeros((R, G)).at[:, jnp.array(keep)].set(beta_grid)    # (R, G), 0 on dropped
S_rjt = _s_ijt(delta_inside_hat, sigma_vec, xi_hat, x2, nu_full, avail)  # (R, J, T)
Z = np.asarray(S_rjt.reshape(R, J * T).T)         # (JT, R)
s_obs = np.asarray((q_jt / q_jt.sum(axis=0)).reshape(J * T))  # observed shares, (JT,)
```

Reduce to the **Gram form** once (keeps the QP \(R\)-dimensional, independent of \(JT\)):
`Gram = Z.T @ Z` (`(R,R)`, PSD), `b = Z.T @ s_obs` (`(R,)`). Symmetrize `Gram` to kill
numerical asymmetry before handing to cvxpy.

### 5.4 Solve the QP (cvxpy, behind an interface)

Internal contract: `_solve_qp(Gram, b, mu, warm_start=None) -> theta`. Default backend
(`_solve_qp_cvxpy`) uses a **DPP-parameterized** problem in `mu` so it canonicalizes once and
re-solves fast across the `mu` sweep and folds:

```python
import cvxpy as cp
theta = cp.Variable(R, nonneg=True)
mu_p  = cp.Parameter(nonneg=True)
G_psd = cp.psd_wrap(Gram)                         # constant PSD across the sweep
obj   = cp.Minimize(cp.quad_form(theta, G_psd) - 2 * b @ theta
                    + mu_p * cp.sum_squares(theta))
prob  = cp.Problem(obj, [cp.sum(theta) == 1])     # nonneg via Variable(nonneg=True)
# sweep: set mu_p.value = mu; prob.solve(solver=cp.OSQP, warm_start=True); check prob.status
```

Notes: OSQP (or Clarabel) is the backend solver; assert `prob.status == "optimal"`;
warm-start along the `mu` path (continuation) so each solve starts from the previous
\(\hat\theta\). A future `_solve_qp_pgd` (pure-JAX projected gradient onto the simplex) can
implement the same signature without touching the rest; not needed at current scale.

### 5.5 Cross-validate `mu` (`cross_validate_mu`)

- `mu_grid`: ascending, default `np.r_[0.0, np.logspace(-6, 1, 24)]` (25 points) — the `0.0`
  endpoint is the FKRB special case.  Kept modest because each `mu` re-factorizes an R×R KKT,
  so CV cost is linear in the number of `mu` points (see §11); 25 is ample for one-SE.
- Folds (`make_market_folds`, PyCMS `create_kfolds` style): `k` *repeated random* train/test
  splits **over markets** — each draws a `train_pct` (default 0.9) fraction of the `T` markets
  without replacement (seed `random_state + fold`), complement = test.  Not a disjoint
  partition.  Folding over markets (not raw `(j,t)` cells) keeps within-market shares coherent.
  The design matrix is built once; folds index its rows by market (`row % T`).
- For each fold, fit the QP on the training markets (warm-started up the `mu` grid via the
  shared `make_qp_solver`) and score held-out **share MSE**
  \(\frac{1}{|\text{val}|}\sum_{(j,t)\in\text{val}}(s^{obs}_{jt}-\sum_r\theta_r s_{jt}(\beta_r))^2\).
- Aggregate per `mu` to mean / std (`ddof=1`) / se (`std/\sqrt{k}`).
- `select="one_se"` (default, per HHO): the largest `mu` whose mean OOS MSE is within one SE
  of the minimum (more regularization).  `select="argmin"` is available.  Refit `theta_hat` on
  all markets at the selected `mu`.

### 5.6 Refit and wrap

Refit \(\hat\theta\) at `mu_hat` on all markets; return `FixedGridResult`.

## 6. Grid construction

HHO lay the grid with a **Halton sequence over a box** covering the support, chosen to keep
correlation among grid points low (their continuous-distribution MC, §4.2: a Halton grid on
\([-4.5, 3.5]^2\), \(R\in\{25,50,100,250\}\)). So Halton is the default — easy, with one caveat
about the existing helpers.

**Do not reuse `halton_draws` verbatim.** It is a *normal integration rule*: it applies the
inverse-normal transform (`norm.ppf`) to the Halton uniforms and returns equal weights `1/n`.
A fixed grid wants (i) low-discrepancy points that **cover the support of \(\beta\)**
(space-filling over a box, not concentrated near 0 by a Gaussian density), and (ii) **no
weights** — the weights are exactly what the QP estimates. Likewise `sparse_grid` returns a
Genz–Keister normal *quadrature* rule (nodes + weights tuned for integration), not a coverage
grid.

Add a dedicated helper `beta_grid_box(sigma_hat, *, span=3.0, n, seed=0)` to `quadrature.py`:

```python
from scipy.stats.qmc import Halton
u = Halton(d=G, scramble=True, seed=seed).random(n)   # uniform in [0,1]^G, NO ppf
lo, hi = -span * sigma_hat, span * sigma_hat           # box per dimension, centered at 0
beta_grid = lo[None, :] + (hi - lo)[None, :] * u       # (n, G), space-filling over the box
```

i.e. low-discrepancy uniform coverage of
\([-\text{span}\cdot\hat\sigma_g,\ \text{span}\cdot\hat\sigma_g]\) per dimension, centered at 0
(the mean is absorbed in \(\hat\delta\)). Return only the grid; weights come from the QP.
With `drop_cols`, `FixedGridRC` calls this with `sigma_hat[keep]`, so the grid has dimension
\(G_{\text{grid}} = G - |\text{drop\_cols}|\) and the dropped columns are padded with zeros
when building \(Z\) (Section 5.3).

Caveats:

- **Curse of dimensionality.** \(R\) grows fast in \(G\); a dense 4-D grid is the regime where
  \(Z^\top Z\) goes near-singular — exactly why HHO (ridge), not plain FKRB, is the default here.
- A coarse first pass then a refined grid around the support of \(\hat\theta\) is a reasonable
  manual workflow (don't automate yet).
- **Optional normal-placement variant:** inverse-normal Halton scaled by \(\hat\sigma\) (the
  existing `halton_draws` nodes, weights discarded) concentrates grid resolution where mass is
  more likely. Defensible, but it bakes in a normal-ish placement, slightly against the
  cover-the-support / low-correlation rationale. Default to the box.

## 7. Observed shares and weighting

Default objective is unweighted LS on shares (matches HHO Eq. 9–10). The outside good is row
`J-1`; `q_jt` includes it, so `s_obs = q_jt / q_jt.sum(axis=0)`. **Open question (Section 10):**
whether to weight observations by market size \(\sum_j q_{jt}\) (a GLS-flavored variant);
default off to stay faithful to HHO.

## 8. Dependencies / packaging

Add cvxpy as an **optional extra**, mirroring the existing `sparse = ["chaospy"]` pattern:

```toml
[project.optional-dependencies]
sparse = ["chaospy"]
hho    = ["cvxpy"]
```

Core install stays `numpy / scipy / jax / jaxlib`. `FixedGridRC` raises a clear ImportError
with the install hint if cvxpy is missing (same pattern as `sparse_grid`).

## 9. Public surface

```python
from logit_mle import RandomCoefficients, FixedGridRC, beta_grid_box

rc    = RandomCoefficients(avail, q_jt, x2=x2, nu_i=nu_i, w_i=w_i, market_fe=True)
step1 = rc.fit(seed=2025)

# (a) from a fitted model + result; drop the price RC (col 0), grid only the embeddings
fg  = FixedGridRC.from_fitted(rc, step1, drop_cols=(0,))
res = fg.fit(n_folds=10, select="one_se")

# (b) or from raw arrays (e.g. delta/xi/sigma loaded from a pyBLP run elsewhere)
fg2 = FixedGridRC(delta_hat, sigma_hat, xi_hat,
                  x2=x2, availability_matrix=avail, q_jt=q_jt,
                  market_fe=True, drop_cols=(0,))

D_np = res.diversion_matrix()              # nonparametric f(beta)
D_n  = rc.diversion_matrix(step1.x)        # normal-RC, for comparison
beta_grid, theta_hat = res.f_beta()        # the estimated mixing distribution
```

## 10. Open questions / decisions

1. **Observation weighting** — unweighted (HHO default) vs. weight by market size. *Default
   unweighted; revisit if shares are very uneven across markets.*
2. **Updating \(\delta\)** — we deliberately do **not** (HHO's recommendation; Houde–Myers).
   The iterative update (HHO App. B / Train 2008) is out of scope; note as a known
   non-goal.
3. **Price random coefficient (decided: `drop_cols`).** Listed columns are held homogeneous
   (deviation \(\equiv 0\), absorbed in \(\hat\delta\)) and excluded from the grid, so e.g.
   `drop_cols=(price_col,)` kills the price RC and grids only the embeddings — reducing \(R\)
   and addressing the calibrated-\(\alpha\) / weak-identification concern. **Consistency
   caveat:** \(\hat\delta\) comes from a step-1 model that *did* include the dropped RC; for a
   fully coherent two-step, run step 1 with the RC also removed on the dropped columns (or
   accept the mild mismatch — what "kill it to make life easier" buys).
4. **CV fold axis (decided: markets, random subsample).** Fold over markets via repeated
   random `train_pct` splits (PyCMS `create_kfolds` style), not a disjoint partition.  Assumes
   markets are exchangeable enough to subsample — fine at the hotel/simulation scale (large
   `T`); revisit only if `T` is small.
5. **Identification needs many markets.** The second stage estimates `R` weights from
   cross-market variation in observed shares, so it needs \(T\) large relative to \(J\) (and
   the grid `R`) — the aggregate analogue of FKRB/HHO's many-observations regime, and the same
   \(T \gg J\) regime the Compiani–Christensen de-biasing requires.  In simulation with
   *noiseless* shares even small \(T\) recovers; with sampling noise (finite consumers per
   market) recovery degrades sharply at small \(T\) and converges only as \(T\) grows (e.g.
   corr-with-truth ~0.86 at \(T=6\) vs ~0.99 at \(T\ge 60\), \(J=9\)), and the nonparametric
   estimator converges *slower* than the correctly-specified normal (it pays a variance cost
   for flexibility).  **What must vary across markets is the identifying variation** — product
   assortment (availability) and/or price/\(\xi\).  The capstone uses assortment variation (the
   cleanest case); the hotel application has ~fixed assortment and relies on price/time
   variation, which is weaker — consistent with the paper's "lack of choice-set variation."

## 11. Performance and solver scaling

- **Cost is factorization-bound.** Each `mu` solve re-factorizes a dense \(R \times R\) KKT
  system (the ridge enters as `P = Gram + mu*I`, a diagonal change OSQP cannot reuse a
  factorization across).  Per-solve scales ~\(R^{2.7}\): ~7 ms at `R=250`, ~37 ms at `R=500`,
  ~220 ms at `R=1000`, ~1.4 s at `R=2000` (`JT=5000`).  A full CV (`k=10`, 25 `mu`) is well
  under ~20 s for `R<=500`, ~1 min at `R=1000`.  Building `Z`/`Gram` is negligible by
  comparison.
- **Loosening tolerance does *not* help** — `eps` from `1e-9` to `1e-4` leaves sweep time
  unchanged (it's the factorization, not iteration count, that dominates).  Warm-start and the
  DPP parameterization help iterations, so their benefit also shrinks as `R` grows.
- **OSQP is the default and is faster than Clarabel** here (≈2× at `R=250`, ≈1.3× at `R=1000`),
  and the two agree to ~1e-5 on `theta` (a useful correctness cross-check).  Use **Clarabel**
  (`solver=cp.CLARABEL`) as a fallback when `Gram` is near-singular (very dense grid,
  \(R \to JT\)) where interior-point is more robust than ADMM, or for a high-accuracy final
  refit.
- **Levers if `R` grows large** (in order): fewer `mu` points / adaptive search; a coarser
  `beta` grid (the curse of dimensionality caps `R` in the low hundreds anyway); then the
  pure-JAX projected-gradient backend (gradient `(Gram+mu*I)theta - b` is an \(O(R^2)\) matvec,
  no factorization, GPU-batchable across `mu`), which slots behind the existing `_solve_qp`
  `backend` switch.

## 12. Testing plan (mirror `tests/`)

- **FKRB special case:** at `mu = 0`, `_solve_qp` reproduces a reference simplex-constrained
  NNLS (cross-check against `scipy.optimize.nnls` projected to the simplex, or an `osqp`
  oracle on a small instance).
- **Constraint sanity:** `theta_hat >= 0` (within tol) and `sum(theta_hat) == 1`.
- **Selection consistency (simulation):** generate data from a *discrete* \(f(\beta)\) whose
  support is a subset of the grid; check `FixedGridRC` puts positive weight on the true
  support points and ~zero elsewhere (HHO §4.1).
- **Normal-recovery / referee check:** when the DGP is normal-RC, the nonparametric diversion
  matrix is close to the normal-RC diversion matrix — the "is normality driving the results"
  reassurance.
- **Reuse identity:** `FixedGridResult.shares()` equals `RandomCoefficients` shares built with
  `(nu_i=beta_grid, w_i=theta_hat, sigma=ones)` — confirms the `(nodes, weights)` reuse.
- **Solver interface:** `cvxpy` and (later) `pgd` backends agree to tolerance on a small QP.

## 13. References

- Heiss, F., Hetzenecker, S., Osterhaus, M. (2022). "Nonparametric estimation of the random
  coefficients model: An elastic net approach." *Journal of Econometrics* 229(2), 299–321.
- Fox, J. T., Kim, K. I., Ryan, S. P., Bajari, P. (2011). "A simple estimator for the
  distribution of random coefficients." *Quantitative Economics* 2, 381–418.
- Fox, Kim, Ryan, Bajari (2016); Houde & Myers (2019) — two-step plug-in for models with
  fixed and random coefficients (HHO §5).
- Train, K. (2008). "EM algorithms for nonparametric estimation of mixing distributions."
  (iterative \(\delta\)-updating alternative; not implemented.)
- Wu & Yang (2014), nonnegative elastic net; Duchi et al. (2008) / Condat (2016), simplex
  projection (relevant only to a future PGD backend).
