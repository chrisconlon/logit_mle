# logit_mle

JAX-based maximum likelihood estimation for discrete choice models: Logit, Nested Logit, and Random Coefficients.

## Installation

```bash
uv pip install -e .

# With sparse grid support (adds chaospy dependency):
uv pip install -e ".[sparse]"
```

Requires Python 3.10+ and JAX.

## Models

All models share a common base class `DiscreteChoiceModel` with a unified interface:

```python
model.fit(...)                              # MLE via L-BFGS-B
model.shares(theta)                         # Market shares s_jt, shape (J, T)
model.diversion_matrix(theta)               # Diversion ratios D_jk, shape (J, J)
model.jacobian(theta)                       # ds_j/dd_k, shape (J, J)
model.elasticity_matrix(theta, prices=...)  # Price elasticity matrix, shape (J, J)
model.log_likelihood(theta)                 # Negative log-likelihood (scalar)
```

### Logit

```python
from logit_mle import Logit

model = Logit(availability_matrix, q_jt)
result = model.fit(seed=2025)
D = model.diversion_matrix(result.x)
```

**Parameters:** `theta = delta[J-1]` (mean utility; outside good normalized to 0).

**Diversion:** Closed-form IIA formula `D_jk = s_k / (1 - s_j)`.

### Nested Logit

```python
from logit_mle import NestedLogit

# nesting_ids: integer vector of length J-1 (inside goods only)
# e.g., [0, 0, 0, 1, 1, 1, 2, 2] for 3 nests with 8 inside goods
model = NestedLogit(availability_matrix, q_jt, nesting_ids=nesting_ids)
result = model.fit(seed=2025)
D = model.diversion_matrix(result.x)
```

**Parameters:** `theta = (delta[J-1], sigma)` where `sigma` is Train's nesting parameter (Berry/Cardell `rho = 1 - sigma`).

**Constructor:** `nesting_ids` is a `(J-1,)` integer array assigning each inside good to a nest. The outside good is automatically placed in its own singleton nest.

**Diversion:** Closed-form formula (tau_in/tau_out notation). A vmap product-removal method is also available via `model.compute_diversion_matrix_vmap(theta)` for testing.

**Additional methods:**
- `model.compute_model_shares(theta)` returns `(S_gjt, S_gt, S_jt)` -- full share decomposition by nest
- `model.compute_diversion_matrix_from_formula(theta)` -- alias for `diversion_matrix()`

### Random Coefficients (RCN / RCC)

```python
from logit_mle import RandomCoefficients, halton_draws

nu_i, w_i = halton_draws(G=4, n=1000, seed=42)

# RCN (no market fixed effects)
model = RandomCoefficients(availability_matrix, q_jt,
                           x2=x_jg, nu_i=nu_i, w_i=w_i)

# RCC (with market fixed effects xi_t)
model = RandomCoefficients(availability_matrix, q_jt,
                           x2=x_jg, nu_i=nu_i, w_i=w_i, market_fe=True)

result = model.fit(seed=2025)
D = model.diversion_matrix(result.x)
```

**Parameters:**
- RCN: `theta = (delta[J-1], sigma[G])` -- dim `J-1+G`
- RCC: `theta = (delta[J-1], sigma[G], xi[T-1])` -- dim `J-1+G+T-1`

where `G` is the number of random coefficient characteristics.

**Constructor arguments:**
- `x2`: `(J, G)` product characteristics matrix
- `nu_i`: `(I, G)` integration nodes (from `halton_draws` or `sparse_grid`)
- `w_i`: `(I,)` integration weights
- `market_fe`: if `True`, includes market fixed effects (RCC); default `False` (RCN)

**Diversion:** Integrated over individuals: `D_jk = E_i[ s_ik/(1-s_ij) * s_ij/s_j ]`.

## Elasticities

All models support price elasticity computation:

```python
# Logit / Nested Logit: pass scalar price coefficient
eta = model.elasticity_matrix(theta, prices=p, price_coeff=-0.5)

# Random Coefficients with heterogeneous price sensitivity:
#   beta_i^p = price_coeff + sigma[price_col] * nu_i[price_col]
eta = model.elasticity_matrix(theta, prices=p, price_coeff=-0.5, price_col=0)
```

The Jacobian `ds_j/dd_k` is also available directly via `model.jacobian(theta)`.

## Integration grids

Two built-in methods for generating integration nodes and weights:

### Quasi-Monte Carlo (recommended)

Scrambled Halton sequence with inverse-normal transform. Robust across all sigma regimes, especially with large heterogeneity or high G.

```python
from logit_mle import halton_draws

nu_i, w_i = halton_draws(G=7, n=1000, seed=42)
```

### Sparse grids (requires `chaospy`)

Genz-Keister deterministic quadrature rules. Best when sigma is small-to-moderate and G is low. The `genz_keister_16` rule matches Kronrod-Patterson-Normal (KPN) grid files.

```python
from logit_mle import sparse_grid

nu_i, w_i = sparse_grid(G=4, order=3)                             # GK-16 (default)
nu_i, w_i = sparse_grid(G=4, order=3, rule="genz_keister_24")     # GK-24 (more nodes, higher accuracy)
```

Requires the `sparse` extra: `uv pip install -e ".[sparse]"`

## Data conventions

- **Dimensions:** `J` = total products (inside + outside good), `T` = markets
- **Outside good:** Always the last row/column (index `J-1`), with utility normalized to 0
- **Availability matrix:** `(J, T)` boolean -- `True` if product `j` is available in market `t`
- **Quantities:** `q_jt` is `(J, T)` -- observed purchase counts
- **Diversion matrix:** `(J, J)` computed under full availability (all products available). Diagonal is 0

## Augmented likelihood

All models support estimation with an augmented likelihood that combines a model component and a diversion data component:

```python
model = Logit(availability_matrix, q_jt, diversion_data=D_observed)
result = model.fit(
    aug_div=True,
    penalty=0.0001,                        # weight on model likelihood
    diversion_rows=(3, 7, 17, 18, 29, 30), # which rows of diversion_data to fit (None = all)
)
```
