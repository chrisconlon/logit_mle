# logit_mle — Claude project context

JAX-based maximum likelihood estimation for discrete-choice demand. Models —
`Logit`, `NestedLogit`, `RandomCoefficients` — all subclass `DiscreteChoiceModel`
([src/logit_mle/base.py](src/logit_mle/base.py)) and share `fit()` (L-BFGS-B over a
JAX likelihood), `shares()`, `diversion_matrix()`, `elasticity_matrix()`. Plus a
fixed-grid nonparametric second stage (FKRB/HHO).

## Layout

- [src/logit_mle/](src/logit_mle/) — `base.py`, `logit.py`, `nested_logit.py`,
  `random_coefficients.py`; `quadrature.py` (integration grids); `fixed_grid.py`
  (FKRB/HHO).
- [tests/](tests/) — pytest.
- [docs/hho_design.md](docs/hho_design.md) — full design of the fixed-grid estimator.

## Fixed-grid nonparametric RC (FKRB / HHO)

`fixed_grid.py`: `FixedGridRC` / `FixedGridResult` recover a *nonparametric* mixing
distribution on a fixed grid as a second stage on a fitted `RandomCoefficients`
(`FixedGridRC.from_fitted(rc, step1)`, or raw arrays). It holds the step-1 mean
utilities fixed and estimates grid weights by cross-validated nonnegative elastic
net (the QP is `min ||s_obs - Z theta||^2 + mu||theta||^2` s.t. `theta>=0, sum=1`).

Key design fact: the output is a `(nodes, weights)` rule, so downstream
`shares`/`diversion`/`elasticity` are computed by the **existing** `RandomCoefficients`
JAX functions with the grid as `nu_i`, `theta` as `w_i`, and `sigma` a gridded/dropped
indicator (`drop_cols` holds a characteristic homogeneous, e.g. price). No new share math.

Reuse gotchas:

- `cvxpy` is an optional `[hho]` extra (lazy-imported); solver is OSQP. The hho tests
  skip if cvxpy is absent.
- Cost is factorization-bound (~`R^2.7`/solve); fine for `R <= ~500`. Levers at large
  `R`: fewer `mu` points, coarser grid, or a JAX projected-gradient backend (the
  `_solve_qp` `backend` switch exists for it).
- Identification needs many markets with real cross-market variation (assortment or
  price). Diversion is robust, but the raw grid weights are non-unique on a dense grid
  without a ridge (`mu>0`) — report substitution, not the weights. See docs §10–11.

## Running tests

`pytest`, `cvxpy`, and `chaospy` are **not** in the project venv. Run via uv with
ephemeral extras:

```bash
uv run --with pytest --with cvxpy --with chaospy pytest -q
```

Drop `--with cvxpy` to confirm the hho path stays optional (those tests skip);
`--with chaospy` is needed only for `test_quadrature.py` (sparse grids).

## Conventions

- JAX for the numerical core; numpy in tests; seeds always on; lean type hints.
- Outside good is the **last** row/column (index `J-1`), utility normalized to 0.
- Data: `(J, T)` = products × markets; `availability_matrix` and `q_jt` are `(J, T)`.
