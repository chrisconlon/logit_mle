"""
Integration nodes and weights for numerical integration over standard normals.

Two approaches:

- **Sparse grids** (``sparse_grid``): Deterministic Genz-Keister rules via
  chaospy.  Best when sigma is small-to-moderate and G is low (2-5).
  Matches KPN files at ``genz_keister_16`` orders 0-3.

- **Quasi-Monte Carlo** (``halton_draws``): Scrambled Halton sequence with
  inverse-normal transform.  More robust across sigma regimes, especially
  with large heterogeneity or high G.  Recommended default for G > 5.
"""
from __future__ import annotations

import numpy as np

try:
    import chaospy
    _HAS_CHAOSPY = True
except ImportError:
    _HAS_CHAOSPY = False


def sparse_grid(G, order, *, rule="genz_keister_16"):
    """Generate sparse quadrature nodes and weights for G-dimensional standard normal.

    Parameters
    ----------
    G : int
        Number of dimensions (random coefficient characteristics).
    order : int
        Accuracy level.  Higher = more nodes and better accuracy.
        Typical values: 3-5 for estimation, 1-2 for quick checks.
    rule : str
        Quadrature rule.  Options (from fewest to most nodes per level):

        - ``"genz_keister_16"`` — matches KPN files; most parsimonious (default)
        - ``"genz_keister_18"`` — slightly more nodes, higher accuracy
        - ``"genz_keister_22"`` — more nodes still
        - ``"genz_keister_24"`` — most nodes per level, highest accuracy

    Returns
    -------
    nu_i : ndarray, shape (I, G)
        Quadrature nodes (standard normal draws).
    w_i : ndarray, shape (I,)
        Quadrature weights (sum to 1).

    Examples
    --------
    >>> nu_i, w_i = sparse_grid(G=4, order=3)
    >>> nu_i.shape
    (209, 4)

    >>> from logit_mle import RandomCoefficients
    >>> model = RandomCoefficients(avail, q_jt, x2=x_jg, nu_i=nu_i, w_i=w_i)
    """
    if not _HAS_CHAOSPY:
        raise ImportError(
            "chaospy is required for sparse_grid(). "
            "Install it with: uv pip install chaospy"
        )

    dist = chaospy.Iid(chaospy.Normal(0, 1), G)
    nodes, weights = chaospy.generate_quadrature(
        order, dist, rule=rule, sparse=True
    )

    # chaospy returns nodes as (G, I); transpose to (I, G)
    return np.asarray(nodes.T), np.asarray(weights)


def halton_draws(G, n, *, seed=0, scramble=True):
    """Generate quasi-Monte Carlo draws from a G-dimensional standard normal.

    Uses a scrambled Halton sequence transformed via the inverse normal CDF.
    More robust than sparse grids when sigma is large or G is high.

    Parameters
    ----------
    G : int
        Number of dimensions (random coefficient characteristics).
    n : int
        Number of draws.  Typical values: 1000-5000 for estimation.
    seed : int
        Seed for the scrambled Halton sequence (default 0).
    scramble : bool
        If True (default), use Owen scrambling for better uniformity.

    Returns
    -------
    nu_i : ndarray, shape (n, G)
        Quasi-random draws from standard normal.
    w_i : ndarray, shape (n,)
        Uniform weights (1/n each, sum to 1).

    Examples
    --------
    >>> nu_i, w_i = halton_draws(G=4, n=1000)
    >>> nu_i.shape
    (1000, 4)

    >>> from logit_mle import RandomCoefficients
    >>> model = RandomCoefficients(avail, q_jt, x2=x_jg, nu_i=nu_i, w_i=w_i)
    """
    from scipy.stats.qmc import Halton
    from scipy.stats import norm

    sampler = Halton(d=G, scramble=scramble, seed=seed)
    uniform = sampler.random(n=n)
    nu_i = norm.ppf(uniform)
    w_i = np.ones(n) / n
    return nu_i, w_i
