"""
Quadrature rule comparison for mixed logit share computation.

1. Verify that chaospy's Patterson rule matches the KPN .asc grid files.
2. Compare quadrature rules (Patterson sparse, Gauss-Hermite sparse,
   Clenshaw-Curtis sparse, tensor-product Gauss-Hermite) for computing
   E_ν[s_j(ν)] in a 3-dimensional mixed logit with J=25 products.

The "truth" is a high-order tensor-product Gauss-Hermite rule (e.g., 30^3 = 27000 nodes).
"""
import numpy as np
import chaospy
from pathlib import Path

# ── Test problem: mixed logit shares ─────────────────────────────

def mixed_logit_shares(delta, x_jg, sigma, nu_i, w_i):
    """Compute E_i[s_j(nu_i)] for a mixed logit.

    Parameters
    ----------
    delta : (J,) mean utility (outside good = 0 at index J-1)
    x_jg  : (J, G) product characteristics
    sigma : (G,) std dev of random coefficients
    nu_i  : (I, G) quadrature nodes (standard normal draws)
    w_i   : (I,) quadrature weights

    Returns
    -------
    s_j : (J,) expected market shares
    """
    # V_ij = delta_j + sum_g x_jg * sigma_g * nu_ig
    # Shape: (I, J)
    V_ij = delta[None, :] + (x_jg[None, :, :] * sigma[None, None, :] * nu_i[:, None, :]).sum(axis=-1)

    # Softmax over products for each individual
    V_max = V_ij.max(axis=1, keepdims=True)
    exp_V = np.exp(V_ij - V_max)
    s_ij = exp_V / exp_V.sum(axis=1, keepdims=True)  # (I, J)

    # Integrate: E[s_j] = sum_i w_i * s_ij
    s_j = (w_i[:, None] * s_ij).sum(axis=0)  # (J,)
    return s_j


def setup_test_problem(J=25, G=3, seed=42):
    """Create a fixed test problem."""
    rng = np.random.RandomState(seed)
    J_in = J
    J_total = J + 1  # including outside good

    delta = np.zeros(J_total)
    delta[:J_in] = rng.randn(J_in) * 1.5  # spread utilities

    x_jg = rng.randn(J_total, G)
    sigma = np.array([0.5, 0.8, 0.3])  # moderate heterogeneity

    return delta, x_jg, sigma


# ── 1. Verify chaospy Patterson matches KPN files ────────────────

def test_chaospy_patterson_matches_kpn():
    """Check that chaospy's Patterson nodes/weights match the KPN .asc files."""
    kpn_dir = Path(__file__).parent.parent.parent.parent / \
        "Library/CloudStorage/Dropbox/CMS_Estimator/data/sparse_grids"

    if not kpn_dir.exists():
        # Try relative to CMS_Estimator
        kpn_dir = Path("/Users/christopherconlon/Library/CloudStorage/Dropbox/"
                       "CMS_Estimator/data/sparse_grids")

    if not kpn_dir.exists():
        import pytest
        pytest.skip("KPN grid files not found")

    G = 3
    dist = chaospy.Iid(chaospy.Normal(0, 1), G)

    for level in range(1, 8):
        kpn_file = kpn_dir / f"KPN_d{G}_l{level}.asc"
        if not kpn_file.exists():
            continue

        # Load KPN file
        kpn_data = np.loadtxt(kpn_file, delimiter=",")
        if kpn_data.ndim == 1:
            kpn_data = kpn_data.reshape(1, -1)
        kpn_nodes = kpn_data[:, :-1]   # (I, G)
        kpn_weights = kpn_data[:, -1]   # (I,)
        I_kpn = len(kpn_weights)

        # Generate chaospy Patterson at same level
        # chaospy's "order" parameter may not map 1:1 to KPN "level"
        # Try matching by node count
        cp_nodes, cp_weights = chaospy.generate_quadrature(
            level, dist, rule="patterson", sparse=True
        )
        cp_nodes = cp_nodes.T  # chaospy returns (G, I), transpose to (I, G)
        I_cp = len(cp_weights)

        print(f"Level {level}: KPN has {I_kpn} nodes, chaospy has {I_cp} nodes")

        if I_kpn == I_cp:
            # Sort both by nodes to align (order may differ)
            kpn_order = np.lexsort(kpn_nodes.T)
            cp_order = np.lexsort(cp_nodes.T)

            kpn_sorted = kpn_nodes[kpn_order]
            cp_sorted = cp_nodes[cp_order]
            kpn_w_sorted = kpn_weights[kpn_order]
            cp_w_sorted = cp_weights[cp_order]

            nodes_match = np.allclose(kpn_sorted, cp_sorted, atol=1e-10)
            weights_match = np.allclose(kpn_w_sorted, cp_w_sorted, atol=1e-10)

            print(f"  Nodes match: {nodes_match}")
            print(f"  Weights match: {weights_match}")

            if not nodes_match or not weights_match:
                # Show max differences
                if nodes_match is False:
                    print(f"  Max node diff: {np.max(np.abs(kpn_sorted - cp_sorted)):.2e}")
                if weights_match is False:
                    print(f"  Max weight diff: {np.max(np.abs(kpn_w_sorted - cp_w_sorted)):.2e}")
        else:
            print(f"  Node counts differ — level mapping may differ between KPN and chaospy")


# ── 2. Quadrature rule comparison ────────────────────────────────

def test_quadrature_accuracy():
    """Compare quadrature rules for computing mixed logit shares.

    Rules tested:
    - Patterson sparse grid (levels 1-7)
    - Gauss-Hermite sparse grid (levels 1-7)
    - Clenshaw-Curtis sparse grid (levels 1-7)
    - Tensor-product Gauss-Hermite (n^3 for n = 3, 5, 7, 10, 15)

    Truth: tensor-product Gauss-Hermite with n=30 (27000 nodes).
    """
    delta, x_jg, sigma = setup_test_problem(J=25, G=3)
    G = 3

    dist = chaospy.Iid(chaospy.Normal(0, 1), G)

    # ── Truth: high-order tensor product ──
    n_truth = 30
    nodes_truth, weights_truth = chaospy.generate_quadrature(
        n_truth, dist, rule="gaussian", sparse=False
    )
    s_truth = mixed_logit_shares(delta, x_jg, sigma, nodes_truth.T, weights_truth)
    I_truth = len(weights_truth)
    print(f"\nTruth: tensor-product Gauss-Hermite, n={n_truth}^{G} = {I_truth} nodes")
    print(f"  Shares sum: {s_truth.sum():.10f} (should be 1.0)")

    # ── Compare rules ──
    results = []

    # Sparse grid rules
    for rule_name in ["patterson", "gaussian", "clenshaw_curtis"]:
        for level in range(1, 8):
            try:
                nodes, weights = chaospy.generate_quadrature(
                    level, dist, rule=rule_name, sparse=True
                )
            except Exception as e:
                print(f"  {rule_name} level {level}: FAILED ({e})")
                continue

            nu_i = nodes.T
            w_i = weights
            I = len(w_i)
            s = mixed_logit_shares(delta, x_jg, sigma, nu_i, w_i)

            max_abs_err = np.max(np.abs(s - s_truth))
            rmse = np.sqrt(np.mean((s - s_truth) ** 2))
            share_sum = s.sum()

            results.append({
                "rule": f"{rule_name} (sparse)",
                "level": level,
                "nodes": I,
                "max_abs_err": max_abs_err,
                "rmse": rmse,
                "share_sum": share_sum,
            })

    # Tensor-product Gauss-Hermite
    for n in [3, 5, 7, 10, 15, 20]:
        nodes, weights = chaospy.generate_quadrature(
            n, dist, rule="gaussian", sparse=False
        )
        nu_i = nodes.T
        w_i = weights
        I = len(w_i)
        s = mixed_logit_shares(delta, x_jg, sigma, nu_i, w_i)

        max_abs_err = np.max(np.abs(s - s_truth))
        rmse = np.sqrt(np.mean((s - s_truth) ** 2))

        results.append({
            "rule": "gauss-hermite (tensor)",
            "level": n,
            "nodes": I,
            "max_abs_err": max_abs_err,
            "rmse": rmse,
            "share_sum": s.sum(),
        })

    # KPN files (if available)
    kpn_dir = Path("/Users/christopherconlon/Library/CloudStorage/Dropbox/"
                   "CMS_Estimator/data/sparse_grids")
    if kpn_dir.exists():
        for level in range(1, 8):
            kpn_file = kpn_dir / f"KPN_d{G}_l{level}.asc"
            if not kpn_file.exists():
                continue
            kpn_data = np.loadtxt(kpn_file, delimiter=",")
            if kpn_data.ndim == 1:
                kpn_data = kpn_data.reshape(1, -1)
            nu_i = kpn_data[:, :-1]
            w_i = kpn_data[:, -1]
            I = len(w_i)
            s = mixed_logit_shares(delta, x_jg, sigma, nu_i, w_i)

            max_abs_err = np.max(np.abs(s - s_truth))
            rmse = np.sqrt(np.mean((s - s_truth) ** 2))

            results.append({
                "rule": "KPN file",
                "level": level,
                "nodes": I,
                "max_abs_err": max_abs_err,
                "rmse": rmse,
                "share_sum": s.sum(),
            })

    # ── Print results table ──
    print(f"\n{'Rule':<30} {'Level':>5} {'Nodes':>7} {'Max Err':>12} {'RMSE':>12} {'Σs':>12}")
    print("-" * 80)

    for r in sorted(results, key=lambda x: (x["rule"], x["level"])):
        print(f"{r['rule']:<30} {r['level']:>5} {r['nodes']:>7} "
              f"{r['max_abs_err']:>12.2e} {r['rmse']:>12.2e} {r['share_sum']:>12.8f}")

    # ── Assertions ──
    # All share vectors should sum to 1
    for r in results:
        assert abs(r["share_sum"] - 1.0) < 1e-6, (
            f"{r['rule']} level {r['level']}: shares sum to {r['share_sum']}"
        )


if __name__ == "__main__":
    print("=" * 80)
    print("1. VERIFYING CHAOSPY PATTERSON vs KPN FILES")
    print("=" * 80)
    test_chaospy_patterson_matches_kpn()

    print("\n" + "=" * 80)
    print("2. QUADRATURE RULE COMPARISON FOR MIXED LOGIT SHARES")
    print("=" * 80)
    test_quadrature_accuracy()
