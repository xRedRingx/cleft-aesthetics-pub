"""APPNP parity: does the direct propagation equal torch_geometric's?

**This is the one part of the SR-GNN port whose provenance is not the weight
transplant.** Stage 1 calls ``torch_geometric.nn.APPNP``; this project specified
the propagation directly to avoid the dependency, on the grounds that at K=1,
alpha=0.3, normalize=False, add_self_loops=False it reduces to one line.

"Should reduce to" is exactly what a parity check is for.

Run OUTSIDE the project venv: torch stays out of the test extra, and this is a
one-off verification, not a test dependency.
"""
from __future__ import annotations

import torch
from torch_geometric.nn import APPNP

# The port's own settings.
K = 1
ALPHA = 0.3
N_REGIONS = 27
NODE_DIM = 1024


def fc_edge_weights(n, device="cpu"):
    """Stage 1's ``_fc_edge_weights``, transcribed verbatim.

    Fully connected graph with GCN normalisation applied by hand:
    A_tilde = A + I, D_tilde[i] = n+1, so A_hat[i,i] = 2/(n+1) and
    A_hat[i,j] = 1/(n+1). APPNP is called with normalize=False so these are
    used as given.
    """
    idx = torch.arange(n, device=device)
    row = idx.repeat_interleave(n)
    col = idx.repeat(n)
    edge_index = torch.stack([row, col], 0)
    w = torch.full((n * n,), 1.0 / (n + 1), device=device)
    w[row == col] = 2.0 / (n + 1)
    return edge_index, w


def dense_adjacency(n, device="cpu"):
    """The same A_hat as a dense (n, n) matrix."""
    a = torch.full((n, n), 1.0 / (n + 1), device=device)
    a.fill_diagonal_(2.0 / (n + 1))
    return a


def direct_propagation(h, adjacency, k=K, alpha=ALPHA):
    """The port's specified propagation, written out.

    APPNP: x_{t+1} = (1 - alpha) * A_hat @ x_t + alpha * x_0
    """
    initial = h
    for _ in range(k):
        h = (1.0 - alpha) * (adjacency @ h) + alpha * initial
    return h


def main() -> None:
    torch.manual_seed(1337)
    print(f"torch {torch.__version__}")
    import torch_geometric

    print(f"torch_geometric {torch_geometric.__version__}")
    print(f"settings: K={K} alpha={ALPHA} normalize=False add_self_loops=False\n")

    # **Measured in ULPs, not in absolute difference.** The first version of this
    # script compared a float32 result against a 1e-10 constant -- a threshold
    # only meaningful for float64 -- and declared a parity failure on agreement
    # that was exact to machine precision. Comparing an absolute difference
    # against a dtype-independent number is the same quantity confusion this
    # project keeps finding, this time in the check rather than the code.
    worst_ulps = 0.0
    for n in (N_REGIONS, 37, 5):
        for dim in (NODE_DIM, 16):
            for dtype in (torch.float32, torch.float64):
                h = torch.randn(n, dim, dtype=dtype)
                edge_index, weight = fc_edge_weights(n)
                weight = weight.to(dtype)
                adjacency = dense_adjacency(n).to(dtype)

                reference = APPNP(
                    K=K, alpha=ALPHA, add_self_loops=False, normalize=False
                )(h, edge_index, weight)
                ours = direct_propagation(h, adjacency)

                diff = (reference - ours).abs().max().item()
                scale = reference.abs().max().item()
                eps = torch.finfo(dtype).eps
                ulps = diff / (eps * scale) if scale else 0.0
                worst_ulps = max(worst_ulps, ulps)
                print(
                    f"  n={n:<3} dim={dim:<5} {str(dtype).split('.')[-1]:<8} "
                    f"max|diff|={diff:.3e}  {ulps:5.2f} ULP  "
                    f"{'OK' if ulps <= 8 else 'DIFFERS'}"
                )

    # K>1 too: the port fixes K=1, but if that ever changes the equivalence
    # must still hold, and a check that only covers the current value would not
    # say so.
    print("\n  K sweep (n=27, dim=64, float64):")
    for k in (1, 2, 5, 10):
        h = torch.randn(27, 64, dtype=torch.float64)
        edge_index, weight = fc_edge_weights(27)
        reference = APPNP(K=k, alpha=ALPHA, add_self_loops=False, normalize=False)(
            h, edge_index, weight.double()
        )
        ours = direct_propagation(h, dense_adjacency(27).double(), k=k)
        diff = (reference - ours).abs().max().item()
        scale = reference.abs().max().item()
        ulps = diff / (torch.finfo(torch.float64).eps * scale)
        worst_ulps = max(worst_ulps, ulps)
        print(f"    K={k:<3} max|diff|={diff:.3e}  {ulps:5.2f} ULP")

    print(f"\nworst agreement across all cases: {worst_ulps:.2f} ULP")
    print(
        "VERDICT:",
        "PARITY — the direct propagation IS torch_geometric's APPNP at these "
        "settings, to machine precision"
        if worst_ulps <= 8
        else "DIFFERS — do not use the direct form",
    )


if __name__ == "__main__":
    main()
