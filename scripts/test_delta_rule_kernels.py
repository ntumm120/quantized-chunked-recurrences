"""
Correctness test for all delta rule kernel implementations.

Compares:
1. delta_rule_recurrence (sequential)
2. delta_rule_chunkwise
3. delta_rule_parallel
4. delta_rule_parallel_scan (Hillis-Steele)
"""

import sys
sys.path.insert(0, '../3rdparty/flash-linear-attention')

import torch
import torch.nn.functional as F
from fla.ops.delta_rule.naive import (
    delta_rule_recurrence,
    delta_rule_chunkwise,
    delta_rule_parallel,
    delta_rule_parallel_scan,
    delta_rule_recurrence_native_dtype,
    delta_rule_parallel_scan_native_dtype,
)


def test_correctness(B=2, H=4, L=128, D_K=64, D_V=64, device='cuda', dtype=torch.float32):
    """Test all implementations against sequential recurrence."""
    torch.manual_seed(42)

    q = torch.randn(B, H, L, D_K, device=device, dtype=dtype)
    k = torch.randn(B, H, L, D_K, device=device, dtype=dtype)
    v = torch.randn(B, H, L, D_V, device=device, dtype=dtype)
    beta = torch.sigmoid(torch.randn(B, H, L, device=device, dtype=dtype))

    # Normalize k
    k = F.normalize(k, dim=-1) * 0.1

    print(f"Testing on {device} with {dtype}")
    print(f"Shape: B={B}, H={H}, L={L}, D_K={D_K}, D_V={D_V}")
    print("=" * 60)

    # Ground truth: sequential recurrence
    o_rec, S_rec = delta_rule_recurrence(q, k, v, beta)

    results = []

    # Chunkwise
    try:
        o_chunk, S_chunk = delta_rule_chunkwise(q, k, v, beta, chunk_size=32)
        diff = (o_rec - o_chunk).abs()
        results.append(("Chunkwise", diff.max().item(), diff.mean().item(), "✓"))
    except Exception as e:
        results.append(("Chunkwise", None, None, f"✗ {e}"))

    # Parallel
    try:
        o_par, _ = delta_rule_parallel(q, k, v, beta, BM=128, BN=32)
        diff = (o_rec - o_par).abs()
        results.append(("Parallel", diff.max().item(), diff.mean().item(), "✓"))
    except Exception as e:
        results.append(("Parallel", None, None, f"✗ {e}"))

    # Parallel scan
    try:
        o_scan, S_scan = delta_rule_parallel_scan(q, k, v, beta)
        diff = (o_rec - o_scan).abs()
        results.append(("Parallel Scan", diff.max().item(), diff.mean().item(), "✓"))
    except Exception as e:
        results.append(("Parallel Scan", None, None, f"✗ {e}"))

    # Print results
    print(f"{'Algorithm':<20} {'Max Diff':<14} {'Mean Diff':<14} {'Status'}")
    print("-" * 60)
    for name, max_diff, mean_diff, status in results:
        if max_diff is not None:
            print(f"{name:<20} {max_diff:<14.6e} {mean_diff:<14.6e} {status}")
        else:
            print(f"{name:<20} {'N/A':<14} {'N/A':<14} {status}")

    return results


def test_perturbation_sensitivity(B=2, H=4, L=512, D_K=64, D_V=64,
                                   noise_scale=1e-4, device='cuda', dtype=torch.bfloat16):
    """
    Compare perturbation sensitivity between sequential and parallel scan.
    Uses native dtype implementations to see precision effects.
    """
    torch.manual_seed(42)

    q = torch.randn(B, H, L, D_K, device=device, dtype=dtype)
    k = torch.randn(B, H, L, D_K, device=device, dtype=dtype)
    v = torch.randn(B, H, L, D_V, device=device, dtype=dtype)
    beta = torch.sigmoid(torch.randn(B, H, L, device=device, dtype=dtype))

    k = F.normalize(k, dim=-1) * 0.1

    # Create perturbations
    q_pert = q + torch.randn_like(q) * noise_scale
    k_pert = k + torch.randn_like(k) * noise_scale
    v_pert = v + torch.randn_like(v) * noise_scale

    # Run original (using native dtype versions)
    o_rec_orig, _ = delta_rule_recurrence_native_dtype(q, k, v, beta)
    o_scan_orig, _ = delta_rule_parallel_scan_native_dtype(q, k, v, beta)

    # Run perturbed
    o_rec_pert, _ = delta_rule_recurrence_native_dtype(q_pert, k_pert, v_pert, beta)
    o_scan_pert, _ = delta_rule_parallel_scan_native_dtype(q_pert, k_pert, v_pert, beta)

    # Compute sensitivities
    rec_sensitivity = (o_rec_pert - o_rec_orig).abs().mean(dim=(0, 1, 3))  # (L,)
    scan_sensitivity = (o_scan_pert - o_scan_orig).abs().mean(dim=(0, 1, 3))  # (L,)

    # Algorithmic difference
    algo_diff = (o_rec_orig - o_scan_orig).abs().mean(dim=(0, 1, 3))

    print(f"\nPerturbation sensitivity test ({dtype}, noise={noise_scale:.0e})")
    print("=" * 60)
    print(f"Sequential mean sensitivity: {rec_sensitivity.float().mean():.6e}")
    print(f"Parallel scan mean sensitivity: {scan_sensitivity.float().mean():.6e}")
    print(f"Ratio (seq/scan): {rec_sensitivity.float().mean() / scan_sensitivity.float().mean():.4f}")
    print(f"Algorithmic difference: {algo_diff.float().mean():.6e}")

    return rec_sensitivity, scan_sensitivity, algo_diff


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Correctness tests
    print("\n" + "=" * 60)
    print("CORRECTNESS TESTS")
    print("=" * 60)
    test_correctness(device=device, dtype=torch.float32)

    # Perturbation tests
    print("\n" + "=" * 60)
    print("PERTURBATION SENSITIVITY TESTS")
    print("=" * 60)

    print("\n--- FP32 ---")
    test_perturbation_sensitivity(device=device, dtype=torch.float32)

    print("\n--- BF16 ---")
    test_perturbation_sensitivity(device=device, dtype=torch.bfloat16)
