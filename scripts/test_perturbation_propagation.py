"""
Compare how input perturbations propagate through all delta rule algorithms.

Compares perturbation sensitivity for:
1. Recurrence (sequential)
2. Chunkwise
3. Parallel
4. Parallel Scan (Hillis-Steele)
"""

import sys
sys.path.insert(0, '../3rdparty/flash-linear-attention')

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from fla.ops.delta_rule.naive import (
    delta_rule_recurrence_native_dtype,
    delta_rule_chunkwise_native_dtype,
    delta_rule_parallel_native_dtype,
    delta_rule_parallel_scan_native_dtype,
)


ALGORITHMS = [
    ('Recurrence', delta_rule_recurrence_native_dtype, {}),
    ('Chunkwise', delta_rule_chunkwise_native_dtype, {'chunk_size': 32}),
    ('Parallel', delta_rule_parallel_native_dtype, {'BM': 128, 'BN': 32}),
    ('Parallel Scan', delta_rule_parallel_scan_native_dtype, {}),
]


def run_perturbation_experiment(
    B=2, H=4, L=512, D_K=64, D_V=64,
    noise_scale=1e-4,
    dtype=torch.float32,
    device='cuda',
    seed=42,
):
    """
    Run perturbation experiment comparing all algorithms.

    Returns:
        dict mapping algorithm name to (L,) deviation array
    """
    torch.manual_seed(seed)

    q = torch.randn(B, H, L, D_K, device=device, dtype=dtype)
    k = torch.randn(B, H, L, D_K, device=device, dtype=dtype)
    v = torch.randn(B, H, L, D_V, device=device, dtype=dtype)
    beta = torch.sigmoid(torch.randn(B, H, L, device=device, dtype=dtype))

    # Normalize k
    k = F.normalize(k, dim=-1) * 0.1

    # Create perturbations
    q_noise = torch.randn_like(q) * noise_scale
    k_noise = torch.randn_like(k) * noise_scale
    v_noise = torch.randn_like(v) * noise_scale

    q_pert = q + q_noise
    k_pert = k + k_noise
    v_pert = v + v_noise

    results = {}

    for name, fn, kwargs in ALGORITHMS:
        try:
            # Run original
            o_orig, _ = fn(q.clone(), k.clone(), v.clone(), beta.clone(), **kwargs)
            # Run perturbed
            o_pert, _ = fn(q_pert.clone(), k_pert.clone(), v_pert.clone(), beta.clone(), **kwargs)
            # Compute deviation at each position
            diff = (o_pert - o_orig).abs().mean(dim=(0, 1, 3))  # (L,)
            results[name] = diff.float().cpu().numpy()
        except Exception as e:
            print(f"Error running {name}: {e}")
            results[name] = None

    return results, L, noise_scale


def plot_results(results, L, noise_scale, dtype_name, save_path):
    """Plot perturbation sensitivity for all algorithms."""
    fig, ax = plt.subplots(figsize=(10, 6))

    positions = np.arange(L)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    for i, (name, diff) in enumerate(results.items()):
        if diff is not None:
            ax.plot(positions, diff, label=name, color=colors[i], alpha=0.8, linewidth=1.5)

    ax.set_xlabel('Sequence Position', fontsize=12)
    ax.set_ylabel('Mean Absolute Deviation', fontsize=12)
    ax.set_title(f'Input Perturbation Sensitivity ({dtype_name}, noise={noise_scale:.0e})', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved plot to {save_path}")


def main():
    import os
    os.makedirs('plots', exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Check for dtype argument
    if len(sys.argv) > 1 and sys.argv[1] == 'bf16':
        dtype = torch.bfloat16
        dtype_name = 'bf16'
    elif len(sys.argv) > 1 and sys.argv[1] == 'fp16':
        dtype = torch.float16
        dtype_name = 'fp16'
    else:
        dtype = torch.float32
        dtype_name = 'fp32'

    print(f"Running on {device} with {dtype_name}")
    print("=" * 60)

    # Run experiment
    results, L, noise_scale = run_perturbation_experiment(
        B=2, H=4, L=512, D_K=64, D_V=64,
        noise_scale=1e-4,
        dtype=dtype,
        device=device,
    )

    # Print summary
    print(f"\nMean sensitivity (noise={noise_scale:.0e}):")
    print("-" * 40)
    for name, diff in results.items():
        if diff is not None:
            print(f"  {name:<15}: {diff.mean():.6e}")

    # Compute ratios relative to parallel scan
    if results.get('Parallel Scan') is not None:
        par_scan_mean = results['Parallel Scan'].mean()
        print(f"\nRatio relative to Parallel Scan:")
        print("-" * 40)
        for name, diff in results.items():
            if diff is not None:
                ratio = diff.mean() / par_scan_mean
                print(f"  {name:<15}: {ratio:.4f}")

    # Plot
    plot_results(results, L, noise_scale, dtype_name,
                 save_path=f'plots/perturbation_sensitivity_{dtype_name}.png')


if __name__ == "__main__":
    main()
