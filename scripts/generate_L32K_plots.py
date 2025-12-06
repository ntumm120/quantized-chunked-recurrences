"""
Generate all 3 plots at L=32K with block size subplots:
1. Perturbation sensitivity
2. Execution time
3. Peak memory

Uses the memory-efficient parallel function (no L×L materialization).
"""

import sys
sys.path.insert(0, '../3rdparty/flash-linear-attention')

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import gc
import time
from fla.ops.delta_rule.naive import (
    delta_rule_recurrence_native_dtype,
    delta_rule_chunkwise_native_dtype,
    delta_rule_parallel_no_materialize_native_dtype,
    delta_rule_chunkwise_parallel_scan_native_dtype,
)

# Configuration
L = 32768
B, H, D_K, D_V = 2, 4, 64, 64
BLOCK_SIZES = [32, 64, 128, 256, 512, 1024]
NOISE_SCALE = 1e-4
DTYPE = torch.bfloat16
DEVICE = 'cuda'

# Colors
COLORS = {
    'Recurrence': '#1f77b4',
    'Chunkwise': '#2ca02c',
    'Hybrid': '#d62728',
    'Parallel': '#ff7f0e',
}


def get_algorithms(block_size):
    """Get algorithm configs for a given block size."""
    return [
        ('Recurrence', delta_rule_recurrence_native_dtype, {}),
        ('Chunkwise', delta_rule_chunkwise_native_dtype, {'chunk_size': block_size}),
        ('Hybrid', delta_rule_chunkwise_parallel_scan_native_dtype, {'chunk_size': block_size}),
        ('Parallel', delta_rule_parallel_no_materialize_native_dtype, {'BM': block_size, 'BN': block_size}),
    ]


def create_inputs(B, H, L, D_K, D_V, dtype, device, seed=42):
    """Create properly scaled inputs."""
    torch.manual_seed(seed)
    q = torch.randn(B, H, L, D_K, device=device, dtype=dtype)
    k = torch.randn(B, H, L, D_K, device=device, dtype=dtype)
    v = torch.randn(B, H, L, D_V, device=device, dtype=dtype)
    beta = torch.sigmoid(torch.randn(B, H, L, device=device, dtype=dtype))
    k = F.normalize(k, dim=-1) * 0.1
    return q, k, v, beta


def measure_perturbation(algorithms, q, k, v, beta, noise_scale):
    """Measure perturbation sensitivity for each algorithm."""
    q_noise = torch.randn_like(q) * noise_scale
    k_noise = torch.randn_like(k) * noise_scale
    v_noise = torch.randn_like(v) * noise_scale

    q_pert = q + q_noise
    k_pert = k + k_noise
    v_pert = v + v_noise

    results = {}
    for name, fn, kwargs in algorithms:
        try:
            torch.cuda.empty_cache()
            o_orig = fn(q.clone(), k.clone(), v.clone(), beta.clone(), **kwargs)
            if isinstance(o_orig, tuple):
                o_orig = o_orig[0]
            o_pert = fn(q_pert.clone(), k_pert.clone(), v_pert.clone(), beta.clone(), **kwargs)
            if isinstance(o_pert, tuple):
                o_pert = o_pert[0]
            diff = (o_pert - o_orig).abs().mean(dim=(0, 1, 3))
            results[name] = diff.float().cpu().numpy()
        except Exception as e:
            print(f"  {name}: Error - {e}")
            results[name] = None
    return results


def measure_speed(algorithms, q, k, v, beta, warmup=2, repeats=5):
    """Measure execution time for each algorithm."""
    results = {}
    for name, fn, kwargs in algorithms:
        try:
            torch.cuda.empty_cache()
            # Warmup
            for _ in range(warmup):
                _ = fn(q.clone(), k.clone(), v.clone(), beta.clone(), **kwargs)
                torch.cuda.synchronize()

            # Time
            times = []
            for _ in range(repeats):
                torch.cuda.synchronize()
                start = time.perf_counter()
                _ = fn(q.clone(), k.clone(), v.clone(), beta.clone(), **kwargs)
                torch.cuda.synchronize()
                times.append((time.perf_counter() - start) * 1000)
            results[name] = np.mean(times)
        except Exception as e:
            print(f"  {name}: Error - {e}")
            results[name] = None
    return results


def measure_memory(algorithms, q, k, v, beta):
    """Measure peak memory for each algorithm."""
    results = {}
    for name, fn, kwargs in algorithms:
        try:
            torch.cuda.empty_cache()
            gc.collect()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

            _ = fn(q.clone(), k.clone(), v.clone(), beta.clone(), **kwargs)
            torch.cuda.synchronize()

            peak_mem = torch.cuda.max_memory_allocated() / (1024**2)
            results[name] = peak_mem
        except torch.cuda.OutOfMemoryError:
            results[name] = None
            print(f"  {name}: OOM")
        except Exception as e:
            print(f"  {name}: Error - {e}")
            results[name] = None
    return results


def ema_smooth(data, alpha=0.05):
    """Apply exponential moving average smoothing."""
    smoothed = np.zeros_like(data)
    smoothed[0] = data[0]
    for i in range(1, len(data)):
        smoothed[i] = alpha * data[i] + (1 - alpha) * smoothed[i-1]
    return smoothed


def plot_perturbation_subplots(all_results, save_path):
    """Plot perturbation sensitivity with subplots for each block size."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    positions = np.arange(L)

    for idx, block_size in enumerate(BLOCK_SIZES):
        ax = axes[idx]
        results = all_results[block_size]['perturbation']

        for name in ['Recurrence', 'Chunkwise', 'Hybrid', 'Parallel']:
            if name in results and results[name] is not None:
                smoothed = ema_smooth(results[name], alpha=0.01)
                ax.plot(positions, smoothed, label=name, color=COLORS[name],
                       linewidth=1.5, alpha=0.9)

        ax.set_xlabel('Sequence Position', fontsize=10)
        ax.set_ylabel('Mean Abs Deviation', fontsize=10)
        ax.set_title(f'Block Size = {block_size}', fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')

    plt.suptitle(f'Perturbation Sensitivity by Block Size (L={L}, bf16, noise={NOISE_SCALE:.0e})',
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_speed_subplots(all_results, save_path):
    """Plot execution time with subplots for each block size."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx, block_size in enumerate(BLOCK_SIZES):
        ax = axes[idx]
        results = all_results[block_size]['speed']

        names = []
        times = []
        colors = []
        for name in ['Recurrence', 'Chunkwise', 'Hybrid', 'Parallel']:
            if name in results and results[name] is not None:
                names.append(name)
                times.append(results[name])
                colors.append(COLORS[name])

        bars = ax.bar(names, times, color=colors)
        ax.set_ylabel('Time (ms)', fontsize=10)
        ax.set_title(f'Block Size = {block_size}', fontsize=12)
        ax.tick_params(axis='x', rotation=45)

        # Add value labels
        for bar, t in zip(bars, times):
            ax.annotate(f'{t:.0f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                       ha='center', va='bottom', fontsize=9)

    plt.suptitle(f'Execution Time by Block Size (L={L})', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_memory_subplots(all_results, save_path):
    """Plot peak memory with subplots for each block size."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx, block_size in enumerate(BLOCK_SIZES):
        ax = axes[idx]
        results = all_results[block_size]['memory']

        names = []
        mems = []
        colors = []
        for name in ['Recurrence', 'Chunkwise', 'Hybrid', 'Parallel']:
            if name in results and results[name] is not None:
                names.append(name)
                mems.append(results[name])
                colors.append(COLORS[name])

        bars = ax.bar(names, mems, color=colors)
        ax.set_ylabel('Memory (MB)', fontsize=10)
        ax.set_title(f'Block Size = {block_size}', fontsize=12)
        ax.tick_params(axis='x', rotation=45)

        # Add value labels
        for bar, m in zip(bars, mems):
            ax.annotate(f'{m:.0f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                       ha='center', va='bottom', fontsize=9)

    plt.suptitle(f'Peak Memory by Block Size (L={L})', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def main():
    import os
    os.makedirs('plots', exist_ok=True)

    print(f"Generating L={L} plots with memory-efficient Parallel")
    print("=" * 60)

    # Create inputs once
    print("Creating inputs...")
    q, k, v, beta = create_inputs(B, H, L, D_K, D_V, DTYPE, DEVICE)

    all_results = {}

    for block_size in BLOCK_SIZES:
        print(f"\nBlock size: {block_size}")
        print("-" * 40)

        algorithms = get_algorithms(block_size)
        all_results[block_size] = {}

        # Measure perturbation
        print("  Measuring perturbation sensitivity...")
        all_results[block_size]['perturbation'] = measure_perturbation(
            algorithms, q, k, v, beta, NOISE_SCALE
        )

        # Measure speed
        print("  Measuring speed...")
        all_results[block_size]['speed'] = measure_speed(
            algorithms, q, k, v, beta
        )
        for name, t in all_results[block_size]['speed'].items():
            if t is not None:
                print(f"    {name}: {t:.0f} ms")

        # Measure memory
        print("  Measuring memory...")
        all_results[block_size]['memory'] = measure_memory(
            algorithms, q, k, v, beta
        )
        for name, m in all_results[block_size]['memory'].items():
            if m is not None:
                print(f"    {name}: {m:.0f} MB")

        torch.cuda.empty_cache()
        gc.collect()

    # Generate plots
    print("\n" + "=" * 60)
    print("Generating plots...")

    plot_perturbation_subplots(all_results, 'plots/sensitivity_by_blocksize_L32K.png')
    plot_speed_subplots(all_results, 'plots/speed_by_blocksize_L32K.png')
    plot_memory_subplots(all_results, 'plots/memory_by_blocksize_L32K.png')

    print("\nDone!")


if __name__ == "__main__":
    main()
