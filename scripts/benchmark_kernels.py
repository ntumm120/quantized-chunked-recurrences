"""
Benchmark delta rule kernel implementations for speed and memory usage.

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
import matplotlib.pyplot as plt
import numpy as np
import gc
from fla.ops.delta_rule.naive import (
    delta_rule_recurrence,
    delta_rule_chunkwise,
    delta_rule_parallel,
    delta_rule_parallel_scan,
)


ALGORITHMS = [
    ('Recurrence', delta_rule_recurrence, {}),
    ('Chunkwise', delta_rule_chunkwise, {'chunk_size': 32}),
    ('Parallel', delta_rule_parallel, {'BM': 128, 'BN': 32}),
    ('Parallel Scan', delta_rule_parallel_scan, {}),
]


def measure_memory_and_time(fn, q, k, v, beta, kwargs, warmup=2, repeats=5):
    """Measure peak memory and average time for a kernel."""
    device = q.device

    # Warmup
    for _ in range(warmup):
        try:
            _ = fn(q, k, v, beta, **kwargs)
            torch.cuda.synchronize()
        except torch.cuda.OutOfMemoryError:
            return None, None

    # Clear cache and reset memory stats
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    times = []
    peak_mem = 0

    for _ in range(repeats):
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        try:
            _ = fn(q, k, v, beta, **kwargs)
        except torch.cuda.OutOfMemoryError:
            return None, None
        end.record()

        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
        peak_mem = max(peak_mem, torch.cuda.max_memory_allocated(device))

    avg_time = np.mean(times)
    return avg_time, peak_mem


def run_benchmark(
    seq_lengths=[256, 512, 1024, 2048, 4096, 8192, 16384, 32768],
    B=2, H=4, D_K=64, D_V=64,
    dtype=torch.float32,
    device='cuda',
):
    """Run benchmark across sequence lengths."""
    results = {name: {'lengths': [], 'times': [], 'memory': []}
               for name, _, _ in ALGORITHMS}

    for L in seq_lengths:
        print(f"\nSequence length: {L}")
        print("-" * 50)

        # Generate inputs
        torch.manual_seed(42)
        q = torch.randn(B, H, L, D_K, device=device, dtype=dtype)
        k = torch.randn(B, H, L, D_K, device=device, dtype=dtype)
        v = torch.randn(B, H, L, D_V, device=device, dtype=dtype)
        beta = torch.sigmoid(torch.randn(B, H, L, device=device, dtype=dtype))
        k = F.normalize(k, dim=-1) * 0.1

        for name, fn, kwargs in ALGORITHMS:
            torch.cuda.empty_cache()
            gc.collect()

            time_ms, mem_bytes = measure_memory_and_time(fn, q, k, v, beta, kwargs)

            if time_ms is not None:
                mem_mb = mem_bytes / (1024 ** 2)
                results[name]['lengths'].append(L)
                results[name]['times'].append(time_ms)
                results[name]['memory'].append(mem_mb)
                print(f"  {name:<15}: {time_ms:>8.2f} ms, {mem_mb:>8.1f} MB")
            else:
                print(f"  {name:<15}: OOM")

        # Clear after each length
        del q, k, v, beta
        torch.cuda.empty_cache()
        gc.collect()

    return results


def plot_results(results, save_path='plots/kernel_benchmark.png'):
    """Plot speed and memory vs sequence length."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = {'Recurrence': '#1f77b4', 'Chunkwise': '#ff7f0e',
              'Parallel': '#2ca02c', 'Parallel Scan': '#d62728'}
    markers = {'Recurrence': 'o', 'Chunkwise': 's',
               'Parallel': '^', 'Parallel Scan': 'd'}

    # Time plot
    ax = axes[0]
    for name, data in results.items():
        if data['lengths']:
            ax.plot(data['lengths'], data['times'],
                   label=name, color=colors[name], marker=markers[name],
                   linewidth=2, markersize=8)
    ax.set_xlabel('Sequence Length', fontsize=12)
    ax.set_ylabel('Time (ms)', fontsize=12)
    ax.set_title('Kernel Execution Time', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')

    # Memory plot
    ax = axes[1]
    for name, data in results.items():
        if data['lengths']:
            ax.plot(data['lengths'], data['memory'],
                   label=name, color=colors[name], marker=markers[name],
                   linewidth=2, markersize=8)
    ax.set_xlabel('Sequence Length', fontsize=12)
    ax.set_ylabel('Peak Memory (MB)', fontsize=12)
    ax.set_title('Peak GPU Memory Usage', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')

    plt.suptitle('Delta Rule Kernel Benchmark (B=2, H=4, D=64)', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved plot to {save_path}")


def load_cached_results(filepath='benchmark_results.pkl'):
    """Load cached benchmark results and convert to new format."""
    import pickle
    try:
        with open(filepath, 'rb') as f:
            old_data = pickle.load(f)

        # Convert old format to new format
        name_map = {
            'recurrence': 'Recurrence',
            'chunkwise': 'Chunkwise',
            'parallel': 'Parallel',
            'parallel_scan': 'Parallel Scan',
        }

        results = {}
        for old_name, new_name in name_map.items():
            times = old_data['results'][old_name]
            lengths = old_data['lengths']

            # Filter out None values
            valid_data = [(l, t) for l, t in zip(lengths, times) if t is not None]
            if valid_data:
                valid_lengths, valid_times = zip(*valid_data)
                results[new_name] = {
                    'lengths': list(valid_lengths),
                    'times': list(valid_times),
                    'memory': [],  # No memory data in old format
                }
            else:
                results[new_name] = {'lengths': [], 'times': [], 'memory': []}

        return results
    except FileNotFoundError:
        return None


def plot_time_only(results, save_path='plots/kernel_benchmark_time.png'):
    """Plot speed vs sequence length (when memory data unavailable)."""
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {'Recurrence': '#1f77b4', 'Chunkwise': '#ff7f0e',
              'Parallel': '#2ca02c', 'Parallel Scan': '#d62728'}
    markers = {'Recurrence': 'o', 'Chunkwise': 's',
               'Parallel': '^', 'Parallel Scan': 'd'}

    for name, data in results.items():
        if data['lengths']:
            ax.plot(data['lengths'], data['times'],
                   label=name, color=colors[name], marker=markers[name],
                   linewidth=2, markersize=8)

    ax.set_xlabel('Sequence Length', fontsize=12)
    ax.set_ylabel('Time (ms)', fontsize=12)
    ax.set_title('Delta Rule Kernel Execution Time (B=2, H=4, D=64)', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved plot to {save_path}")


def main():
    import os
    os.makedirs('plots', exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running on {device}")
    print("=" * 60)

    if device == 'cuda':
        # Run benchmark with memory tracking
        results = run_benchmark(
            seq_lengths=[256, 512, 1024, 2048, 4096, 8192, 16384, 32768],
            B=2, H=4, D_K=64, D_V=64,
            dtype=torch.float32,
            device=device,
        )

        # Print summary table
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)

        all_lengths = set()
        for data in results.values():
            all_lengths.update(data['lengths'])

        for L in sorted(all_lengths):
            print(f"\nL={L}:")
            for name, data in results.items():
                if L in data['lengths']:
                    idx = data['lengths'].index(L)
                    print(f"  {name:<15}: {data['times'][idx]:>8.2f} ms, {data['memory'][idx]:>8.1f} MB")

        # Plot with memory
        plot_results(results)

        # Save results
        import pickle
        with open('benchmark_results_full.pkl', 'wb') as f:
            pickle.dump(results, f)
        print("Saved results to benchmark_results_full.pkl")
    else:
        # Try to load cached results
        print("CUDA not available. Loading cached results...")
        results = load_cached_results()

        if results:
            print("Loaded cached benchmark results.")

            # Print summary
            print("\n" + "=" * 60)
            print("CACHED RESULTS (time only)")
            print("=" * 60)

            for name, data in results.items():
                if data['lengths']:
                    print(f"\n{name}:")
                    for l, t in zip(data['lengths'], data['times']):
                        print(f"  L={l:<5}: {t:>8.2f} ms")

            # Plot time only (no memory data)
            plot_time_only(results)
        else:
            print("No cached results found. Run on GPU to generate benchmarks.")


if __name__ == "__main__":
    main()
