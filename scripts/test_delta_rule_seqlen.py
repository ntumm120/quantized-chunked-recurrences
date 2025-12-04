"""
Test script to plot delta rule implementation differences as a function of sequence length.
"""

import torch
import sys
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, '../3rdparty/flash-linear-attention')

from fla.ops.delta_rule.naive import (
    delta_rule_recurrence,
    delta_rule_chunkwise,
    delta_rule_parallel,
)


def test_at_seqlen(seq_len, device='cuda', d_k=64, d_v=64, scale=0.1):
    """Run all three implementations at a given sequence length and return differences."""
    torch.manual_seed(42)

    batch_size = 1
    num_heads = 1

    # Create random inputs
    q = torch.randn(batch_size, num_heads, seq_len, d_k, dtype=torch.float32, device=device) * scale
    k = torch.randn(batch_size, num_heads, seq_len, d_k, dtype=torch.float32, device=device) * scale
    v = torch.randn(batch_size, num_heads, seq_len, d_v, dtype=torch.float32, device=device) * scale
    beta = torch.sigmoid(torch.randn(batch_size, num_heads, seq_len, dtype=torch.float32, device=device))

    # Run all implementations
    o_recurrence, S_recurrence = delta_rule_recurrence(q, k, v, beta)
    o_chunkwise, S_chunkwise = delta_rule_chunkwise(q, k, v, beta, chunk_size=32)
    o_parallel, A_parallel = delta_rule_parallel(q, k, v, beta, BM=128, BN=32)

    # Compute differences
    results = {
        'rec_vs_chunk_max': (o_recurrence - o_chunkwise).abs().max().item(),
        'rec_vs_chunk_mean': (o_recurrence - o_chunkwise).abs().mean().item(),
        'rec_vs_par_max': (o_recurrence - o_parallel).abs().max().item(),
        'rec_vs_par_mean': (o_recurrence - o_parallel).abs().mean().item(),
        'chunk_vs_par_max': (o_chunkwise - o_parallel).abs().max().item(),
        'chunk_vs_par_mean': (o_chunkwise - o_parallel).abs().mean().item(),
        'state_max': (S_recurrence - S_chunkwise).abs().max().item(),
        'state_mean': (S_recurrence - S_chunkwise).abs().mean().item(),
    }

    return results


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    # Sequence lengths to test (must be divisible by 128 for parallel version)
    seq_lengths = [128, 256, 512, 1024, 2048, 4096, 8192]

    # Storage for results
    results_by_seqlen = {}

    for seq_len in seq_lengths:
        print(f"Testing seq_len={seq_len}...", end=" ", flush=True)
        results = test_at_seqlen(seq_len, device=device)
        results_by_seqlen[seq_len] = results
        print(f"max diff: {results['rec_vs_chunk_max']:.2e}")

    # Extract data for plotting
    seq_lens = list(results_by_seqlen.keys())
    rec_vs_chunk_max = [results_by_seqlen[s]['rec_vs_chunk_max'] for s in seq_lens]
    rec_vs_par_max = [results_by_seqlen[s]['rec_vs_par_max'] for s in seq_lens]
    chunk_vs_par_max = [results_by_seqlen[s]['chunk_vs_par_max'] for s in seq_lens]
    state_max = [results_by_seqlen[s]['state_max'] for s in seq_lens]

    rec_vs_chunk_mean = [results_by_seqlen[s]['rec_vs_chunk_mean'] for s in seq_lens]
    rec_vs_par_mean = [results_by_seqlen[s]['rec_vs_par_mean'] for s in seq_lens]
    chunk_vs_par_mean = [results_by_seqlen[s]['chunk_vs_par_mean'] for s in seq_lens]

    # Create plots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Max absolute differences
    ax1 = axes[0]
    ax1.plot(seq_lens, rec_vs_chunk_max, 'o-', label='Recurrence vs Chunkwise', linewidth=2, markersize=8)
    ax1.plot(seq_lens, rec_vs_par_max, 's-', label='Recurrence vs Parallel', linewidth=2, markersize=8)
    ax1.plot(seq_lens, chunk_vs_par_max, '^-', label='Chunkwise vs Parallel', linewidth=2, markersize=8)
    ax1.set_xlabel('Sequence Length', fontsize=12)
    ax1.set_ylabel('Max Absolute Difference', fontsize=12)
    ax1.set_title('Max Absolute Difference vs Sequence Length', fontsize=14)
    ax1.set_xscale('log', base=2)
    ax1.set_yscale('log')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(seq_lens)
    ax1.set_xticklabels([str(s) for s in seq_lens])

    # Plot 2: Mean absolute differences
    ax2 = axes[1]
    ax2.plot(seq_lens, rec_vs_chunk_mean, 'o-', label='Recurrence vs Chunkwise', linewidth=2, markersize=8)
    ax2.plot(seq_lens, rec_vs_par_mean, 's-', label='Recurrence vs Parallel', linewidth=2, markersize=8)
    ax2.plot(seq_lens, chunk_vs_par_mean, '^-', label='Chunkwise vs Parallel', linewidth=2, markersize=8)
    ax2.set_xlabel('Sequence Length', fontsize=12)
    ax2.set_ylabel('Mean Absolute Difference', fontsize=12)
    ax2.set_title('Mean Absolute Difference vs Sequence Length', fontsize=14)
    ax2.set_xscale('log', base=2)
    ax2.set_yscale('log')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(seq_lens)
    ax2.set_xticklabels([str(s) for s in seq_lens])

    plt.tight_layout()
    plt.savefig('../plots/delta_rule_equivalence_vs_seqlen.png', dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: ../plots/delta_rule_equivalence_vs_seqlen.png")

    # Also print a summary table
    print("\n" + "=" * 80)
    print("Summary Table")
    print("=" * 80)
    print(f"{'Seq Len':<10} {'Rec vs Chunk (max)':<20} {'Rec vs Par (max)':<20} {'Chunk vs Par (max)':<20}")
    print("-" * 80)
    for s in seq_lens:
        r = results_by_seqlen[s]
        print(f"{s:<10} {r['rec_vs_chunk_max']:<20.2e} {r['rec_vs_par_max']:<20.2e} {r['chunk_vs_par_max']:<20.2e}")

    plt.show()


if __name__ == "__main__":
    main()
