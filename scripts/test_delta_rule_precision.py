"""
Test script to plot delta rule implementation differences across precisions and sequence lengths.
Uses multiple GPUs for parallel testing.
"""

import torch
import torch.multiprocessing as mp
import sys
import matplotlib.pyplot as plt

from delta_rule_naive_multiprec import (
    delta_rule_recurrence,
    delta_rule_chunkwise,
    delta_rule_parallel,
)


def test_at_seqlen(seq_len, device='cuda:0', d_k=64, d_v=64, scale=0.1, dtype=torch.float32):
    """Run all three implementations at a given sequence length and return differences."""
    torch.manual_seed(42)

    batch_size = 1
    num_heads = 1

    # Create random inputs in float32 first, then cast to target dtype
    q = (torch.randn(batch_size, num_heads, seq_len, d_k, device=device) * scale).to(dtype)
    k = (torch.randn(batch_size, num_heads, seq_len, d_k, device=device) * scale).to(dtype)
    v = (torch.randn(batch_size, num_heads, seq_len, d_v, device=device) * scale).to(dtype)
    beta = torch.sigmoid(torch.randn(batch_size, num_heads, seq_len, device=device)).to(dtype)

    # Run all implementations in the target dtype
    o_recurrence, _ = delta_rule_recurrence(q, k, v, beta)
    o_chunkwise, _ = delta_rule_chunkwise(q, k, v, beta, chunk_size=32)
    o_parallel, _ = delta_rule_parallel(q, k, v, beta, BM=128, BN=32)

    # Cast to float32 for accurate difference computation
    o_recurrence = o_recurrence.float()
    o_chunkwise = o_chunkwise.float()
    o_parallel = o_parallel.float()

    results = {
        'rec_vs_chunk_max': (o_recurrence - o_chunkwise).abs().max().item(),
        'rec_vs_chunk_mean': (o_recurrence - o_chunkwise).abs().mean().item(),
        'rec_vs_par_max': (o_recurrence - o_parallel).abs().max().item(),
        'rec_vs_par_mean': (o_recurrence - o_parallel).abs().mean().item(),
        'chunk_vs_par_max': (o_chunkwise - o_parallel).abs().max().item(),
        'chunk_vs_par_mean': (o_chunkwise - o_parallel).abs().mean().item(),
    }

    return results


def run_tests_on_gpu(gpu_id, tasks, results_dict):
    """Run a set of test tasks on a specific GPU."""
    device = f'cuda:{gpu_id}'
    torch.cuda.set_device(device)

    for task in tasks:
        seq_len, dtype, dtype_name = task
        result = test_at_seqlen(seq_len, device=device, dtype=dtype)
        results_dict[(dtype_name, seq_len)] = result
        print(f"  GPU {gpu_id}: {dtype_name} seq_len={seq_len} done, max_diff={result['rec_vs_chunk_max']:.2e}")


def main():
    num_gpus = torch.cuda.device_count()
    print(f"Found {num_gpus} GPUs:")
    for i in range(num_gpus):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    print()

    # Sequence lengths to test
    seq_lengths = [128, 256, 512, 1024, 2048, 4096, 8192]

    # Precisions to test
    precisions = {
        'float32': torch.float32,
        'float16': torch.float16,
        'bfloat16': torch.bfloat16,
    }

    # Create all tasks
    all_tasks = []
    for dtype_name, dtype in precisions.items():
        for seq_len in seq_lengths:
            all_tasks.append((seq_len, dtype, dtype_name))

    # Distribute tasks across GPUs
    tasks_per_gpu = [[] for _ in range(num_gpus)]
    for i, task in enumerate(all_tasks):
        tasks_per_gpu[i % num_gpus].append(task)

    print(f"Running {len(all_tasks)} tests across {num_gpus} GPUs...")
    print()

    # Use a manager dict for sharing results
    manager = mp.Manager()
    results_dict = manager.dict()

    # Run tests in parallel across GPUs
    processes = []
    for gpu_id in range(num_gpus):
        p = mp.Process(target=run_tests_on_gpu, args=(gpu_id, tasks_per_gpu[gpu_id], results_dict))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    # Convert manager dict to regular dict
    results_dict = dict(results_dict)

    # Organize results by precision
    all_results = {dtype_name: {} for dtype_name in precisions.keys()}
    for (dtype_name, seq_len), result in results_dict.items():
        all_results[dtype_name][seq_len] = result

    # Print results
    for prec_name in precisions.keys():
        print(f"\n{'='*60}")
        print(f"{prec_name}")
        print('='*60)
        for seq_len in seq_lengths:
            r = all_results[prec_name][seq_len]
            print(f"  seq_len={seq_len}: max diff = {r['rec_vs_chunk_max']:.2e}")

    # Create plots
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))

    colors = {'rec_vs_chunk': 'tab:blue', 'rec_vs_par': 'tab:orange', 'chunk_vs_par': 'tab:green'}
    markers = {'rec_vs_chunk': 'o', 'rec_vs_par': 's', 'chunk_vs_par': '^'}
    labels = {'rec_vs_chunk': 'Recurrence vs Chunkwise', 'rec_vs_par': 'Recurrence vs Parallel', 'chunk_vs_par': 'Chunkwise vs Parallel'}

    for row, prec_name in enumerate(precisions.keys()):
        results_by_seqlen = all_results[prec_name]
        seq_lens = sorted(results_by_seqlen.keys())

        data = {
            'rec_vs_chunk_max': [results_by_seqlen[s]['rec_vs_chunk_max'] for s in seq_lens],
            'rec_vs_par_max': [results_by_seqlen[s]['rec_vs_par_max'] for s in seq_lens],
            'chunk_vs_par_max': [results_by_seqlen[s]['chunk_vs_par_max'] for s in seq_lens],
            'rec_vs_chunk_mean': [results_by_seqlen[s]['rec_vs_chunk_mean'] for s in seq_lens],
            'rec_vs_par_mean': [results_by_seqlen[s]['rec_vs_par_mean'] for s in seq_lens],
            'chunk_vs_par_mean': [results_by_seqlen[s]['chunk_vs_par_mean'] for s in seq_lens],
        }

        # Plot max differences
        ax1 = axes[row, 0]
        for key in ['rec_vs_chunk', 'rec_vs_par', 'chunk_vs_par']:
            ax1.plot(seq_lens, data[f'{key}_max'], f'{markers[key]}-',
                    color=colors[key], label=labels[key], linewidth=2, markersize=8)
        ax1.set_xlabel('Sequence Length', fontsize=11)
        ax1.set_ylabel('Max Absolute Difference', fontsize=11)
        ax1.set_title(f'{prec_name.upper()} - Max Absolute Difference', fontsize=12, fontweight='bold')
        ax1.set_xscale('log', base=2)
        ax1.set_yscale('log')
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3)
        ax1.set_xticks(seq_lens)
        ax1.set_xticklabels([str(s) for s in seq_lens])

        # Plot mean differences
        ax2 = axes[row, 1]
        for key in ['rec_vs_chunk', 'rec_vs_par', 'chunk_vs_par']:
            ax2.plot(seq_lens, data[f'{key}_mean'], f'{markers[key]}-',
                    color=colors[key], label=labels[key], linewidth=2, markersize=8)
        ax2.set_xlabel('Sequence Length', fontsize=11)
        ax2.set_ylabel('Mean Absolute Difference', fontsize=11)
        ax2.set_title(f'{prec_name.upper()} - Mean Absolute Difference', fontsize=12, fontweight='bold')
        ax2.set_xscale('log', base=2)
        ax2.set_yscale('log')
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)
        ax2.set_xticks(seq_lens)
        ax2.set_xticklabels([str(s) for s in seq_lens])

    plt.tight_layout()
    plt.savefig('../plots/delta_rule_precision_comparison.png', dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: ../plots/delta_rule_precision_comparison.png")

    # Summary table
    print("\n" + "=" * 100)
    print("Summary: Max Absolute Difference (Recurrence vs Chunkwise) by Precision")
    print("=" * 100)
    header = f"{'Seq Len':<10}"
    for prec_name in precisions.keys():
        header += f"{prec_name:<20}"
    print(header)
    print("-" * 100)
    for seq_len in seq_lengths:
        row_str = f"{seq_len:<10}"
        for prec_name in precisions.keys():
            val = all_results[prec_name][seq_len]['rec_vs_chunk_max']
            row_str += f"{val:<20.2e}"
        print(row_str)


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()
