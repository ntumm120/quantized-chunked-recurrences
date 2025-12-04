"""
Test script to measure sensitivity of delta rule implementations to input perturbations.

For each algorithm, we:
1. Compute base output with clean inputs
2. Add Gaussian noise (mean=0, std=sigma) to q, k, v
3. Compute perturbed output
4. Measure how much the output changed

This reveals which algorithms are more sensitive to input noise.
"""

import torch
import torch.multiprocessing as mp
import sys
import matplotlib.pyplot as plt
import numpy as np

from delta_rule_naive_multiprec import (
    delta_rule_recurrence,
    delta_rule_chunkwise,
    delta_rule_parallel,
)


def test_perturbation(sigma, device='cuda:0', seq_len=1024, d_k=64, d_v=64,
                      scale=0.1, dtype=torch.float32, num_trials=5, noise_type='gaussian'):
    """
    Test how much each algorithm's output changes when inputs are perturbed.

    Args:
        sigma: standard deviation (gaussian) or half-width (uniform) of noise
        num_trials: number of random perturbations to average over
        noise_type: 'gaussian' or 'uniform'

    Returns:
        dict with mean and max output differences for each algorithm
    """
    torch.manual_seed(42)

    batch_size = 1
    num_heads = 1

    # Create base inputs
    q_base = (torch.randn(batch_size, num_heads, seq_len, d_k, device=device) * scale).to(dtype)
    k_base = (torch.randn(batch_size, num_heads, seq_len, d_k, device=device) * scale).to(dtype)
    v_base = (torch.randn(batch_size, num_heads, seq_len, d_v, device=device) * scale).to(dtype)
    beta = torch.sigmoid(torch.randn(batch_size, num_heads, seq_len, device=device)).to(dtype)

    # Compute base outputs (no perturbation)
    o_rec_base, _ = delta_rule_recurrence(q_base, k_base, v_base, beta)
    o_chunk_base, _ = delta_rule_chunkwise(q_base, k_base, v_base, beta, chunk_size=32)
    o_par_base, _ = delta_rule_parallel(q_base, k_base, v_base, beta, BM=128, BN=32)

    # Storage for differences across trials
    rec_diffs = []
    chunk_diffs = []
    par_diffs = []

    for trial in range(num_trials):
        # Generate perturbations
        torch.manual_seed(42 + trial + 1000)  # Different seed for each trial

        if noise_type == 'gaussian':
            noise_q = torch.randn_like(q_base) * sigma
            noise_k = torch.randn_like(k_base) * sigma
            noise_v = torch.randn_like(v_base) * sigma
        elif noise_type == 'uniform':
            # Uniform[-sigma, sigma] has std = sigma * sqrt(1/3) ≈ 0.577 * sigma
            # To match variance with gaussian, we'd use sigma * sqrt(3), but let's keep it simple
            noise_q = (torch.rand_like(q_base) * 2 - 1) * sigma
            noise_k = (torch.rand_like(k_base) * 2 - 1) * sigma
            noise_v = (torch.rand_like(v_base) * 2 - 1) * sigma
        else:
            raise ValueError(f"Unknown noise type: {noise_type}")

        # Perturbed inputs
        q_pert = q_base + noise_q
        k_pert = k_base + noise_k
        v_pert = v_base + noise_v

        # Compute perturbed outputs
        o_rec_pert, _ = delta_rule_recurrence(q_pert, k_pert, v_pert, beta)
        o_chunk_pert, _ = delta_rule_chunkwise(q_pert, k_pert, v_pert, beta, chunk_size=32)
        o_par_pert, _ = delta_rule_parallel(q_pert, k_pert, v_pert, beta, BM=128, BN=32)

        # Compute differences (in float32 for accuracy)
        rec_diffs.append((o_rec_pert.float() - o_rec_base.float()).abs())
        chunk_diffs.append((o_chunk_pert.float() - o_chunk_base.float()).abs())
        par_diffs.append((o_par_pert.float() - o_par_base.float()).abs())

    # Average over trials
    rec_diff = torch.stack(rec_diffs).mean(dim=0)
    chunk_diff = torch.stack(chunk_diffs).mean(dim=0)
    par_diff = torch.stack(par_diffs).mean(dim=0)

    results = {
        'recurrence_max': rec_diff.max().item(),
        'recurrence_mean': rec_diff.mean().item(),
        'chunkwise_max': chunk_diff.max().item(),
        'chunkwise_mean': chunk_diff.mean().item(),
        'parallel_max': par_diff.max().item(),
        'parallel_mean': par_diff.mean().item(),
    }

    return results


def run_tests_on_gpu(gpu_id, tasks, results_dict):
    """Run a set of test tasks on a specific GPU."""
    device = f'cuda:{gpu_id}'
    torch.cuda.set_device(device)

    for task in tasks:
        sigma, dtype, dtype_name, noise_type = task
        result = test_perturbation(sigma, device=device, dtype=dtype, noise_type=noise_type)
        results_dict[(noise_type, dtype_name, sigma)] = result
        print(f"  GPU {gpu_id}: {noise_type} {dtype_name} sigma={sigma:.1e} done, "
              f"rec={result['recurrence_mean']:.2e}, "
              f"chunk={result['chunkwise_mean']:.2e}, "
              f"par={result['parallel_mean']:.2e}")


def main():
    num_gpus = torch.cuda.device_count()
    print(f"Found {num_gpus} GPUs:")
    for i in range(num_gpus):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    print()

    # Sigma values to test (log scale)
    sigmas = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]

    # Precisions to test
    precisions = {
        'float32': torch.float32,
        'float16': torch.float16,
        'bfloat16': torch.bfloat16,
    }

    # Noise types to test
    noise_types = ['gaussian', 'uniform']

    # Create all tasks
    all_tasks = []
    for noise_type in noise_types:
        for dtype_name, dtype in precisions.items():
            for sigma in sigmas:
                all_tasks.append((sigma, dtype, dtype_name, noise_type))

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

    # Organize results by noise_type -> precision -> sigma
    all_results = {noise_type: {dtype_name: {} for dtype_name in precisions.keys()}
                   for noise_type in noise_types}
    for (noise_type, dtype_name, sigma), result in results_dict.items():
        all_results[noise_type][dtype_name][sigma] = result

    colors = {'recurrence': 'tab:blue', 'chunkwise': 'tab:orange', 'parallel': 'tab:green'}
    markers = {'recurrence': 'o', 'chunkwise': 's', 'parallel': '^'}
    labels = {'recurrence': 'Recurrence', 'chunkwise': 'Chunkwise', 'parallel': 'Parallel'}

    # Create plots for each noise type
    for noise_type in noise_types:
        fig, axes = plt.subplots(3, 2, figsize=(14, 12))
        fig.suptitle(f'{noise_type.upper()} Noise Perturbation', fontsize=14, fontweight='bold', y=1.02)

        for row, prec_name in enumerate(precisions.keys()):
            results_by_sigma = all_results[noise_type][prec_name]
            sigma_vals = sorted(results_by_sigma.keys())

            data = {
                'recurrence_max': [results_by_sigma[s]['recurrence_max'] for s in sigma_vals],
                'chunkwise_max': [results_by_sigma[s]['chunkwise_max'] for s in sigma_vals],
                'parallel_max': [results_by_sigma[s]['parallel_max'] for s in sigma_vals],
                'recurrence_mean': [results_by_sigma[s]['recurrence_mean'] for s in sigma_vals],
                'chunkwise_mean': [results_by_sigma[s]['chunkwise_mean'] for s in sigma_vals],
                'parallel_mean': [results_by_sigma[s]['parallel_mean'] for s in sigma_vals],
            }

            # Plot max differences
            ax1 = axes[row, 0]
            for key in ['recurrence', 'chunkwise', 'parallel']:
                ax1.plot(sigma_vals, data[f'{key}_max'], f'{markers[key]}-',
                        color=colors[key], label=labels[key], linewidth=2, markersize=8)
            ax1.set_xlabel('Perturbation Sigma', fontsize=11)
            ax1.set_ylabel('Max Output Difference', fontsize=11)
            ax1.set_title(f'{prec_name.upper()} - Max Output Difference', fontsize=12, fontweight='bold')
            ax1.set_xscale('log')
            ax1.set_yscale('log')
            ax1.legend(fontsize=9)
            ax1.grid(True, alpha=0.3)

            # Plot mean differences
            ax2 = axes[row, 1]
            for key in ['recurrence', 'chunkwise', 'parallel']:
                ax2.plot(sigma_vals, data[f'{key}_mean'], f'{markers[key]}-',
                        color=colors[key], label=labels[key], linewidth=2, markersize=8)
            ax2.set_xlabel('Perturbation Sigma', fontsize=11)
            ax2.set_ylabel('Mean Output Difference', fontsize=11)
            ax2.set_title(f'{prec_name.upper()} - Mean Output Difference', fontsize=12, fontweight='bold')
            ax2.set_xscale('log')
            ax2.set_yscale('log')
            ax2.legend(fontsize=9)
            ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        filename = f'../plots/delta_rule_perturbation_{noise_type}.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to: {filename}")
        plt.close()

    # Summary tables
    for noise_type in noise_types:
        print("\n" + "=" * 120)
        print(f"Summary: Mean Output Difference vs Perturbation Sigma ({noise_type.upper()} noise)")
        print("=" * 120)

        for prec_name in precisions.keys():
            print(f"\n{prec_name.upper()}:")
            print("-" * 100)
            header = f"{'Sigma':<12} {'Recurrence':<20} {'Chunkwise':<20} {'Parallel':<20}"
            print(header)
            print("-" * 100)
            for sigma in sigmas:
                r = all_results[noise_type][prec_name][sigma]
                print(f"{sigma:<12.1e} {r['recurrence_mean']:<20.2e} {r['chunkwise_mean']:<20.2e} {r['parallel_mean']:<20.2e}")


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()
