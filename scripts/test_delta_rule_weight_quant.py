"""
Test script to measure how weight quantization affects delta rule algorithm outputs.

We:
1. Create nn.Linear layers for q, k, v projections
2. Quantize weights to fp16, bf16, int8 (per-tensor symmetric)
3. Compare outputs vs fp32 baseline for each algorithm
"""

import torch
import torch.nn as nn
import torch.multiprocessing as mp
import sys
import matplotlib.pyplot as plt
import copy

from delta_rule_naive_multiprec import (
    delta_rule_recurrence,
    delta_rule_chunkwise,
    delta_rule_parallel,
)


class DeltaRuleProjection(nn.Module):
    """Simple module that projects input to q, k, v and computes beta."""

    def __init__(self, d_model, d_k, d_v, scale=0.1):
        super().__init__()
        self.q_proj = nn.Linear(d_model, d_k, bias=False)
        self.k_proj = nn.Linear(d_model, d_k, bias=False)
        self.v_proj = nn.Linear(d_model, d_v, bias=False)
        self.beta_proj = nn.Linear(d_model, 1, bias=False)
        self.scale = scale

        # Initialize with small weights for numerical stability
        for module in [self.q_proj, self.k_proj, self.v_proj, self.beta_proj]:
            nn.init.normal_(module.weight, std=0.02)

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            q, k, v: (batch, 1, seq_len, d_k/d_v)  # 1 head
            beta: (batch, 1, seq_len)
        """
        q = self.q_proj(x).unsqueeze(1) * self.scale  # (batch, 1, seq_len, d_k)
        k = self.k_proj(x).unsqueeze(1) * self.scale
        v = self.v_proj(x).unsqueeze(1) * self.scale
        beta = torch.sigmoid(self.beta_proj(x)).squeeze(-1).unsqueeze(1)  # (batch, 1, seq_len)
        return q, k, v, beta


def quantize_tensor_int8_symmetric(tensor):
    """
    Per-tensor symmetric int8 quantization.

    Args:
        tensor: fp32 tensor to quantize

    Returns:
        dequantized tensor (fp32, but with int8 precision loss)
    """
    # Find scale
    max_abs = tensor.abs().max()
    scale = max_abs / 127.0

    # Quantize and dequantize
    if scale > 0:
        quantized = torch.clamp(torch.round(tensor / scale), -128, 127)
        dequantized = quantized * scale
    else:
        dequantized = tensor.clone()

    return dequantized


def quantize_model_weights(model, quant_type):
    """
    Quantize all weights in the model.

    Args:
        model: nn.Module
        quant_type: 'fp32', 'fp16', 'bf16', 'int8'

    Returns:
        New model with quantized weights (always returns fp32 model for computation)
    """
    model_copy = copy.deepcopy(model)

    for name, param in model_copy.named_parameters():
        if quant_type == 'fp32':
            pass  # No change
        elif quant_type == 'fp16':
            # Quantize to fp16 and back to fp32
            param.data = param.data.half().float()
        elif quant_type == 'bf16':
            # Quantize to bf16 and back to fp32
            param.data = param.data.bfloat16().float()
        elif quant_type == 'int8':
            # Per-tensor symmetric int8 quantization
            param.data = quantize_tensor_int8_symmetric(param.data)
        else:
            raise ValueError(f"Unknown quant_type: {quant_type}")

    return model_copy


def test_weight_quantization(device='cuda:0', seq_len=1024, d_model=256, d_k=64, d_v=64):
    """
    Test how weight quantization affects each algorithm's output.

    Returns:
        dict mapping quant_type -> algorithm -> {max_diff, mean_diff}
    """
    torch.manual_seed(42)

    batch_size = 1

    # Create input with reasonable scale
    x = torch.randn(batch_size, seq_len, d_model, device=device) * 0.1

    # Create model with fp32 weights
    model_fp32 = DeltaRuleProjection(d_model, d_k, d_v).to(device)

    # Get fp32 baseline outputs
    with torch.no_grad():
        q_fp32, k_fp32, v_fp32, beta_fp32 = model_fp32(x)

        o_rec_fp32, _ = delta_rule_recurrence(q_fp32, k_fp32, v_fp32, beta_fp32)
        o_chunk_fp32, _ = delta_rule_chunkwise(q_fp32, k_fp32, v_fp32, beta_fp32, chunk_size=32)
        o_par_fp32, _ = delta_rule_parallel(q_fp32, k_fp32, v_fp32, beta_fp32, BM=128, BN=32)

    quant_types = ['fp16', 'bf16', 'int8']
    results = {}

    for quant_type in quant_types:
        # Quantize model weights
        model_quant = quantize_model_weights(model_fp32, quant_type)

        with torch.no_grad():
            # Get q, k, v from quantized model
            q_quant, k_quant, v_quant, beta_quant = model_quant(x)

            # Run each algorithm
            o_rec_quant, _ = delta_rule_recurrence(q_quant, k_quant, v_quant, beta_quant)
            o_chunk_quant, _ = delta_rule_chunkwise(q_quant, k_quant, v_quant, beta_quant, chunk_size=32)
            o_par_quant, _ = delta_rule_parallel(q_quant, k_quant, v_quant, beta_quant, BM=128, BN=32)

            # Compute differences vs fp32 baseline
            results[quant_type] = {
                'recurrence_max': (o_rec_quant - o_rec_fp32).abs().max().item(),
                'recurrence_mean': (o_rec_quant - o_rec_fp32).abs().mean().item(),
                'chunkwise_max': (o_chunk_quant - o_chunk_fp32).abs().max().item(),
                'chunkwise_mean': (o_chunk_quant - o_chunk_fp32).abs().mean().item(),
                'parallel_max': (o_par_quant - o_par_fp32).abs().max().item(),
                'parallel_mean': (o_par_quant - o_par_fp32).abs().mean().item(),
            }

    return results


def run_tests_on_gpu(gpu_id, tasks, results_dict):
    """Run a set of test tasks on a specific GPU."""
    device = f'cuda:{gpu_id}'
    torch.cuda.set_device(device)

    for task in tasks:
        seq_len = task
        result = test_weight_quantization(device=device, seq_len=seq_len)
        results_dict[seq_len] = result
        print(f"  GPU {gpu_id}: seq_len={seq_len} done")


def main():
    num_gpus = torch.cuda.device_count()
    print(f"Found {num_gpus} GPUs:")
    for i in range(num_gpus):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    print()

    # Sequence lengths to test
    seq_lengths = [128, 256, 512, 1024, 2048, 4096, 8192]

    # Distribute tasks across GPUs
    tasks_per_gpu = [[] for _ in range(num_gpus)]
    for i, seq_len in enumerate(seq_lengths):
        tasks_per_gpu[i % num_gpus].append(seq_len)

    print(f"Running {len(seq_lengths)} tests across {num_gpus} GPUs...")
    print()

    # Use a manager dict for sharing results
    manager = mp.Manager()
    results_dict = manager.dict()

    # Run tests in parallel across GPUs
    processes = []
    for gpu_id in range(num_gpus):
        if tasks_per_gpu[gpu_id]:
            p = mp.Process(target=run_tests_on_gpu, args=(gpu_id, tasks_per_gpu[gpu_id], results_dict))
            p.start()
            processes.append(p)

    for p in processes:
        p.join()

    # Convert manager dict to regular dict
    results_dict = dict(results_dict)

    quant_types = ['fp16', 'bf16', 'int8']
    algorithms = ['recurrence', 'chunkwise', 'parallel']

    # Print results
    for quant_type in quant_types:
        print(f"\n{'='*60}")
        print(f"Weight Quantization: {quant_type.upper()}")
        print('='*60)
        print(f"{'Seq Len':<10} {'Recurrence':<20} {'Chunkwise':<20} {'Parallel':<20}")
        print("-" * 70)
        for seq_len in seq_lengths:
            r = results_dict[seq_len][quant_type]
            print(f"{seq_len:<10} {r['recurrence_mean']:<20.2e} {r['chunkwise_mean']:<20.2e} {r['parallel_mean']:<20.2e}")

    # Create plot - comparing algorithms across quantization types
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    colors = {'recurrence': 'tab:blue', 'chunkwise': 'tab:orange', 'parallel': 'tab:green'}
    markers = {'recurrence': 'o', 'chunkwise': 's', 'parallel': '^'}

    for col, quant_type in enumerate(quant_types):
        # Max differences
        ax1 = axes[0, col]
        for algo in algorithms:
            y_vals = [results_dict[s][quant_type][f'{algo}_max'] for s in seq_lengths]
            ax1.plot(seq_lengths, y_vals, f'{markers[algo]}-',
                    color=colors[algo], label=algo.capitalize(), linewidth=2, markersize=8)
        ax1.set_xlabel('Sequence Length', fontsize=11)
        ax1.set_ylabel('Max Output Diff vs FP32', fontsize=11)
        ax1.set_title(f'{quant_type.upper()} Weights - Max Diff', fontsize=12, fontweight='bold')
        ax1.set_xscale('log', base=2)
        ax1.set_yscale('log')
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3)
        ax1.set_xticks(seq_lengths)
        ax1.set_xticklabels([str(s) for s in seq_lengths], rotation=45)

        # Mean differences
        ax2 = axes[1, col]
        for algo in algorithms:
            y_vals = [results_dict[s][quant_type][f'{algo}_mean'] for s in seq_lengths]
            ax2.plot(seq_lengths, y_vals, f'{markers[algo]}-',
                    color=colors[algo], label=algo.capitalize(), linewidth=2, markersize=8)
        ax2.set_xlabel('Sequence Length', fontsize=11)
        ax2.set_ylabel('Mean Output Diff vs FP32', fontsize=11)
        ax2.set_title(f'{quant_type.upper()} Weights - Mean Diff', fontsize=12, fontweight='bold')
        ax2.set_xscale('log', base=2)
        ax2.set_yscale('log')
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)
        ax2.set_xticks(seq_lengths)
        ax2.set_xticklabels([str(s) for s in seq_lengths], rotation=45)

    plt.tight_layout()
    plt.savefig('../plots/delta_rule_weight_quantization.png', dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: ../plots/delta_rule_weight_quantization.png")

    # Also create a comparison plot across quantization types for a fixed seq_len
    fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))

    seq_len_ref = 1024
    quant_labels = quant_types
    x_pos = range(len(quant_types))
    width = 0.25

    # Max diff comparison
    ax1 = axes2[0]
    for i, algo in enumerate(algorithms):
        y_vals = [results_dict[seq_len_ref][qt][f'{algo}_max'] for qt in quant_types]
        ax1.bar([x + i*width for x in x_pos], y_vals, width,
               label=algo.capitalize(), color=colors[algo])
    ax1.set_xlabel('Quantization Type', fontsize=11)
    ax1.set_ylabel('Max Output Diff vs FP32', fontsize=11)
    ax1.set_title(f'Max Diff by Quant Type (seq_len={seq_len_ref})', fontsize=12, fontweight='bold')
    ax1.set_xticks([x + width for x in x_pos])
    ax1.set_xticklabels([qt.upper() for qt in quant_types])
    ax1.legend(fontsize=9)
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3, axis='y')

    # Mean diff comparison
    ax2 = axes2[1]
    for i, algo in enumerate(algorithms):
        y_vals = [results_dict[seq_len_ref][qt][f'{algo}_mean'] for qt in quant_types]
        ax2.bar([x + i*width for x in x_pos], y_vals, width,
               label=algo.capitalize(), color=colors[algo])
    ax2.set_xlabel('Quantization Type', fontsize=11)
    ax2.set_ylabel('Mean Output Diff vs FP32', fontsize=11)
    ax2.set_title(f'Mean Diff by Quant Type (seq_len={seq_len_ref})', fontsize=12, fontweight='bold')
    ax2.set_xticks([x + width for x in x_pos])
    ax2.set_xticklabels([qt.upper() for qt in quant_types])
    ax2.legend(fontsize=9)
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('../plots/delta_rule_weight_quant_comparison.png', dpi=150, bbox_inches='tight')
    print(f"Plot saved to: ../plots/delta_rule_weight_quant_comparison.png")


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()
