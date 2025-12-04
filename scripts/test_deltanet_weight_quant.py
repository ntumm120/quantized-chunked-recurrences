"""
Test weight quantization impact on DeltaNet using the three naive PyTorch kernels:
- Recurrence (step-by-step)
- Chunkwise
- Parallel
"""

import torch
import torch.nn as nn
import sys
import copy
import matplotlib.pyplot as plt

from delta_rule_naive_multiprec import (
    delta_rule_recurrence,
    delta_rule_chunkwise,
    delta_rule_parallel,
)


class SimpleDeltaNetLayer(nn.Module):
    """Simplified DeltaNet layer for testing weight quantization."""

    def __init__(self, d_model=512, num_heads=8, d_head=64):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_head
        self.d_k = num_heads * d_head
        self.d_v = num_heads * d_head

        self.q_proj = nn.Linear(d_model, self.d_k, bias=False)
        self.k_proj = nn.Linear(d_model, self.d_k, bias=False)
        self.v_proj = nn.Linear(d_model, self.d_v, bias=False)
        self.beta_proj = nn.Linear(d_model, num_heads, bias=False)
        self.o_proj = nn.Linear(self.d_v, d_model, bias=False)

    def forward(self, x, kernel='recurrence'):
        """
        Args:
            x: (batch, seq_len, d_model)
            kernel: 'recurrence', 'chunkwise', 'parallel'
        """
        B, L, _ = x.shape

        # Project to q, k, v - output shape (B, L, num_heads, d_head)
        q = self.q_proj(x).view(B, L, self.num_heads, self.d_head)
        k = self.k_proj(x).view(B, L, self.num_heads, self.d_head)
        v = self.v_proj(x).view(B, L, self.num_heads, self.d_head)
        beta = torch.sigmoid(self.beta_proj(x))  # (B, L, num_heads)

        # Apply activation (silu) like real DeltaNet
        q = torch.nn.functional.silu(q)
        k = torch.nn.functional.silu(k)
        v = torch.nn.functional.silu(v)

        # Naive kernels expect (B, H, L, D) format
        q = q.transpose(1, 2)  # (B, H, L, D)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        beta = beta.transpose(1, 2)  # (B, H, L)

        # Choose kernel
        if kernel == 'recurrence':
            o, _ = delta_rule_recurrence(q, k, v, beta)
        elif kernel == 'chunkwise':
            o, _ = delta_rule_chunkwise(q, k, v, beta, chunk_size=32)
        elif kernel == 'parallel':
            o, _ = delta_rule_parallel(q, k, v, beta, BM=128, BN=32)
        else:
            raise ValueError(f"Unknown kernel: {kernel}")

        # Back to (B, L, H, D) then flatten
        o = o.transpose(1, 2).reshape(B, L, -1)
        o = self.o_proj(o)
        return o


def quantize_tensor_int8_symmetric(tensor):
    """Per-tensor symmetric int8 quantization."""
    max_abs = tensor.abs().max()
    scale = max_abs / 127.0
    if scale > 0:
        quantized = torch.clamp(torch.round(tensor / scale), -128, 127)
        dequantized = quantized * scale
    else:
        dequantized = tensor.clone()
    return dequantized


def quantize_model_weights(model, quant_type):
    """Quantize all weights in the model."""
    model_copy = copy.deepcopy(model)
    for name, param in model_copy.named_parameters():
        if quant_type == 'fp32':
            pass
        elif quant_type == 'fp16':
            param.data = param.data.half().float()
        elif quant_type == 'bf16':
            param.data = param.data.bfloat16().float()
        elif quant_type == 'int8':
            param.data = quantize_tensor_int8_symmetric(param.data)
        else:
            raise ValueError(f"Unknown quant_type: {quant_type}")
    return model_copy


def test_kernel_equivalence(device='cuda'):
    """First verify that all kernels produce equivalent outputs."""
    print("=" * 60)
    print("Testing kernel equivalence (all should match)")
    print("=" * 60)

    torch.manual_seed(42)
    model = SimpleDeltaNetLayer(d_model=256, num_heads=4, d_head=64).to(device)
    x = torch.randn(1, 128, 256, device=device) * 0.1

    kernels = ['recurrence', 'chunkwise', 'parallel']
    outputs = {}

    with torch.no_grad():
        for kernel in kernels:
            outputs[kernel] = model(x, kernel=kernel)
            print(f"  {kernel}: output shape = {outputs[kernel].shape}, "
                  f"mean = {outputs[kernel].mean().item():.6f}")

    print("\nPairwise differences:")
    for i, k1 in enumerate(kernels):
        for k2 in kernels[i+1:]:
            diff = (outputs[k1] - outputs[k2]).abs()
            print(f"  {k1} vs {k2}: max={diff.max().item():.2e}, mean={diff.mean().item():.2e}")


def test_weight_quantization(device='cuda', seq_len=512):
    """Test weight quantization impact on each kernel."""
    torch.manual_seed(42)

    model_fp32 = SimpleDeltaNetLayer(d_model=512, num_heads=8, d_head=64).to(device)
    x = torch.randn(1, seq_len, 512, device=device) * 0.1

    kernels = ['recurrence', 'chunkwise', 'parallel']
    quant_types = ['fp16', 'bf16', 'int8']

    # Get fp32 baseline outputs
    baseline = {}
    with torch.no_grad():
        for kernel in kernels:
            baseline[kernel] = model_fp32(x, kernel=kernel)

    results = {qt: {} for qt in quant_types}

    for quant_type in quant_types:
        model_quant = quantize_model_weights(model_fp32, quant_type)

        with torch.no_grad():
            for kernel in kernels:
                out_quant = model_quant(x, kernel=kernel)
                diff = (out_quant - baseline[kernel]).abs()
                results[quant_type][kernel] = {
                    'max': diff.max().item(),
                    'mean': diff.mean().item(),
                }

    return results


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    # First test kernel equivalence
    test_kernel_equivalence(device)
    print()

    # Test weight quantization at different sequence lengths
    seq_lengths = [128, 256, 512, 1024, 2048]
    kernels = ['recurrence', 'chunkwise', 'parallel']
    quant_types = ['fp16', 'bf16', 'int8']

    all_results = {}
    for seq_len in seq_lengths:
        print(f"Testing seq_len={seq_len}...")
        all_results[seq_len] = test_weight_quantization(device, seq_len)

    # Print results
    for quant_type in quant_types:
        print(f"\n{'='*80}")
        print(f"Weight Quantization: {quant_type.upper()}")
        print('='*80)
        header = f"{'Seq Len':<10}"
        for kernel in kernels:
            header += f"{kernel:<20}"
        print(header)
        print("-" * 90)
        for seq_len in seq_lengths:
            row = f"{seq_len:<10}"
            for kernel in kernels:
                val = all_results[seq_len][quant_type][kernel]['mean']
                row += f"{val:<20.2e}"
            print(row)

    # Create plots
    fig, axes = plt.subplots(len(quant_types), 2, figsize=(14, 4*len(quant_types)))

    colors = {'recurrence': 'tab:blue', 'chunkwise': 'tab:orange', 'parallel': 'tab:green'}
    markers = {'recurrence': 'o', 'chunkwise': 's', 'parallel': '^'}

    for row, quant_type in enumerate(quant_types):
        # Max diff
        ax1 = axes[row, 0]
        for kernel in kernels:
            y_vals = [all_results[s][quant_type][kernel]['max'] for s in seq_lengths]
            ax1.plot(seq_lengths, y_vals, f'{markers[kernel]}-',
                    color=colors[kernel], label=kernel, linewidth=2, markersize=8)
        ax1.set_xlabel('Sequence Length')
        ax1.set_ylabel('Max Output Diff vs FP32')
        ax1.set_title(f'{quant_type.upper()} Weights - Max Diff')
        ax1.set_xscale('log', base=2)
        ax1.set_yscale('log')
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)

        # Mean diff
        ax2 = axes[row, 1]
        for kernel in kernels:
            y_vals = [all_results[s][quant_type][kernel]['mean'] for s in seq_lengths]
            ax2.plot(seq_lengths, y_vals, f'{markers[kernel]}-',
                    color=colors[kernel], label=kernel, linewidth=2, markersize=8)
        ax2.set_xlabel('Sequence Length')
        ax2.set_ylabel('Mean Output Diff vs FP32')
        ax2.set_title(f'{quant_type.upper()} Weights - Mean Diff')
        ax2.set_xscale('log', base=2)
        ax2.set_yscale('log')
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('../plots/deltanet_weight_quant_kernels.png', dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: ../plots/deltanet_weight_quant_kernels.png")


if __name__ == "__main__":
    main()
