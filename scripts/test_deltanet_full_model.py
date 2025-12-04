"""
Test weight quantization impact on a FULL DeltaNet model loaded from HuggingFace,
comparing the three naive PyTorch kernels (recurrence, chunkwise, parallel).

For each (weight_quant, compute_precision) pair:
- Baseline: Unquantized (fp32 weights) at that compute precision
- Compare: Quantized weights at that same compute precision
- Plot all 3 algorithms, x-axis = sequence length
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import sys
import os
import matplotlib.pyplot as plt
import numpy as np

# Add the flash-linear-attention to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../3rdparty/flash-linear-attention'))

# Import fla.models.delta_net to register custom models with transformers
from fla.models.delta_net import DeltaNetConfig, DeltaNetForCausalLM

from transformers import AutoTokenizer, AutoModelForCausalLM
from einops import rearrange

# Import naive PyTorch implementations (native dtype versions for precision studies)
from fla.ops.delta_rule.naive import (
    delta_rule_recurrence_native_dtype as delta_rule_recurrence,
    delta_rule_chunkwise_native_dtype as delta_rule_chunkwise,
    delta_rule_parallel_native_dtype as delta_rule_parallel,
)


def patch_deltanet_layer_forward(layer, kernel_mode='recurrence', compute_dtype=None):
    """
    Patch a DeltaNet layer to use naive PyTorch kernels instead of Triton.
    """
    def patched_forward(
        hidden_states,
        attention_mask=None,
        past_key_values=None,
        use_cache=False,
        output_attentions=False,
        **kwargs,
    ):
        batch_size, q_len, _ = hidden_states.shape

        # Project q, k, v through convolutions
        if layer.use_short_conv:
            q = layer.q_conv1d(x=layer.q_proj(hidden_states), cache=None, output_final_state=False)[0]
            k = layer.k_conv1d(x=layer.k_proj(hidden_states), cache=None, output_final_state=False)[0]
            v = layer.v_conv1d(x=layer.v_proj(hidden_states), cache=None, output_final_state=False)[0]
        else:
            q = layer.q_proj(hidden_states)
            k = layer.k_proj(hidden_states)
            if layer.qk_activation == 'silu':
                q, k = F.silu(q), F.silu(k)
            v = F.silu(layer.v_proj(hidden_states))

        # Reshape to (batch, seq, heads, dim)
        q = rearrange(q, '... (h d) -> ... h d', d=layer.head_k_dim)
        k = rearrange(k, '... (h d) -> ... h d', d=layer.head_k_dim)
        v = rearrange(v, '... (h d) -> ... h d', d=layer.head_v_dim)

        # Apply L2 normalization if configured
        if layer.qk_norm == 'l2':
            q = F.normalize(q, p=2, dim=-1)
            k = F.normalize(k, p=2, dim=-1)

        # Get beta
        if layer.use_beta:
            beta = layer.b_proj(hidden_states).sigmoid()
        else:
            beta = torch.ones(batch_size, q_len, layer.num_heads, device=q.device, dtype=q.dtype)

        if layer.allow_neg_eigval:
            beta = beta * 2.

        # Transpose to (batch, heads, seq, dim) for naive kernels
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        beta = beta.transpose(1, 2)

        # Convert to compute dtype if specified
        orig_dtype = q.dtype
        if compute_dtype is not None:
            q = q.to(compute_dtype)
            k = k.to(compute_dtype)
            v = v.to(compute_dtype)
            beta = beta.to(compute_dtype)

        # Call the appropriate naive kernel
        if kernel_mode == 'recurrence':
            o, _ = delta_rule_recurrence(q, k, v, beta)
        elif kernel_mode == 'chunkwise':
            chunk_size = 32
            if q_len % chunk_size != 0:
                pad_len = chunk_size - (q_len % chunk_size)
                q = F.pad(q, (0, 0, 0, pad_len))
                k = F.pad(k, (0, 0, 0, pad_len))
                v = F.pad(v, (0, 0, 0, pad_len))
                beta = F.pad(beta, (0, pad_len))
                o, _ = delta_rule_chunkwise(q, k, v, beta, chunk_size=chunk_size)
                o = o[:, :, :q_len, :]
            else:
                o, _ = delta_rule_chunkwise(q, k, v, beta, chunk_size=chunk_size)
        elif kernel_mode == 'parallel':
            BM, BN = 128, 32
            if q_len % BN != 0:
                pad_len = BN - (q_len % BN)
                q = F.pad(q, (0, 0, 0, pad_len))
                k = F.pad(k, (0, 0, 0, pad_len))
                v = F.pad(v, (0, 0, 0, pad_len))
                beta = F.pad(beta, (0, pad_len))
                o, _ = delta_rule_parallel(q, k, v, beta, BM=BM, BN=BN)
                o = o[:, :, :q_len, :]
            else:
                o, _ = delta_rule_parallel(q, k, v, beta, BM=BM, BN=BN)
        else:
            raise ValueError(f"Unknown kernel mode: {kernel_mode}")

        # Convert back to original dtype
        o = o.to(orig_dtype)

        # Apply output norm and projection
        o = o.transpose(1, 2)
        if layer.use_gate:
            g = rearrange(layer.g_proj(hidden_states), '... (h d) -> ... h d', d=layer.head_v_dim)
            o = layer.o_norm(o, g)
        else:
            o = layer.o_norm(o)
        o = rearrange(o, 'b t h d -> b t (h d)')
        o = layer.o_proj(o)

        return o, None, past_key_values

    layer.forward = patched_forward
    return layer


def patch_model_kernels(model, kernel_mode='recurrence', compute_dtype=None):
    """Patch all DeltaNet layers in the model."""
    for name, module in model.named_modules():
        if module.__class__.__name__ == 'DeltaNet':
            patch_deltanet_layer_forward(module, kernel_mode, compute_dtype)
    return model


def quantize_tensor_int8_per_tensor(tensor):
    """Per-tensor symmetric INT8 quantization."""
    max_abs = tensor.abs().max()
    scale = max_abs / 127.0
    if scale > 0:
        quantized = torch.clamp(torch.round(tensor / scale), -128, 127)
        return quantized * scale
    return tensor.clone()


def quantize_tensor_int8_per_channel(tensor):
    """Per-channel symmetric INT8 quantization (along output dimension)."""
    if tensor.ndim < 2:
        return quantize_tensor_int8_per_tensor(tensor)

    # Per-channel: quantize along dim 0 (output channels)
    # Shape: (out_features, in_features) for Linear
    max_abs = tensor.abs().amax(dim=tuple(range(1, tensor.ndim)), keepdim=True)
    scale = max_abs / 127.0
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    quantized = torch.clamp(torch.round(tensor / scale), -128, 127)
    return quantized * scale


def quantize_tensor_int8_groupwise(tensor, group_size=128):
    """Groupwise symmetric INT8 quantization.

    Quantizes along the last dimension in groups of `group_size`.
    Each group gets its own scale.
    """
    if tensor.ndim < 2:
        return quantize_tensor_int8_per_tensor(tensor)

    # Flatten to 2D: (everything_else, last_dim)
    original_shape = tensor.shape
    tensor_2d = tensor.view(-1, original_shape[-1])
    num_rows, num_cols = tensor_2d.shape

    # If last dim is smaller than group_size, fall back to per-channel
    if num_cols < group_size:
        return quantize_tensor_int8_per_channel(tensor)

    # Pad if necessary
    if num_cols % group_size != 0:
        pad_size = group_size - (num_cols % group_size)
        tensor_padded = F.pad(tensor_2d, (0, pad_size))
    else:
        tensor_padded = tensor_2d
        pad_size = 0

    # Reshape to (num_rows, num_groups, group_size)
    num_groups = tensor_padded.shape[1] // group_size
    reshaped = tensor_padded.view(num_rows, num_groups, group_size)

    # Compute scale per group
    max_abs = reshaped.abs().amax(dim=-1, keepdim=True)
    scale = max_abs / 127.0
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))

    # Quantize and dequantize
    quantized = torch.clamp(torch.round(reshaped / scale), -128, 127)
    dequantized = quantized * scale

    # Reshape back to 2D
    result_2d = dequantized.view(num_rows, -1)

    # Remove padding
    if pad_size > 0:
        result_2d = result_2d[:, :num_cols]

    # Reshape back to original
    return result_2d.view(original_shape)


def quantize_tensor_fp8_e4m3(tensor):
    """FP8 E4M3 quantization (4 exponent bits, 3 mantissa bits).

    E4M3 range: [-448, 448], good for weights.
    """
    # Convert to FP8 E4M3 and back to float32
    # torch.float8_e4m3fn is the "fn" variant (no NaN, larger range)
    fp8 = tensor.to(torch.float8_e4m3fn)
    return fp8.to(torch.float32)


def quantize_tensor_fp8_e5m2(tensor):
    """FP8 E5M2 quantization (5 exponent bits, 2 mantissa bits).

    E5M2 has larger dynamic range but less precision.
    """
    fp8 = tensor.to(torch.float8_e5m2)
    return fp8.to(torch.float32)


def quantize_model_weights(model, quant_type, group_size=None):
    """Quantize all weights in the model.

    Args:
        model: The model to quantize
        quant_type: 'fp32', 'fp16', 'bf16', 'int8_per_tensor', 'int8_per_channel',
                    'int8_groupwise', 'fp8_e4m3', 'fp8_e5m2'
        group_size: For groupwise quantization, the group size
    """
    model_copy = copy.deepcopy(model)
    for name, param in model_copy.named_parameters():
        if quant_type == 'fp32':
            pass
        elif quant_type == 'fp16':
            param.data = param.data.half().float()
        elif quant_type == 'bf16':
            param.data = param.data.bfloat16().float()
        elif quant_type == 'int8_per_tensor':
            param.data = quantize_tensor_int8_per_tensor(param.data)
        elif quant_type == 'int8_per_channel':
            param.data = quantize_tensor_int8_per_channel(param.data)
        elif quant_type == 'int8_groupwise':
            param.data = quantize_tensor_int8_groupwise(param.data, group_size=group_size)
        elif quant_type == 'fp8_e4m3':
            param.data = quantize_tensor_fp8_e4m3(param.data)
        elif quant_type == 'fp8_e5m2':
            param.data = quantize_tensor_fp8_e5m2(param.data)
    return model_copy


def run_model(model_fp32, input_ids, kernel, compute_dtype):
    """Run model with specified kernel and compute dtype."""
    model_copy = copy.deepcopy(model_fp32)
    patch_model_kernels(model_copy, kernel_mode=kernel, compute_dtype=compute_dtype)
    with torch.no_grad():
        outputs = model_copy(input_ids)
        logits = outputs.logits.clone()
    del model_copy
    torch.cuda.empty_cache()
    return logits


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    model_name = 'fla-hub/delta_net-1.3B-100B'
    print(f"Loading model: {model_name}")
    model_fp32 = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
    ).float().to(device)
    model_fp32.eval()

    vocab_size = model_fp32.config.vocab_size
    print(f"Model: {model_fp32.__class__.__name__}")
    print(f"Parameters: {sum(p.numel() for p in model_fp32.parameters()):,}")
    print(f"Vocab size: {vocab_size}")
    print()

    # Configuration
    seq_lengths = [128, 256, 512, 1024]
    kernels = ['recurrence', 'chunkwise', 'parallel']

    # Different INT8 granularities (coarse to fine)
    quant_configs = [
        ('int8_per_tensor', None, 'Per-Tensor'),
        ('int8_per_channel', None, 'Per-Channel'),
        ('int8_groupwise', 128, 'Group-128'),
        ('int8_groupwise', 64, 'Group-64'),
        ('int8_groupwise', 32, 'Group-32'),
    ]

    # Focus on bf16 compute since that's where algorithmic differences show up
    compute_precisions = {
        'bf16': torch.bfloat16,
    }

    print("Input: Random token IDs (seed=42)")
    print(f"Sequence lengths: {seq_lengths}")
    print(f"Algorithms: {kernels}")
    print(f"Quantization granularities: {[c[2] for c in quant_configs]}")
    print(f"Compute precision: bf16 (where algorithmic differences are most visible)")
    print()

    # Results structure: results[quant_name][seq_len][kernel] = {'max': ..., 'mean': ...}
    results = {c[2]: {s: {} for s in seq_lengths} for c in quant_configs}

    for seq_len in seq_lengths:
        print(f"\n{'='*60}")
        print(f"Sequence length: {seq_len}")
        print('='*60)

        # Generate random input
        torch.manual_seed(42)
        input_ids = torch.randint(100, vocab_size - 100, (1, seq_len), device=device)

        for quant_type, group_size, quant_name in quant_configs:
            print(f"\n  Quantization: {quant_name}")
            model_quant = quantize_model_weights(model_fp32, quant_type, group_size=group_size)

            for compute_name, compute_dtype in compute_precisions.items():
                results[quant_name][seq_len] = {}

                for kernel in kernels:
                    baseline = run_model(model_fp32, input_ids, kernel, compute_dtype)
                    logits_quant = run_model(model_quant, input_ids, kernel, compute_dtype)

                    diff = (logits_quant - baseline).abs()
                    results[quant_name][seq_len][kernel] = {
                        'max': diff.max().item(),
                        'mean': diff.mean().item(),
                    }
                    print(f"    {kernel}: max={diff.max().item():.2e}, mean={diff.mean().item():.2e}")

            del model_quant
            torch.cuda.empty_cache()

    # Create plots: 1 row with subplots for each granularity
    colors = {'recurrence': 'tab:blue', 'chunkwise': 'tab:orange', 'parallel': 'tab:green'}
    markers = {'recurrence': 'o', 'chunkwise': 's', 'parallel': '^'}
    quant_names = [c[2] for c in quant_configs]

    for metric in ['mean', 'max']:
        fig, axes = plt.subplots(1, len(quant_names), figsize=(4*len(quant_names), 5))

        for col, quant_name in enumerate(quant_names):
            ax = axes[col]

            for kernel in kernels:
                y_vals = [results[quant_name][s][kernel][metric] for s in seq_lengths]
                ax.plot(seq_lengths, y_vals, f'{markers[kernel]}-',
                        color=colors[kernel], label=kernel, linewidth=2, markersize=8)

            ax.set_xlabel('Sequence Length')
            ax.set_ylabel(f'{metric.capitalize()} Abs Diff')
            ax.set_title(f'{quant_name}')
            ax.set_xscale('log', base=2)
            min_val = min(results[quant_name][seq_lengths[0]][k][metric] for k in kernels)
            if min_val > 0:
                ax.set_yscale('log')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        plt.suptitle(f'DeltaNet 1.3B: INT8 Quantization Granularity (BF16 Compute)\n'
                     f'{metric.capitalize()} Abs Diff vs Sequence Length',
                     fontsize=12)
        plt.tight_layout()
        plt.savefig(f'../plots/deltanet_1.3B_int8_granularity_{metric}.png', dpi=150, bbox_inches='tight')
        print(f"\nPlot saved: ../plots/deltanet_1.3B_int8_granularity_{metric}.png")
        plt.close()

    # Also plot error vs granularity at fixed seq_len=1024
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    x_labels = quant_names
    x_pos = np.arange(len(x_labels))
    width = 0.25

    for i, kernel in enumerate(kernels):
        means = [results[qn][1024][kernel]['mean'] for qn in quant_names]
        ax.bar(x_pos + i*width, means, width, label=kernel, color=colors[kernel])

    ax.set_xlabel('Quantization Granularity')
    ax.set_ylabel('Mean Abs Diff')
    ax.set_title('DeltaNet 1.3B: Quantization Error by Granularity (seq_len=1024, BF16 compute)')
    ax.set_xticks(x_pos + width)
    ax.set_xticklabels(x_labels)
    ax.legend()
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('../plots/deltanet_1.3B_int8_granularity_bar.png', dpi=150, bbox_inches='tight')
    print(f"Plot saved: ../plots/deltanet_1.3B_int8_granularity_bar.png")
    plt.close()

    # Print summary table
    print("\n" + "="*80)
    print("SUMMARY: INT8 Quantization Error at seq_len=1024 (BF16 compute)")
    print("="*80)
    print(f"\n{'Granularity':<15} {'recurrence':<15} {'chunkwise':<15} {'parallel':<15} {'rec/par ratio':<15}")
    print("-"*75)
    for quant_name in quant_names:
        rec = results[quant_name][1024]['recurrence']['mean']
        chu = results[quant_name][1024]['chunkwise']['mean']
        par = results[quant_name][1024]['parallel']['mean']
        ratio = rec / par if par > 0 else 0
        print(f"{quant_name:<15} {rec:<15.2e} {chu:<15.2e} {par:<15.2e} {ratio:<15.2f}")


if __name__ == "__main__":
    main()
