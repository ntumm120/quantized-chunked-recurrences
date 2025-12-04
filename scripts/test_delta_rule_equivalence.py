"""
Test script to verify that all delta rule implementations produce equivalent outputs.
"""

import torch
import sys
sys.path.insert(0, '../3rdparty/flash-linear-attention')

from fla.ops.delta_rule.naive import (
    delta_rule_recurrence,
    delta_rule_chunkwise,
    delta_rule_parallel,
)


def test_delta_rule_equivalence(device='cuda', d_k=64, d_v=64, scale=1.0):
    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Parameters
    batch_size = 2
    num_heads = 4
    seq_len = 128  # Must be divisible by chunk_size (32) and BM (128)

    # Create random inputs with specified scale
    q = torch.randn(batch_size, num_heads, seq_len, d_k, dtype=torch.float32, device=device) * scale
    k = torch.randn(batch_size, num_heads, seq_len, d_k, dtype=torch.float32, device=device) * scale
    v = torch.randn(batch_size, num_heads, seq_len, d_v, dtype=torch.float32, device=device) * scale
    beta = torch.sigmoid(torch.randn(batch_size, num_heads, seq_len, dtype=torch.float32, device=device))

    print("Input shapes:")
    print(f"  q: {q.shape}")
    print(f"  k: {k.shape}")
    print(f"  v: {v.shape}")
    print(f"  beta: {beta.shape}")
    print()

    # Call each function
    print("Running delta_rule_recurrence...")
    o_recurrence, S_recurrence = delta_rule_recurrence(q, k, v, beta)
    print(f"  Output shape: {o_recurrence.shape}")
    print(f"  State shape: {S_recurrence.shape}")
    print(f"  Output has NaN: {torch.isnan(o_recurrence).any().item()}")
    print(f"  Output range: [{o_recurrence[~torch.isnan(o_recurrence)].min().item():.4f}, {o_recurrence[~torch.isnan(o_recurrence)].max().item():.4f}]" if not torch.isnan(o_recurrence).all() else "  All NaN")
    print()

    print("Running delta_rule_chunkwise...")
    o_chunkwise, S_chunkwise = delta_rule_chunkwise(q, k, v, beta, chunk_size=32)
    print(f"  Output shape: {o_chunkwise.shape}")
    print(f"  State shape: {S_chunkwise.shape}")
    print(f"  Output has NaN: {torch.isnan(o_chunkwise).any().item()}")
    print(f"  Output range: [{o_chunkwise[~torch.isnan(o_chunkwise)].min().item():.4f}, {o_chunkwise[~torch.isnan(o_chunkwise)].max().item():.4f}]" if not torch.isnan(o_chunkwise).all() else "  All NaN")
    print()

    print("Running delta_rule_parallel...")
    o_parallel, A_parallel = delta_rule_parallel(q, k, v, beta, BM=128, BN=32)
    print(f"  Output shape: {o_parallel.shape}")
    print(f"  Attention matrix shape: {A_parallel.shape}")
    print(f"  Output has NaN: {torch.isnan(o_parallel).any().item()}")
    print(f"  Output range: [{o_parallel[~torch.isnan(o_parallel)].min().item():.4f}, {o_parallel[~torch.isnan(o_parallel)].max().item():.4f}]" if not torch.isnan(o_parallel).all() else "  All NaN")
    print()

    # Compare outputs
    print("=" * 60)
    print("Comparing outputs (o):")
    print("=" * 60)

    # Recurrence vs Chunkwise
    diff_rec_chunk = (o_recurrence - o_chunkwise).abs()
    max_diff_rec_chunk = diff_rec_chunk.max().item()
    mean_diff_rec_chunk = diff_rec_chunk.mean().item()
    print(f"\nRecurrence vs Chunkwise:")
    print(f"  Max absolute difference: {max_diff_rec_chunk:.2e}")
    print(f"  Mean absolute difference: {mean_diff_rec_chunk:.2e}")
    print(f"  All close (atol=1e-4): {torch.allclose(o_recurrence, o_chunkwise, atol=1e-4)}")

    # Recurrence vs Parallel
    diff_rec_par = (o_recurrence - o_parallel).abs()
    max_diff_rec_par = diff_rec_par.max().item()
    mean_diff_rec_par = diff_rec_par.mean().item()
    print(f"\nRecurrence vs Parallel:")
    print(f"  Max absolute difference: {max_diff_rec_par:.2e}")
    print(f"  Mean absolute difference: {mean_diff_rec_par:.2e}")
    print(f"  All close (atol=1e-4): {torch.allclose(o_recurrence, o_parallel, atol=1e-4)}")

    # Chunkwise vs Parallel
    diff_chunk_par = (o_chunkwise - o_parallel).abs()
    max_diff_chunk_par = diff_chunk_par.max().item()
    mean_diff_chunk_par = diff_chunk_par.mean().item()
    print(f"\nChunkwise vs Parallel:")
    print(f"  Max absolute difference: {max_diff_chunk_par:.2e}")
    print(f"  Mean absolute difference: {mean_diff_chunk_par:.2e}")
    print(f"  All close (atol=1e-4): {torch.allclose(o_chunkwise, o_parallel, atol=1e-4)}")

    # Compare final states (recurrence vs chunkwise)
    print()
    print("=" * 60)
    print("Comparing final states (S):")
    print("=" * 60)
    diff_state = (S_recurrence - S_chunkwise).abs()
    max_diff_state = diff_state.max().item()
    mean_diff_state = diff_state.mean().item()
    print(f"\nRecurrence vs Chunkwise final state:")
    print(f"  Max absolute difference: {max_diff_state:.2e}")
    print(f"  Mean absolute difference: {mean_diff_state:.2e}")
    print(f"  All close (atol=1e-4): {torch.allclose(S_recurrence, S_chunkwise, atol=1e-4)}")

    # Summary
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    all_outputs_equivalent = (
        torch.allclose(o_recurrence, o_chunkwise, atol=1e-4) and
        torch.allclose(o_recurrence, o_parallel, atol=1e-4) and
        torch.allclose(o_chunkwise, o_parallel, atol=1e-4)
    )

    if all_outputs_equivalent:
        print("✓ All three implementations produce equivalent outputs!")
    else:
        print("✗ Implementations produce different outputs!")

    states_equivalent = torch.allclose(S_recurrence, S_chunkwise, atol=1e-4)
    if states_equivalent:
        print("✓ Recurrence and Chunkwise produce equivalent final states!")
    else:
        print("✗ Final states differ between recurrence and chunkwise!")

    return all_outputs_equivalent and states_equivalent


if __name__ == "__main__":
    if torch.cuda.is_available():
        device = 'cuda'
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        device = 'cpu'
        print("CUDA is not available. Running on CPU...")
    print()

    # Run tests with different configurations
    test_configs = [
        {"d_k": 16, "d_v": 16, "scale": 0.1, "name": "Small dimensions (d=16, scale=0.1)"},
        {"d_k": 64, "d_v": 64, "scale": 0.1, "name": "Large dimensions (d=64, scale=0.1)"},
        {"d_k": 64, "d_v": 64, "scale": 0.01, "name": "Large dimensions (d=64, scale=0.01)"},
    ]

    all_success = True
    for config in test_configs:
        print("\n" + "=" * 70)
        print(f"Testing: {config['name']}")
        print("=" * 70 + "\n")
        success = test_delta_rule_equivalence(
            device=device,
            d_k=config["d_k"],
            d_v=config["d_v"],
            scale=config["scale"]
        )
        all_success = all_success and success

    print("\n" + "=" * 70)
    print("FINAL RESULT")
    print("=" * 70)
    if all_success:
        print("All tests passed!")
    else:
        print("Some tests failed!")
    sys.exit(0 if all_success else 1)
