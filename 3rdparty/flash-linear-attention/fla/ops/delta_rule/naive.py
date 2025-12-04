"""
Naive PyTorch implementations of delta rule algorithms.

Standard implementations (compute in float32):
1. delta_rule_recurrence: Sequential O(L) step-by-step
2. delta_rule_chunkwise: Chunkwise O(L/C) chunks
3. delta_rule_parallel: Fully parallel O(L^2) attention
4. delta_rule_parallel_scan: Tree-based O(log L) parallel scan

Native dtype implementations (preserve fp16/bf16):
5. delta_rule_recurrence_native_dtype: Sequential, native precision
6. delta_rule_chunkwise_native_dtype: Chunkwise, native precision
7. delta_rule_parallel_native_dtype: Parallel, native precision
8. delta_rule_parallel_scan_native_dtype: Parallel scan, native precision
"""

import torch
from einops import rearrange


def delta_rule_recurrence(q, k, v, beta, initial_state=None, output_final_state=True):
    orig_dtype = q.dtype
    b, h, l, d_k = q.shape
    q, k, v, beta = map(lambda x: x.float(), [q, k, v, beta])
    d_v = v.shape[-1]
    o = torch.zeros_like(v)
    S = torch.zeros(b, h, d_k, d_v).to(v)
    q = q * (d_k ** -0.5)

    if beta.ndim < v.ndim:
        beta = beta[..., None]

    if initial_state is not None:
        S += initial_state

    for i in range(l):
        _k = k[:, :, i]
        _q = q[:, :, i]
        _v = v[:, :, i].clone()
        beta_i = beta[:, :, i]
        _v = _v - (S.clone() * _k[..., None]).sum(-2)
        _v = _v * beta_i
        S = S.clone() + _k.unsqueeze(-1) * _v.unsqueeze(-2)
        o[:, :, i] = torch.einsum('bhd,bhdm->bhm', _q, S)
    S = None if output_final_state is False else S
    return o.to(orig_dtype), S


def delta_rule_chunkwise(q, k, v, beta, chunk_size=32):
    b, h, l, d_k = q.shape
    d_v = v.shape[-1]
    q = q * (d_k ** -0.5)
    v = v * beta[..., None]
    k_beta = k * beta[..., None]

    assert l % chunk_size == 0

    # compute (I - tri(diag(beta) KK^T))^{-1}
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device), diagonal=0)
    q, k, v, k_beta = map(lambda x: rearrange(x, 'b h (n c) d -> b h n c d', c=chunk_size), [q, k, v, k_beta])
    attn = -(k_beta @ k.transpose(-1, -2)).masked_fill(mask, 0)
    for i in range(1, chunk_size):
        attn[..., i, :i] = attn[..., i, :i] + (attn[..., i, :, None].clone() * attn[..., :, :i].clone()).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=torch.float, device=q.device)

    u = attn @ v
    w = attn @ k_beta
    S = k.new_zeros(b, h, d_k, d_v)
    o = torch.zeros_like(v)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device), diagonal=1)
    for i in range(0, l // chunk_size):
        q_i, k_i = q[:, :, i], k[:, :, i]
        attn = (q_i @ k_i.transpose(-1, -2)).masked_fill_(mask, 0)
        u_i = u[:, :, i] - w[:, :, i] @ S
        o_inter = q_i @ S
        o[:, :, i] = o_inter + attn @ u_i
        S = S + k_i.transpose(-1, -2) @ u_i

    return rearrange(o, 'b h n c d -> b h (n c) d'), S


def delta_rule_parallel(q, k, v, beta, BM=128, BN=32):
    b, h, l, d_k = q.shape
    # d_v = v.shape[-1]
    q = q * (d_k ** -0.5)
    v = v * beta[..., None]
    k_beta = k * beta[..., None]
    # compute (I - tri(diag(beta) KK^T))^{-1}
    q, k, v, k_beta = map(lambda x: rearrange(x, 'b h (n c) d -> b h n c d', c=BN), [q, k, v, k_beta])
    mask = torch.triu(torch.ones(BN, BN, dtype=torch.bool, device=q.device), diagonal=0)
    T = -(k_beta @ k.transpose(-1, -2)).masked_fill(mask, 0)
    for i in range(1, BN):
        T[..., i, :i] = T[..., i, :i].clone() + (T[..., i, :, None].clone() * T[..., :, :i].clone()).sum(-2)
    T = T + torch.eye(BN, dtype=torch.float, device=q.device)

    mask2 = torch.triu(torch.ones(BN, BN, dtype=torch.bool, device=q.device), diagonal=1)
    A_local = (q @ k.transpose(-1, -2)).masked_fill(mask2, 0) @ T
    o_intra = A_local @ v

    # apply cumprod transition matrices on k to the last position within the chunk
    k = k - ((k @ k.transpose(-1, -2)).masked_fill(mask, 0) @ T).transpose(-1, -2) @ k_beta
    # apply cumprod transition matrices on q to the first position within the chunk
    q = q - A_local @ k_beta
    o_intra = A_local @ v

    A = torch.zeros(b, h, l, l, device=q.device)

    q, k, v, k_beta, o_intra = map(lambda x: rearrange(x, 'b h n c d -> b h (n c) d'), [q, k, v, k_beta, o_intra])
    o = torch.empty_like(v)
    for i in range(0, l, BM):
        q_i = q[:, :, i:i+BM]
        o_i = o_intra[:, :, i:i+BM]
        # intra block
        for j in range(i + BM - 2 * BN, i-BN, -BN):
            k_j = k[:, :, j:j+BN]
            A_ij = q_i @ k_j.transpose(-1, -2)
            mask = torch.arange(i, i+BM) >= (j + BN)
            A_ij = A_ij.masked_fill_(~mask[:, None].to(A_ij.device), 0)
            A[:, :, i:i+BM, j:j+BN] = A_ij
            q_i = q_i - A_ij @ k_beta[:, :, j:j+BN]
            o_i += A_ij @ v[:, :, j:j+BN]
        # inter block
        for j in range(i - BN, -BN, -BN):
            k_j = k[:, :, j:j+BN]
            A_ij = q_i @ k_j.transpose(-1, -2)
            A[:, :, i:i+BM, j:j+BN] = A_ij
            q_i = q_i - A_ij @ k_beta[:, :, j:j+BN]
            o_i += A_ij @ v[:, :, j:j+BN]
        o[:, :, i:i+BM] = o_i

    for i in range(0, l//BN):
        A[:, :, i*BN:i*BN+BN, i*BN:i*BN+BN] = A_local[:, :, i]

    return o, A


def delta_rule_parallel_scan(q, k, v, beta, initial_state=None, output_final_state=True):
    """
    Parallel scan implementation using Hillis-Steele algorithm.

    The delta rule recurrence S_t = S_{t-1} + k_t ⊗ (β_t * (v_t - k_t^T @ S_{t-1}))
    expands to a linear recurrence with matrix coefficients:
        S_t = A_t @ S_{t-1} + b_t

    where:
        A_t = I - β_t * k_t @ k_t^T   (rank-1 perturbation of identity)
        b_t = β_t * k_t @ v_t^T

    The associative operation for parallel scan is:
        (A₂, b₂) ⊕ (A₁, b₁) = (A₂ @ A₁, A₂ @ b₁ + b₂)

    Complexity:
        Time: O(log L) parallel depth, O(L * d_k^2 * d_v) total work
        Space: O(L * d_k^2 + L * d_k * d_v) for A and b matrices
    """
    orig_dtype = q.dtype
    device = q.device
    B, H, L, d_k = q.shape
    d_v = v.shape[-1]

    q_scaled = q.float() * (d_k ** -0.5)
    k = k.float()
    v = v.float()
    beta = beta.float()

    if beta.ndim < v.ndim:
        beta = beta[..., None]

    I = torch.eye(d_k, dtype=torch.float32, device=device)

    # Build A_t = I - β_t * k_t @ k_t^T and b_t = β_t * k_t @ v_t^T
    k_outer = k.unsqueeze(-1) * k.unsqueeze(-2)  # (B, H, L, d_k, d_k)
    A = I - beta.unsqueeze(-1) * k_outer  # (B, H, L, d_k, d_k)
    b_mat = beta.unsqueeze(-1) * k.unsqueeze(-1) * v.unsqueeze(-2)  # (B, H, L, d_k, d_v)

    # Hillis-Steele parallel prefix scan
    A_scan = A.clone()
    b_scan = b_mat.clone()

    step = 1
    while step < L:
        # Create shifted versions (pad with identity for A, zeros for b)
        A_shifted = torch.cat([
            I.expand(B, H, step, d_k, d_k),
            A_scan[:, :, :-step]
        ], dim=2)
        b_shifted = torch.cat([
            torch.zeros(B, H, step, d_k, d_v, dtype=torch.float32, device=device),
            b_scan[:, :, :-step]
        ], dim=2)

        # Combine: (A_scan, b_scan) ⊕ (A_shifted, b_shifted)
        A_new = torch.einsum('bhlij,bhljk->bhlik', A_scan, A_shifted)
        b_new = torch.einsum('bhlij,bhljk->bhlik', A_scan, b_shifted) + b_scan

        A_scan = A_new
        b_scan = b_new
        step *= 2

    # b_scan contains the prefix scan results (S_t for each t)
    S_all = b_scan

    # Handle initial state
    if initial_state is not None:
        S_all = torch.einsum('bhlij,bhjk->bhlik', A_scan, initial_state.unsqueeze(2).float()) + b_scan

    # Compute output o_t = q_t @ S_t
    o = torch.einsum('bhld,bhldv->bhlv', q_scaled, S_all)

    S_final = S_all[:, :, -1] if output_final_state else None
    return o.to(orig_dtype), S_final


# =============================================================================
# Native dtype implementations (preserve fp16/bf16 precision)
# =============================================================================

def delta_rule_recurrence_native_dtype(q, k, v, beta, initial_state=None, output_final_state=True):
    """
    Sequential recurrence that preserves input dtype (fp16/bf16).

    Unlike delta_rule_recurrence which casts to float32 internally,
    this version computes entirely in the input dtype to study
    precision effects.
    """
    dtype = q.dtype
    device = q.device
    b, h, l, d_k = q.shape
    d_v = v.shape[-1]

    o = torch.zeros(b, h, l, d_v, dtype=dtype, device=device)
    S = torch.zeros(b, h, d_k, d_v, dtype=dtype, device=device)
    q = q * (d_k ** -0.5)

    if beta.ndim < v.ndim:
        beta = beta[..., None]

    if initial_state is not None:
        S = S + initial_state

    for i in range(l):
        _k = k[:, :, i]
        _q = q[:, :, i]
        _v = v[:, :, i].clone()
        beta_i = beta[:, :, i]
        _v = _v - (S.clone() * _k[..., None]).sum(-2)
        _v = _v * beta_i
        S = S.clone() + _k.unsqueeze(-1) * _v.unsqueeze(-2)
        o[:, :, i] = torch.einsum('bhd,bhdm->bhm', _q, S)

    S = None if output_final_state is False else S
    return o, S


def delta_rule_chunkwise_native_dtype(q, k, v, beta, chunk_size=32):
    """
    Chunkwise implementation that preserves input dtype (fp16/bf16).

    Unlike delta_rule_chunkwise which uses float32 for some operations,
    this version computes entirely in the input dtype.
    """
    dtype = q.dtype
    device = q.device
    b, h, l, d_k = q.shape
    d_v = v.shape[-1]

    q = q * (d_k ** -0.5)
    v = v * beta[..., None]
    k_beta = k * beta[..., None]

    assert l % chunk_size == 0

    # compute (I - tri(diag(beta) KK^T))^{-1}
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=device), diagonal=0)
    q, k, v, k_beta = map(lambda x: rearrange(x, 'b h (n c) d -> b h n c d', c=chunk_size), [q, k, v, k_beta])
    attn = -(k_beta @ k.transpose(-1, -2)).masked_fill(mask, 0)
    for i in range(1, chunk_size):
        attn[..., i, :i] = attn[..., i, :i] + (attn[..., i, :, None].clone() * attn[..., :, :i].clone()).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=dtype, device=device)

    u = attn @ v
    w = attn @ k_beta
    S = torch.zeros(b, h, d_k, d_v, dtype=dtype, device=device)
    o = torch.zeros_like(v)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=device), diagonal=1)
    for i in range(0, l // chunk_size):
        q_i, k_i = q[:, :, i], k[:, :, i]
        attn = (q_i @ k_i.transpose(-1, -2)).masked_fill_(mask, 0)
        u_i = u[:, :, i] - w[:, :, i] @ S
        o_inter = q_i @ S
        o[:, :, i] = o_inter + attn @ u_i
        S = S + k_i.transpose(-1, -2) @ u_i

    return rearrange(o, 'b h n c d -> b h (n c) d'), S


def delta_rule_parallel_native_dtype(q, k, v, beta, BM=128, BN=32):
    """
    Parallel implementation that preserves input dtype (fp16/bf16).

    Unlike delta_rule_parallel which uses float32 for some operations,
    this version computes entirely in the input dtype.
    """
    dtype = q.dtype
    device = q.device
    b, h, l, d_k = q.shape

    q = q * (d_k ** -0.5)
    v = v * beta[..., None]
    k_beta = k * beta[..., None]

    # compute (I - tri(diag(beta) KK^T))^{-1}
    q, k, v, k_beta = map(lambda x: rearrange(x, 'b h (n c) d -> b h n c d', c=BN), [q, k, v, k_beta])
    mask = torch.triu(torch.ones(BN, BN, dtype=torch.bool, device=device), diagonal=0)
    T = -(k_beta @ k.transpose(-1, -2)).masked_fill(mask, 0)
    for i in range(1, BN):
        T[..., i, :i] = T[..., i, :i].clone() + (T[..., i, :, None].clone() * T[..., :, :i].clone()).sum(-2)
    T = T + torch.eye(BN, dtype=dtype, device=device)

    mask2 = torch.triu(torch.ones(BN, BN, dtype=torch.bool, device=device), diagonal=1)
    A_local = (q @ k.transpose(-1, -2)).masked_fill(mask2, 0) @ T
    o_intra = A_local @ v

    # apply cumprod transition matrices on k to the last position within the chunk
    k = k - ((k @ k.transpose(-1, -2)).masked_fill(mask, 0) @ T).transpose(-1, -2) @ k_beta
    # apply cumprod transition matrices on q to the first position within the chunk
    q = q - A_local @ k_beta
    o_intra = A_local @ v

    A = torch.zeros(b, h, l, l, dtype=dtype, device=device)

    q, k, v, k_beta, o_intra = map(lambda x: rearrange(x, 'b h n c d -> b h (n c) d'), [q, k, v, k_beta, o_intra])
    o = torch.empty_like(v)
    for i in range(0, l, BM):
        q_i = q[:, :, i:i+BM]
        o_i = o_intra[:, :, i:i+BM]
        # intra block
        for j in range(i + BM - 2 * BN, i-BN, -BN):
            k_j = k[:, :, j:j+BN]
            A_ij = q_i @ k_j.transpose(-1, -2)
            mask = torch.arange(i, i+BM, device=device) >= (j + BN)
            A_ij = A_ij.masked_fill_(~mask[:, None], 0)
            A[:, :, i:i+BM, j:j+BN] = A_ij
            q_i = q_i - A_ij @ k_beta[:, :, j:j+BN]
            o_i = o_i + A_ij @ v[:, :, j:j+BN]
        # inter block
        for j in range(i - BN, -BN, -BN):
            k_j = k[:, :, j:j+BN]
            A_ij = q_i @ k_j.transpose(-1, -2)
            A[:, :, i:i+BM, j:j+BN] = A_ij
            q_i = q_i - A_ij @ k_beta[:, :, j:j+BN]
            o_i = o_i + A_ij @ v[:, :, j:j+BN]
        o[:, :, i:i+BM] = o_i

    for i in range(0, l//BN):
        A[:, :, i*BN:i*BN+BN, i*BN:i*BN+BN] = A_local[:, :, i]

    return o, A


def delta_rule_parallel_scan_native_dtype(q, k, v, beta, initial_state=None, output_final_state=True):
    """
    Parallel scan (Hillis-Steele) that preserves input dtype (fp16/bf16).

    Unlike delta_rule_parallel_scan which casts to float32 internally,
    this version computes entirely in the input dtype to study
    precision effects.

    The recurrence S_t = A_t @ S_{t-1} + b_t is computed with:
        A_t = I - β_t * k_t @ k_t^T
        b_t = β_t * k_t @ v_t^T

    Complexity:
        Time: O(log L) parallel depth
        Space: O(L * d_k^2 + L * d_k * d_v)
    """
    dtype = q.dtype
    device = q.device
    B, H, L, d_k = q.shape
    d_v = v.shape[-1]

    q_scaled = q * (d_k ** -0.5)

    if beta.ndim < v.ndim:
        beta = beta[..., None]

    I = torch.eye(d_k, dtype=dtype, device=device)

    # Build A_t = I - β_t * k_t @ k_t^T and b_t = β_t * k_t @ v_t^T
    k_outer = k.unsqueeze(-1) * k.unsqueeze(-2)  # (B, H, L, d_k, d_k)
    A = I - beta.unsqueeze(-1) * k_outer  # (B, H, L, d_k, d_k)
    b_mat = beta.unsqueeze(-1) * k.unsqueeze(-1) * v.unsqueeze(-2)  # (B, H, L, d_k, d_v)

    # Hillis-Steele parallel prefix scan
    A_scan = A.clone()
    b_scan = b_mat.clone()

    step = 1
    while step < L:
        # Create shifted versions (pad with identity for A, zeros for b)
        A_shifted = torch.cat([
            I.expand(B, H, step, d_k, d_k),
            A_scan[:, :, :-step]
        ], dim=2)
        b_shifted = torch.cat([
            torch.zeros(B, H, step, d_k, d_v, dtype=dtype, device=device),
            b_scan[:, :, :-step]
        ], dim=2)

        # Combine: (A_scan, b_scan) ⊕ (A_shifted, b_shifted)
        A_new = torch.einsum('bhlij,bhljk->bhlik', A_scan, A_shifted)
        b_new = torch.einsum('bhlij,bhljk->bhlik', A_scan, b_shifted) + b_scan

        A_scan = A_new
        b_scan = b_new
        step *= 2

    # b_scan contains the prefix scan results (S_t for each t)
    S_all = b_scan

    # Handle initial state
    if initial_state is not None:
        S_all = torch.einsum('bhlij,bhjk->bhlik', A_scan, initial_state.unsqueeze(2)) + b_scan

    # Compute output o_t = q_t @ S_t
    o = torch.einsum('bhld,bhldv->bhlv', q_scaled, S_all)

    S_final = S_all[:, :, -1] if output_final_state else None
    return o, S_final
