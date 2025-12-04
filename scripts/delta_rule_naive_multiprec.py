"""
Multi-precision naive implementations of delta rule.
These preserve the input dtype throughout computation (fp32, fp16, bf16).
"""

import torch
from einops import rearrange


def delta_rule_recurrence(q, k, v, beta, initial_state=None, output_final_state=True):
    """
    Step-by-step recurrence implementation.
    Preserves input dtype throughout computation.

    Args:
        q: (batch, heads, seq_len, d_k)
        k: (batch, heads, seq_len, d_k)
        v: (batch, heads, seq_len, d_v)
        beta: (batch, heads, seq_len) or (batch, heads, seq_len, 1)
        initial_state: optional (batch, heads, d_k, d_v)
        output_final_state: whether to return final state

    Returns:
        o: (batch, heads, seq_len, d_v)
        S: (batch, heads, d_k, d_v) or None
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


def delta_rule_chunkwise(q, k, v, beta, chunk_size=32):
    """
    Chunkwise implementation.
    Preserves input dtype throughout computation.

    Args:
        q: (batch, heads, seq_len, d_k)
        k: (batch, heads, seq_len, d_k)
        v: (batch, heads, seq_len, d_v)
        beta: (batch, heads, seq_len)
        chunk_size: size of each chunk

    Returns:
        o: (batch, heads, seq_len, d_v)
        S: (batch, heads, d_k, d_v)
    """
    dtype = q.dtype
    device = q.device
    b, h, l, d_k = q.shape
    d_v = v.shape[-1]

    q = q * (d_k ** -0.5)
    v = v * beta[..., None]
    k_beta = k * beta[..., None]

    assert l % chunk_size == 0, f"seq_len {l} must be divisible by chunk_size {chunk_size}"

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


def delta_rule_parallel(q, k, v, beta, BM=128, BN=32):
    """
    Parallel implementation.
    Preserves input dtype throughout computation.

    Args:
        q: (batch, heads, seq_len, d_k)
        k: (batch, heads, seq_len, d_k)
        v: (batch, heads, seq_len, d_v)
        beta: (batch, heads, seq_len)
        BM: block size M
        BN: block size N

    Returns:
        o: (batch, heads, seq_len, d_v)
        A: (batch, heads, seq_len, seq_len) attention matrix
    """
    dtype = q.dtype
    device = q.device
    b, h, l, d_k = q.shape

    q = q * (d_k ** -0.5)
    v = v * beta[..., None]
    k_beta = k * beta[..., None]

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
