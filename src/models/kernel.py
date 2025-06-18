import math

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


# Disabling autotune for now, set num_warps=4 if headdim=64 and num_warps=8 if headdim=128
# @triton.autotune(
#     configs=[
#         triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=4, num_stages=1),
#         # This config has a race condition when EVEN_M == False, disabling it for now.
#         # triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=1),
#     ],
#     key=['CACHE_KEY_SEQLEN_Q', 'CACHE_KEY_SEQLEN_K', 'BIAS_TYPE', 'IS_CAUSAL', 'BLOCK_HEADDIM']
# )
@triton.heuristics(
    {
        "EVEN_M": lambda args: args["seqlen_q"] % args["BLOCK_M"] == 0,
        "EVEN_N": lambda args: args["seqlen_k"] % args["BLOCK_N"] == 0,
        "EVEN_HEADDIM": lambda args: args["headdim"] == args["BLOCK_HEADDIM"],
    }
)
@triton.jit
def _fwd_kernel(
    Q,
    K,
    V,
    Out,
    Lse,
    TMP,  # NOTE: TMP is a scratchpad buffer to workaround a compiler bug
    softmax_scale,
    stride_qb,
    stride_qh,
    stride_qm,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_vb,
    stride_vh,
    stride_vn,
    stride_ob,
    stride_oh,
    stride_om,
    nheads,
    seqlen_q,
    seqlen_k,
    seqlen_q_rounded,
    headdim,
    CACHE_KEY_SEQLEN_Q,
    CACHE_KEY_SEQLEN_K,
    IS_CAUSAL: tl.constexpr,
    BLOCK_HEADDIM: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_HEADDIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_b = tl.program_id(1)
    off_h = tl.program_id(2)
    off_hb = off_b * nheads + off_h

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)

    offs_d = tl.arange(0, BLOCK_HEADDIM)
    q_ptrs = (
        Q + off_b * stride_qb + off_h * stride_qh + (offs_m[:, None] * stride_qm + offs_d[None, :])
    )
    k_ptrs = (
        K + off_b * stride_kb + off_h * stride_kh + (offs_n[:, None] * stride_kn + offs_d[None, :])
    )
    v_ptrs = (
        V + off_b * stride_vb + off_h * stride_vh + (offs_n * stride_vn)[:, None]
    )

    lse_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    acc_o = tl.zeros([BLOCK_M, 1], dtype=tl.float32)

    # Load Q
    if EVEN_M & EVEN_N:
        if EVEN_HEADDIM:
            q = tl.load(q_ptrs)
        else:
            q = tl.load(q_ptrs, mask=offs_d[None, :] < headdim, other=0.0)
    else:
        if EVEN_HEADDIM:
            q = tl.load(q_ptrs, mask=offs_m[:, None] < seqlen_q, other=0.0)
        else:
            q = tl.load(
                q_ptrs, mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < headdim), other=0.0
            )
    end_n = seqlen_k if not IS_CAUSAL else tl.minimum((start_m + 1) * BLOCK_M, seqlen_k)

    for start_n in range(0, end_n, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)

        if EVEN_N & EVEN_M:  # If we just do "if EVEN_N", there seems to be some race condition
            if EVEN_HEADDIM:
                k = tl.load(k_ptrs + start_n * stride_kn)
            else:
                k = tl.load(k_ptrs + start_n * stride_kn, mask=offs_d[None, :] < headdim, other=0.0)
        else:
            if EVEN_HEADDIM:
                k = tl.load(
                    k_ptrs + start_n * stride_kn,
                    mask=(start_n + offs_n)[:, None] < seqlen_k,
                    other=0.0,
                )
            else:
                k = tl.load(
                    k_ptrs + start_n * stride_kn,
                    mask=((start_n + offs_n)[:, None] < seqlen_k) & (offs_d[None, :] < headdim),
                    other=0.0,
                )
        qk = tl.dot(q, tl.trans(k), input_precision="tf32x3")
        if not EVEN_N:
            qk += tl.where((start_n + offs_n)[None, :] < seqlen_k, 0, float("-inf"))
        if IS_CAUSAL:
            qk += tl.where(offs_m[:, None] >= (start_n + offs_n)[None, :], 0, float("-inf"))

        m_ij = tl.maximum(tl.max(qk, 1) * softmax_scale, lse_i)
        p = tl.exp(qk * softmax_scale - m_ij[:, None])
        l_ij = tl.sum(p, 1)

        acc_o_scale = tl.exp(m_i - m_ij)
        acc_o = acc_o * acc_o_scale[:, None]

        if EVEN_N & EVEN_M:
            v = tl.load(v_ptrs + start_n * stride_vn)
        else:
            v = tl.load(
                v_ptrs + start_n * stride_vn,
                mask=(start_n + offs_n)[:, None] < seqlen_k,
                other=0.0,
            )

        p = p.to(v.dtype)
        acc_o += tl.sum(p * v.reshape([1, BLOCK_N]), axis=1, keep_dims=True)

        m_i = m_ij
        l_i_new = tl.exp(lse_i - m_ij) + l_ij
        lse_i = m_ij + tl.log(l_i_new)

    o_scale = tl.exp(m_i - lse_i)
    acc_o = acc_o * o_scale[:, None]

    start_m = tl.program_id(0)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)

    lse_ptrs = Lse + off_hb * seqlen_q_rounded + offs_m
    tl.store(lse_ptrs, lse_i)

    out_ptrs = (
        Out + off_b * stride_ob + off_h * stride_oh + (offs_m * stride_om)[:, None]
    )

    if EVEN_M:
        tl.store(out_ptrs, acc_o)
    else:
        tl.store(out_ptrs, acc_o, mask=offs_m[:, None] < seqlen_q)

@triton.jit
def _topk_bwd_preprocess_do_o_dot(
    Out,
    DO,
    Delta,
    TopkIndices,
    stride_ob,
    stride_oh,
    stride_oq,
    stride_dob,
    stride_doh,
    stride_doq,
    stride_topkindicesb,
    stride_topkindicesi,
    stride_topkindicesq,
    stride_deltab,
    stride_deltah,
    stride_deltaq,
    nheads,
    seqlen_q,
    seqlen_q_rounded,
    headdim,
    TOPK,
):
    for off_i in range(TOPK):
        off_b = tl.program_id(0)
        off_q = tl.program_id(1)
        indices = tl.load(TopkIndices + off_b * stride_topkindicesb + off_q * stride_topkindicesq + off_i * stride_topkindicesi)
        o = tl.load(
            Out + off_b * stride_ob + off_q * stride_oq + indices * stride_oh,
        ).to(tl.float32)
        do = tl.load(
            DO + off_b * stride_dob + off_q * stride_doq + indices * stride_doh,
        ).to(tl.float32)
        delta = o * do
        tl.store(Delta + off_b * stride_deltab + off_q * stride_deltaq + indices * stride_deltah, delta)

def init_to_zero(*names):
    def hook(nargs):
        for name in names:
            nargs[name].zero_()
    return hook

@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_N": 128},
            num_warps=8,
            num_stages=1,
            pre_hook=init_to_zero("DK", "DV"),
        )
    ],
    key=["CACHE_KEY_SEQLEN_Q", "CACHE_KEY_SEQLEN_K", "IS_CAUSAL", "BLOCK_HEADDIM"],
)
@triton.jit
def _topk_bwd_kernel(
    Q,
    K,
    V,
    DO,
    DQ,
    DK,
    DV,
    LSE,
    D,
    TopkIndices,
    softmax_scale,
    stride_qb,
    stride_qh,
    stride_qq,
    stride_kb,
    stride_kh,
    stride_kk,
    stride_vb,
    stride_vh,
    stride_vk,
    stride_dob,
    stride_doh,
    stride_doq,
    stride_dqb,
    stride_dqh,
    stride_dqq,
    stride_dkb,
    stride_dkh,
    stride_dkk,
    stride_dvb,
    stride_dvh,
    stride_dvk,
    stride_topkindicesb,
    stride_topkindicesi,
    stride_topkindicesq,
    stride_deltab,
    stride_deltah,
    stride_deltaq,
    nheads,
    seqlen_q,
    seqlen_k,
    seqlen_q_rounded,
    headdim,
    CACHE_KEY_SEQLEN_Q,
    CACHE_KEY_SEQLEN_K,
    TOPK: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    BLOCK_HEADDIM: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    for off_i in range(TOPK):
        begin_n = 0
        
        off_b = tl.program_id(1)
        off_q = tl.program_id(0)
        # off_i = tl.program_id(2)
        off_h = tl.load(TopkIndices + stride_topkindicesb * off_b + stride_topkindicesq * off_q + stride_topkindicesi * off_i)
        
        offs_n = tl.arange(0, BLOCK_N)
        offs_d = tl.arange(0, BLOCK_HEADDIM)
        
        q_ptrs = Q + stride_qb * off_b + stride_qh * off_h + stride_qq * off_q + (offs_d[None, :])
        k_ptrs = K + stride_kb * off_b + stride_kh * off_h + (stride_kk * offs_n[:, None] + offs_d[None, :])
        v_ptrs = V + stride_vb * off_b + stride_vh * off_h + (stride_vk * offs_n[:, None])
        do_ptrs = DO + stride_dob * off_b + stride_doh * off_h + stride_doq * off_q
        dq_ptrs = DQ + stride_dqb * off_b + stride_dqh * off_h + stride_dqq * off_q + (offs_d[None, :])
        dk_ptrs = DK + stride_dkb * off_b + stride_dkh * off_h + (stride_dkk * offs_n[:, None] + offs_d[None, :])
        dv_ptrs = DV + stride_dvb * off_b + stride_dvh * off_h + (stride_dvk * offs_n[:, None])
        Di_ptr = D + stride_deltab * off_b + stride_deltah * off_h + stride_deltaq * off_q
        lse_i_ptr = LSE + stride_deltab * off_b + stride_deltah * off_h + stride_deltaq * off_q
        
        dq = tl.zeros([1, BLOCK_HEADDIM], dtype=tl.float32)
        
        q = tl.load(q_ptrs)
        do = tl.load(do_ptrs)
        
        Di = tl.load(Di_ptr)
        lse_i = tl.load(lse_i_ptr)
        
        # loop over rows
        num_block_n = tl.cdiv(seqlen_k, BLOCK_N)
        for start_n in range(begin_n, num_block_n * BLOCK_N, BLOCK_N):
            start_n = tl.multiple_of(start_n, BLOCK_N)
            offs_n_curr = start_n + offs_n
            # load q, k, v, do on-chip
            # Same bug as below. Otherwise gives wrong result for headdim=40, seqlen=(128, 117)
            k = tl.load(k_ptrs)
            v = tl.load(v_ptrs)
            # recompute p = softmax(qk, dim=-1).T
            # qk = tl.dot(q, tl.trans(k))
            qk = tl.sum(q * k, axis=1).reshape([1, BLOCK_N])
            # Trying to combine the two masks seem to make the result wrong
            if IS_CAUSAL:
                qk = tl.where(off_q[:, None] >= (offs_n_curr[None, :]), qk, float("-inf"))
            # There seems to be a race condition when headdim=48/96, and dq, dk, dv are wrong.
            # Also wrong for headdim=64.
            p = tl.exp(qk * softmax_scale - lse_i[:, None])
            # dp = tl.dot(do, tl.trans(v))
            dp = do * tl.trans(v)
            # There's a race condition for headdim=48
            # compute ds = p * (dp - delta[:, None])
            # Putting the subtraction after the dp matmul (instead of before) is slightly faster
            # Converting ds to q.dtype here reduces register pressure and makes it much faster
            # for BLOCK_HEADDIM=128
            ds = (p * (dp - Di[:, None]) * softmax_scale).to(q.dtype)
            # compute dq
            # dq += tl.dot(ds, k)
            dq += tl.sum(tl.trans(ds) * k, axis=0, keep_dims=True)
            # compute dk, dv
            # dk = tl.dot(tl.trans(ds), q)
            dk = tl.trans(ds) * q
            dv = tl.sum(p.to(do.dtype) * do, axis=0).reshape([BLOCK_N, 1])
            tl.atomic_add(dk_ptrs, dk)
            tl.atomic_add(dv_ptrs, dv)
            # increment pointers
            dk_ptrs += BLOCK_N * stride_dkk
            k_ptrs += BLOCK_N * stride_kk
            dv_ptrs += BLOCK_N * stride_dvk
            v_ptrs += BLOCK_N * stride_vk
            
        # write-back
        tl.store(dq_ptrs, dq)

def _flash_attn_forward(q, k, v, causal=False, softmax_scale=None):
    # shape constraints
    batch, seqlen_q, nheads, d = q.shape
    _, seqlen_k, _, _ = k.shape
    assert k.shape == (batch, seqlen_k, nheads, d)
    # assert v.shape == (batch, seqlen_k, nheads, 1)
    assert d <= 128, "FlashAttention only support head dimensions up to 128"
    assert q.dtype == k.dtype == v.dtype, "All tensors must have the same type"
    # assert q.dtype in [torch.float16, torch.bfloat16], "Only support fp16 and bf16"
    assert q.is_cuda and k.is_cuda and v.is_cuda
    softmax_scale = softmax_scale or 1.0 / math.sqrt(d)

    seqlen_q_rounded = math.ceil(seqlen_q / 128) * 128
    lse = torch.empty((batch, nheads, seqlen_q_rounded), device=q.device, dtype=torch.float32)
    tmp = torch.empty((batch, nheads, seqlen_q_rounded), device=q.device, dtype=torch.float32)
    o = torch.empty_like(v)

    BLOCK_HEADDIM = max(triton.next_power_of_2(d), 16)
    BLOCK = 128
    num_warps = 4 if d <= 64 else 8
    # if batch_size=32, it will rise some error
    grid = lambda META: (triton.cdiv(seqlen_q, META["BLOCK_M"]), batch, nheads)
    _fwd_kernel[grid](
        q,
        k,
        v,
        o,
        lse,
        tmp,
        softmax_scale,
        q.stride(0),
        q.stride(2),
        q.stride(1),
        k.stride(0),
        k.stride(2),
        k.stride(1),
        v.stride(0),
        v.stride(2),
        v.stride(1),
        o.stride(0),
        o.stride(2),
        o.stride(1),
        nheads,
        seqlen_q,
        seqlen_k,
        seqlen_q_rounded,
        d,
        seqlen_q // 32,
        seqlen_k // 32,  # key for triton cache (limit number of compilations)
        # Can't use kwargs here because triton autotune expects key to be args, not kwargs
        # IS_CAUSAL=causal, BLOCK_HEADDIM=d,
        causal,
        BLOCK_HEADDIM,
        BLOCK_M=BLOCK,
        BLOCK_N=BLOCK,
        num_warps=num_warps,
        num_stages=1,
    )
    return o, lse, softmax_scale  # softmax_scale could have been updated


def _topk_flash_attn_backward(
    do, q, k, v, o, lse, topk_indices, dq, dk, dv, causal=False, softmax_scale=None
):
    # Make sure that the last dimension is contiguous
    if do.stride(-1) != 1:
        do = do.contiguous()
    batch, seqlen_q, nheads, d = q.shape
    _, seqlen_k, _, _ = k.shape
    _, _, TOPK = topk_indices.shape
    assert d <= 128
    seqlen_q_rounded = math.ceil(seqlen_q / 128) * 128
    assert lse.shape == (batch, nheads, seqlen_q_rounded)
    assert q.stride(-1) == k.stride(-1) == v.stride(-1) == o.stride(-1) == 1
    assert dq.stride(-1) == dk.stride(-1) == dv.stride(-1) == 1
    softmax_scale = softmax_scale or 1.0 / math.sqrt(d)
    dk_accum = torch.zeros_like(k, dtype=torch.float32)
    dv_accum = torch.zeros_like(v, dtype=torch.float32)
    delta = torch.zeros_like(lse)

    BLOCK_HEADDIM = max(triton.next_power_of_2(d), 16)
    grid = lambda META: (batch, seqlen_q)
    _topk_bwd_preprocess_do_o_dot[grid](
        o,
        do,
        delta,
        topk_indices,
        o.stride(0),
        o.stride(2),
        o.stride(1),
        do.stride(0),
        do.stride(2),
        do.stride(1),
        topk_indices.stride(0),
        topk_indices.stride(2),
        topk_indices.stride(1),
        delta.stride(0),
        delta.stride(1),
        delta.stride(2),
        nheads,
        seqlen_q,
        seqlen_q_rounded,
        d,
        TOPK,
    )
    # BLOCK_M = 128
    # BLOCK_N = 64
    # num_warps = 4
    grid = lambda META: (
        seqlen_q,
        batch,
    )
    _topk_bwd_kernel[grid](
        q,
        k,
        v,
        do,
        dq,
        dk_accum,
        dv_accum,
        lse,
        delta,
        topk_indices,
        softmax_scale,
        q.stride(0),
        q.stride(2),
        q.stride(1),
        k.stride(0),
        k.stride(2),
        k.stride(1),
        v.stride(0),
        v.stride(2),
        v.stride(1),
        do.stride(0),
        do.stride(2),
        do.stride(1),
        dq.stride(0),
        dq.stride(2),
        dq.stride(1),
        dk_accum.stride(0),
        dk_accum.stride(2),
        dk_accum.stride(1),
        dv_accum.stride(0),
        dv_accum.stride(2),
        dv_accum.stride(1),
        topk_indices.stride(0),
        topk_indices.stride(2),
        topk_indices.stride(1),
        delta.stride(0),
        delta.stride(1),
        delta.stride(2),
        nheads,
        seqlen_q,
        seqlen_k,
        seqlen_q_rounded,
        d,
        seqlen_q // 32,
        seqlen_k // 32,  # key for triton cache (limit number of compilations)
        # Can't use kwargs here because triton autotune expects key to be args, not kwargs
        # IS_CAUSAL=causal, BLOCK_HEADDIM=d,
        TOPK, # TOPK
        causal,
        BLOCK_HEADDIM,
        # SEQUENCE_PARALLEL=False,
        # BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        # num_warps=num_warps,
        # num_stages=1,
    )
    dk.copy_(dk_accum)
    dv.copy_(dv_accum)
    

@triton.jit
def _bwd_preprocess_do_o_dot(
    Out,
    DO,
    Delta,
    stride_ob,
    stride_oh,
    stride_om,
    stride_dob,
    stride_doh,
    stride_dom,
    nheads,
    seqlen_q,
    seqlen_q_rounded,
    headdim,
    BLOCK_M: tl.constexpr,
    BLOCK_HEADDIM: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hb = tl.program_id(1)
    off_b = off_hb // nheads
    off_h = off_hb % nheads
    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_HEADDIM)
    # load
    o = tl.load(
        Out + off_b * stride_ob + off_h * stride_oh + (offs_m * stride_om)[:, None],
        mask=(offs_m[:, None] < seqlen_q),
        other=0.0,
    ).to(tl.float32)
    do = tl.load(
        DO
        + off_b * stride_dob
        + off_h * stride_doh
        + (offs_m * stride_om)[:, None],
        mask=(offs_m[:, None] < seqlen_q),
        other=0.0,
    ).to(tl.float32)
    delta = tl.sum(o * do, axis=1)
    # write-back
    tl.store(Delta + off_hb * seqlen_q_rounded + offs_m, delta)


@triton.jit
def _bwd_store_dk_dv(
    dk_ptrs,
    dv_ptrs,
    dk,
    dv,
    offs_n,
    offs_d,
    seqlen_k,
    headdim,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_HEADDIM: tl.constexpr,
):
    # [2022-11-01] TD: Same bug. In the case of EVEN_N=True and EVEN_M=False,
    # if we just call tl.store(dv_ptrs), there's a race condition
    if EVEN_N & EVEN_M:
        if EVEN_HEADDIM:
            tl.store(dv_ptrs, dv)
            tl.store(dk_ptrs, dk)
        else:
            tl.store(dv_ptrs, dv)
            tl.store(dk_ptrs, dk, mask=offs_d[None, :] < headdim)
    else:
        if EVEN_HEADDIM:
            tl.store(dv_ptrs, dv, mask=offs_n[:, None] < seqlen_k)
            tl.store(dk_ptrs, dk, mask=offs_n[:, None] < seqlen_k)
        else:
            tl.store(dv_ptrs, dv, mask=(offs_n[:, None] < seqlen_k))
            tl.store(dk_ptrs, dk, mask=(offs_n[:, None] < seqlen_k) & (offs_d[None, :] < headdim))


@triton.jit
def _bwd_kernel_one_col_block(
    start_n,
    Q,
    K,
    V,
    DO,
    DQ,
    DK,
    DV,
    LSE,
    D,
    softmax_scale,
    stride_qm,
    stride_kn,
    stride_vn,
    stride_dom,
    stride_dqm,
    stride_dkn,
    stride_dvn,
    seqlen_q,
    seqlen_k,
    headdim,
    ATOMIC_ADD: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    BLOCK_HEADDIM: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_HEADDIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # We need to make sure begin_m is a multiple of BLOCK_M (not BLOCK_N)
    begin_m = 0 if not IS_CAUSAL else ((start_n * BLOCK_N) // BLOCK_M) * BLOCK_M
    # initialize row/col offsets
    offs_qm = begin_m + tl.arange(0, BLOCK_M)
    offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_HEADDIM)
    # initialize pointers to value-like data
    q_ptrs = Q + (offs_qm[:, None] * stride_qm + offs_d[None, :])
    k_ptrs = K + (offs_n[:, None] * stride_kn + offs_d[None, :])
    v_ptrs = V + (offs_n[:, None] * stride_vn)
    do_ptrs = DO + (offs_qm[:, None] * stride_dom)
    dq_ptrs = DQ + (offs_qm[:, None] * stride_dqm + offs_d[None, :])
    # initialize dv and dk
    dv = tl.zeros([BLOCK_N, 1], dtype=tl.float32)
    dk = tl.zeros([BLOCK_N, BLOCK_HEADDIM], dtype=tl.float32)
    # There seems to be some problem with Triton pipelining that makes results wrong for
    # headdim=64, seqlen=(113, 255), bias_type='matrix'. In this case the for loop
    # may have zero step, and pipelining with the bias matrix could screw it up.
    # So we just exit early.
    if begin_m >= seqlen_q:
        dv_ptrs = DV + (offs_n[:, None] * stride_dvn)
        dk_ptrs = DK + (offs_n[:, None] * stride_dkn + offs_d[None, :])
        _bwd_store_dk_dv(
            dk_ptrs,
            dv_ptrs,
            dk,
            dv,
            offs_n,
            offs_d,
            seqlen_k,
            headdim,
            EVEN_M=EVEN_M,
            EVEN_N=EVEN_N,
            EVEN_HEADDIM=EVEN_HEADDIM,
        )
        return
    # k and v stay in SRAM throughout
    # [2022-10-30] TD: Same bug as the fwd. In the case of EVEN_N=True and EVEN_M=False,
    # if we just call tl.load(k_ptrs), we get the wrong output!
    if EVEN_N & EVEN_M:
        if EVEN_HEADDIM:
            k = tl.load(k_ptrs)
            v = tl.load(v_ptrs)
        else:
            k = tl.load(k_ptrs, mask=offs_d[None, :] < headdim, other=0.0)
            v = tl.load(v_ptrs)
    else:
        if EVEN_HEADDIM:
            k = tl.load(k_ptrs, mask=offs_n[:, None] < seqlen_k, other=0.0)
            v = tl.load(v_ptrs, mask=offs_n[:, None] < seqlen_k, other=0.0)
        else:
            k = tl.load(
                k_ptrs, mask=(offs_n[:, None] < seqlen_k) & (offs_d[None, :] < headdim), other=0.0
            )
            v = tl.load(
                v_ptrs, mask=(offs_n[:, None] < seqlen_k), other=0.0
            )
    # loop over rows
    num_block_m = tl.cdiv(seqlen_q, BLOCK_M)
    for start_m in range(begin_m, num_block_m * BLOCK_M, BLOCK_M):
        start_m = tl.multiple_of(start_m, BLOCK_M)
        offs_m_curr = start_m + offs_m
        # load q, k, v, do on-chip
        # Same bug as below. Otherwise gives wrong result for headdim=40, seqlen=(128, 117)
        if EVEN_M & EVEN_HEADDIM:
            q = tl.load(q_ptrs)
        else:
            if EVEN_HEADDIM:
                q = tl.load(q_ptrs, mask=offs_m_curr[:, None] < seqlen_q, other=0.0)
            else:
                q = tl.load(
                    q_ptrs,
                    mask=(offs_m_curr[:, None] < seqlen_q) & (offs_d[None, :] < headdim),
                    other=0.0,
                )
        # recompute p = softmax(qk, dim=-1).T
        qk = tl.dot(q, tl.trans(k))
        # Trying to combine the two masks seem to make the result wrong
        if not EVEN_N:  # Need to mask out otherwise the softmax is wrong
            qk = tl.where(offs_n[None, :] < seqlen_k, qk, float("-inf"))
        if IS_CAUSAL:
            qk = tl.where(offs_m_curr[:, None] >= (offs_n[None, :]), qk, float("-inf"))
        # There seems to be a race condition when headdim=48/96, and dq, dk, dv are wrong.
        # Also wrong for headdim=64.
        if not (EVEN_M & EVEN_HEADDIM):
            tl.debug_barrier()
        lse_i = tl.load(LSE + offs_m_curr)
        p = tl.exp(qk * softmax_scale - lse_i[:, None])
        # compute dv
        # [2022-10-30] TD: A Triton bug: if EVEN_M=True and EVEN_HEADDIM=False, if we call
        # do = tl.load(do_ptrs, mask=offs_d[None, :] < headdim, other=0.0), we get wrong outputs
        # in the case of headdim=48/96, seqlen_q & seqlen_k >= 512. If headdim=40 or seqlen < 512,
        # the output is correct.
        if EVEN_M & EVEN_HEADDIM:
            do = tl.load(do_ptrs)
        else:
            # [2022-11-01] TD: Triton bug, there's a race condition if we just use m_mask and not d_mask.
            do = tl.load(
                do_ptrs,
                mask=(offs_m_curr[:, None] < seqlen_q),
                other=0.0,
            )
        # if EVEN_M:
        #     if EVEN_HEADDIM:
        #         do = tl.load(do_ptrs)
        #     else:
        #         do = tl.load(do_ptrs, mask=offs_d[None, :] < headdim, other=0.0)
        # else:
        #     if EVEN_HEADDIM:
        #         do = tl.load(do_ptrs, mask=offs_m_curr[:, None] < seqlen_q, other=0.0)
        #     else:
        #         do = tl.load(do_ptrs, mask=(offs_m_curr[:, None] < seqlen_q)
        #                                    & (offs_d[None, :] < headdim), other=0.0)
        dv += tl.sum(p.to(do.dtype) * do, axis=0).reshape([BLOCK_N, 1])
        # compute dp = dot(v, do)
        # There seems to be a race condition when headdim=48/96, and dq, dk are wrong.
        # Also wrong for headdim=128, seqlen=(108, 256), and ATOMIC_ADD=True
        # Also wrong for headdim=64, seqlen=(1023, 1024), and ATOMIC_ADD=False
        if not (EVEN_M & EVEN_HEADDIM):
            tl.debug_barrier()
        # dp = tl.dot(do, tl.trans(v))
        dp = do * tl.trans(v)
        # There's a race condition for headdim=48
        if not EVEN_HEADDIM:
            tl.debug_barrier()
        # compute ds = p * (dp - delta[:, None])
        # Putting the subtraction after the dp matmul (instead of before) is slightly faster
        Di = tl.load(D + offs_m_curr)
        # Converting ds to q.dtype here reduces register pressure and makes it much faster
        # for BLOCK_HEADDIM=128
        ds = (p * (dp - Di[:, None]) * softmax_scale).to(q.dtype)
        # compute dk = dot(ds.T, q)
        dk += tl.dot(tl.trans(ds), q)
        # compute dq
        if not (
            EVEN_M & EVEN_HEADDIM
        ):  # Otherewise there's a race condition when BIAS_TYPE='matrix'
            tl.debug_barrier()
        if not ATOMIC_ADD:
            if EVEN_M & EVEN_HEADDIM:  # Race condition if we just do EVEN_M
                dq = tl.load(dq_ptrs, eviction_policy="evict_last")
                dq += tl.dot(ds, k)
                tl.store(dq_ptrs, dq, eviction_policy="evict_last")
            else:
                if EVEN_HEADDIM:
                    dq = tl.load(
                        dq_ptrs,
                        mask=offs_m_curr[:, None] < seqlen_q,
                        other=0.0,
                        eviction_policy="evict_last",
                    )
                    dq += tl.dot(ds, k)
                    tl.store(
                        dq_ptrs,
                        dq,
                        mask=offs_m_curr[:, None] < seqlen_q,
                        eviction_policy="evict_last",
                    )
                else:
                    dq = tl.load(
                        dq_ptrs,
                        mask=(offs_m_curr[:, None] < seqlen_q) & (offs_d[None, :] < headdim),
                        other=0.0,
                        eviction_policy="evict_last",
                    )
                    dq += tl.dot(ds, k)
                    tl.store(
                        dq_ptrs,
                        dq,
                        mask=(offs_m_curr[:, None] < seqlen_q) & (offs_d[None, :] < headdim),
                        eviction_policy="evict_last",
                    )
        else:  # If we're parallelizing across the seqlen_k dimension
            dq = tl.dot(ds, k)
            if EVEN_M & EVEN_HEADDIM:  # Race condition if we just do EVEN_M
                tl.atomic_add(dq_ptrs, dq)
            else:
                if EVEN_HEADDIM:
                    tl.atomic_add(dq_ptrs, dq, mask=offs_m_curr[:, None] < seqlen_q)
                else:
                    tl.atomic_add(
                        dq_ptrs,
                        dq,
                        mask=(offs_m_curr[:, None] < seqlen_q) & (offs_d[None, :] < headdim),
                    )
        # increment pointers
        dq_ptrs += BLOCK_M * stride_dqm
        q_ptrs += BLOCK_M * stride_qm
        do_ptrs += BLOCK_M * stride_dom
    # write-back
    dv_ptrs = DV + (offs_n[:, None] * stride_dvn)
    dk_ptrs = DK + (offs_n[:, None] * stride_dkn + offs_d[None, :])
    _bwd_store_dk_dv(
        dk_ptrs,
        dv_ptrs,
        dk,
        dv,
        offs_n,
        offs_d,
        seqlen_k,
        headdim,
        EVEN_M=EVEN_M,
        EVEN_N=EVEN_N,
        EVEN_HEADDIM=EVEN_HEADDIM,
    )

@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "SEQUENCE_PARALLEL": False},
            num_warps=8,
            num_stages=1,
            pre_hook=init_to_zero("DQ"),
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "SEQUENCE_PARALLEL": True},
            num_warps=8,
            num_stages=1,
            pre_hook=init_to_zero("DQ"),
        ),
        # Other configs seem to give wrong results when seqlen_q % 128 != 0, disabling them for now
        # # Kernel is buggy (give wrong result) if we set BLOCK_m=128, BLOCK_n=64, num_warps=*4*
        # triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "SEQUENCE_PARALLEL": False}, num_warps=8, num_stages=1, pre_hook=init_to_zero('DQ')),
        # triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "SEQUENCE_PARALLEL": True}, num_warps=8, num_stages=1, pre_hook=init_to_zero('DQ')),
        # triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "SEQUENCE_PARALLEL": False}, num_warps=4, num_stages=1, pre_hook=init_to_zero('DQ')),
        # triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "SEQUENCE_PARALLEL": True}, num_warps=4, num_stages=1, pre_hook=init_to_zero('DQ')),
    ],
    key=["CACHE_KEY_SEQLEN_Q", "CACHE_KEY_SEQLEN_K", "IS_CAUSAL", "BLOCK_HEADDIM"],
)
@triton.heuristics(
    {
        "EVEN_M": lambda args: args["seqlen_q"] % args["BLOCK_M"] == 0,
        "EVEN_N": lambda args: args["seqlen_k"] % args["BLOCK_N"] == 0,
        "EVEN_HEADDIM": lambda args: args["headdim"] == args["BLOCK_HEADDIM"],
    }
)
@triton.jit
def _bwd_kernel(
    Q,
    K,
    V,
    DO,
    DQ,
    DK,
    DV,
    LSE,
    D,
    softmax_scale,
    stride_qb,
    stride_qh,
    stride_qm,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_vb,
    stride_vh,
    stride_vn,
    stride_dob,
    stride_doh,
    stride_dom,
    stride_dqb,
    stride_dqh,
    stride_dqm,
    stride_dkb,
    stride_dkh,
    stride_dkn,
    stride_dvb,
    stride_dvh,
    stride_dvn,
    nheads,
    seqlen_q,
    seqlen_k,
    seqlen_q_rounded,
    headdim,
    CACHE_KEY_SEQLEN_Q,
    CACHE_KEY_SEQLEN_K,
    IS_CAUSAL: tl.constexpr,
    BLOCK_HEADDIM: tl.constexpr,
    SEQUENCE_PARALLEL: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_HEADDIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    off_hb = tl.program_id(1)
    off_b = off_hb // nheads
    off_h = off_hb % nheads
    # offset pointers for batch/head
    Q += off_b * stride_qb + off_h * stride_qh
    K += off_b * stride_kb + off_h * stride_kh
    V += off_b * stride_vb + off_h * stride_vh
    DO += off_b * stride_dob + off_h * stride_doh
    DQ += off_b * stride_dqb + off_h * stride_dqh
    DK += off_b * stride_dkb + off_h * stride_dkh
    DV += off_b * stride_dvb + off_h * stride_dvh
    # pointer to row-wise quantities in value-like data
    D += off_hb * seqlen_q_rounded
    LSE += off_hb * seqlen_q_rounded
    if not SEQUENCE_PARALLEL:
        num_block_n = tl.cdiv(seqlen_k, BLOCK_N)
        for start_n in range(0, num_block_n):
            _bwd_kernel_one_col_block(
                start_n,
                Q,
                K,
                V,
                DO,
                DQ,
                DK,
                DV,
                LSE,
                D,
                softmax_scale,
                stride_qm,
                stride_kn,
                stride_vn,
                stride_dom,
                stride_dqm,
                stride_dkn,
                stride_dvn,
                seqlen_q,
                seqlen_k,
                headdim,
                ATOMIC_ADD=False,
                IS_CAUSAL=IS_CAUSAL,
                BLOCK_HEADDIM=BLOCK_HEADDIM,
                EVEN_M=EVEN_M,
                EVEN_N=EVEN_N,
                EVEN_HEADDIM=EVEN_HEADDIM,
                BLOCK_M=BLOCK_M,
                BLOCK_N=BLOCK_N,
            )
    else:
        start_n = tl.program_id(0)
        _bwd_kernel_one_col_block(
            start_n,
            Q,
            K,
            V,
            DO,
            DQ,
            DK,
            DV,
            LSE,
            D,
            softmax_scale,
            stride_qm,
            stride_kn,
            stride_vn,
            stride_dom,
            stride_dqm,
            stride_dkn,
            stride_dvn,
            seqlen_q,
            seqlen_k,
            headdim,
            ATOMIC_ADD=True,
            IS_CAUSAL=IS_CAUSAL,
            BLOCK_HEADDIM=BLOCK_HEADDIM,
            EVEN_M=EVEN_M,
            EVEN_N=EVEN_N,
            EVEN_HEADDIM=EVEN_HEADDIM,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
        )
        

def _flash_attn_backward(
    do, q, k, v, o, lse, dq, dk, dv, causal=False, softmax_scale=None
):
    # Make sure that the last dimension is contiguous
    if do.stride(-1) != 1:
        do = do.contiguous()
    batch, seqlen_q, nheads, d = q.shape
    _, seqlen_k, _, _ = k.shape
    # assert d in {16, 32, 64, 128}
    assert d <= 128
    seqlen_q_rounded = math.ceil(seqlen_q / 128) * 128
    assert lse.shape == (batch, nheads, seqlen_q_rounded)
    assert q.stride(-1) == k.stride(-1) == v.stride(-1) == o.stride(-1) == 1
    assert dq.stride(-1) == dk.stride(-1) ==  dv.stride(-1) == 1
    softmax_scale = softmax_scale or 1.0 / math.sqrt(d)
    # dq_accum = torch.zeros_like(q, dtype=torch.float32)
    dq_accum = torch.empty_like(q, dtype=torch.float32)
    delta = torch.empty_like(lse)
    # delta = torch.zeros_like(lse)

    BLOCK_HEADDIM = max(triton.next_power_of_2(d), 16)
    grid = lambda META: (triton.cdiv(seqlen_q, META["BLOCK_M"]), batch * nheads)
    _bwd_preprocess_do_o_dot[grid](
        o,
        do,
        delta,
        o.stride(0),
        o.stride(2),
        o.stride(1),
        do.stride(0),
        do.stride(2),
        do.stride(1),
        nheads,
        seqlen_q,
        seqlen_q_rounded,
        d,
        BLOCK_M=128,
        BLOCK_HEADDIM=BLOCK_HEADDIM,
    )

    # BLOCK_M = 128
    # BLOCK_N = 64
    # num_warps = 4
    grid = lambda META: (
        triton.cdiv(seqlen_k, META["BLOCK_N"]) if META["SEQUENCE_PARALLEL"] else 1,
        batch * nheads,
    )
    _bwd_kernel[grid](
        q,
        k,
        v,
        do,
        dq_accum,
        dk,
        dv,
        lse,
        delta,
        softmax_scale,
        q.stride(0),
        q.stride(2),
        q.stride(1),
        k.stride(0),
        k.stride(2),
        k.stride(1),
        v.stride(0),
        v.stride(2),
        v.stride(1),
        do.stride(0),
        do.stride(2),
        do.stride(1),
        dq_accum.stride(0),
        dq_accum.stride(2),
        dq_accum.stride(1),
        dk.stride(0),
        dk.stride(2),
        dk.stride(1),
        dv.stride(0),
        dv.stride(2),
        dv.stride(1),
        nheads,
        seqlen_q,
        seqlen_k,
        seqlen_q_rounded,
        d,
        seqlen_q // 32,
        seqlen_k // 32,  # key for triton cache (limit number of compilations)
        # Can't use kwargs here because triton autotune expects key to be args, not kwargs
        # IS_CAUSAL=causal, BLOCK_HEADDIM=d,
        causal,
        BLOCK_HEADDIM,
        # SEQUENCE_PARALLEL=False,
        # BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        # num_warps=num_warps,
        # num_stages=1,
    )
    dq.copy_(dq_accum)