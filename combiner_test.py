# -*- coding: utf-8 -*-
"""Numba Boilerplate for tbcc-decoder-test
=================================================
This module provides a lightweight Numba-based wrapper that mirrors the
CUDA kernel defined in ``combiner.cu``. The goal is to give a quick way to
experiment with the ``combineTrellisStagesKernel`` logic from Python without
building a full C++ extension.

Running the test:

    python combiner_test.py

If everything is set up correctly you should see a short success message and
benchmark timings.
"""

import math
import time
import numpy as np
from numba import cuda, float32, int32

# ---------------------------------------------------------------------------
# Kernel implementation (mirrors combiner.cu)
# ---------------------------------------------------------------------------

@cuda.jit
def combine_trellis_stages_kernel(
    input_stage_left,
    input_stage_right,
    output_stage,
    output_argmin,
    M,
    N,
):
    """Combine two trellis stages.

    Each block handles one stage pair (indexed by blockIdx.x).
    Threads cooperatively load the left/right M×M matrices into shared
    memory, compute the min-plus product, then reduce to find the block
    minimum for normalization.

    Parameters
    ----------
    input_stage_left, input_stage_right : 1-D device arrays, length N*M*M
        Flattened M×M metric matrices for the left and right stages.
    output_stage : 1-D device array, same length
        Stores the minimum metric for each (s, d) pair (normalized).
    output_argmin : 1-D device array, same length
        Stores the intermediate state r that yields the minimum.
    M : int
        Number of trellis states (e.g. 32 or 64).
    N : int
        Number of independent stage pairs.
    """
    i = cuda.blockIdx.x  # stage-pair index
    if i >= N:
        return

    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    bw = cuda.blockDim.x
    bh = cuda.blockDim.y
    tid = ty * bw + tx
    block_size = bw * bh  # total threads in this block (max 1024 for 32x32)

    # ------------------------------------------------------------------
    # Shared memory layout:
    #   [0        .. M*M-1]       left matrix
    #   [M*M      .. 2*M*M-1]     right matrix
    #   [2*M*M    .. 2*M*M+1023]  reduction buffer (1024 slots for 32x32)
    # ------------------------------------------------------------------
    smem = cuda.shared.array(shape=(2 * 64 * 64 + 1024,), dtype=float32)
    left_offset   = 0
    right_offset  = M * M
    reduce_offset = 2 * M * M  # reduction buffer starts here

    # --- Load matrices into shared memory ---
    for row in range(ty, M, bh):
        for col in range(tx, M, bw):
            idx = i * M * M + row * M + col
            smem[left_offset  + row * M + col] = input_stage_left[idx]
            smem[right_offset + row * M + col] = input_stage_right[idx]
    cuda.syncthreads()

    # --- Min-plus product ---
    thread_min_metric = math.inf
    for s in range(ty, M, bh):
        for d in range(tx, M, bw):
            min_val = math.inf
            best_r  = 0
            for r in range(M):
                val = smem[left_offset + s * M + r] + smem[right_offset + r * M + d]
                if val < min_val:
                    min_val = val
                    best_r  = r
            target_idx = i * M * M + s * M + d
            output_stage[target_idx]  = min_val
            output_argmin[target_idx] = best_r
            if min_val < thread_min_metric:
                thread_min_metric = min_val

    # --- Parallel tree reduction to find block minimum ---
    # Each thread writes its local min into the reduction buffer.
    smem[reduce_offset + tid] = thread_min_metric
    cuda.syncthreads()

    stride = block_size // 2
    while stride > 0:
        if tid < stride:
            a = smem[reduce_offset + tid]
            b = smem[reduce_offset + tid + stride]
            if b < a:
                smem[reduce_offset + tid] = b
        cuda.syncthreads()
        stride //= 2

    # smem[reduce_offset] now holds the true block minimum
    block_min = smem[reduce_offset]
    cuda.syncthreads()

    # --- Normalize ---
    for s in range(ty, M, bh):
        for d in range(tx, M, bw):
            target_idx = i * M * M + s * M + d
            output_stage[target_idx] -= block_min


# ---------------------------------------------------------------------------
# Host helper
# ---------------------------------------------------------------------------

def launch_combine(left, right, M, stream=None):
    """Launch combine_trellis_stages_kernel.

    Parameters
    ----------
    left, right : np.ndarray, shape (N, M, M), dtype float32
    M : int
    stream : cuda.Stream, optional

    Returns
    -------
    output_stage, output_argmin : np.ndarray, shape (N, M, M)
    """
    assert left.shape == right.shape, "left and right must have identical shapes"
    assert M <= 64, "kernel shared memory sized for M <= 64"

    N = left.shape[0]
    d_left   = cuda.to_device(left.ravel().astype(np.float32))
    d_right  = cuda.to_device(right.ravel().astype(np.float32))
    d_out    = cuda.device_array(N * M * M, dtype=np.float32)
    d_argmin = cuda.device_array(N * M * M, dtype=np.int32)

    block_x = min(M, 32)
    block_y = min(M, 32)
    block   = (block_x, block_y)
    grid    = (N,)

    combine_trellis_stages_kernel[grid, block, stream](
        d_left, d_right, d_out, d_argmin, M, N
    )
    out    = d_out.copy_to_host().reshape(N, M, M)
    argmin = d_argmin.copy_to_host().reshape(N, M, M)
    return out, argmin


# ---------------------------------------------------------------------------
# Reference implementation (pure NumPy)
# ---------------------------------------------------------------------------

def _numpy_reference(left, right):
    N, M, _ = left.shape
    out    = np.empty_like(left)
    argmin = np.empty((N, M, M), dtype=np.int32)
    for i in range(N):
        for s in range(M):
            for d in range(M):
                vals         = left[i, s, :] + right[i, :, d]
                best_r       = int(np.argmin(vals))
                out[i, s, d] = vals[best_r]
                argmin[i, s, d] = best_r
        out[i] -= out[i].min()
    return out, argmin


# ---------------------------------------------------------------------------
# Test / benchmark
# ---------------------------------------------------------------------------

def test_combiner():
    M   = 32
    N   = 4
    rng = np.random.default_rng(0)
    left  = rng.random((N, M, M)).astype(np.float32)
    right = rng.random((N, M, M)).astype(np.float32)

    # NumPy reference
    t0    = time.time()
    ref_out, ref_argmin = _numpy_reference(left, right)
    t_ref = time.time() - t0
    print(f"NumPy reference:  {t_ref * 1000:.2f} ms")

    # Numba GPU kernel
    t0     = time.time()
    gpu_out, gpu_argmin = launch_combine(left, right, M)
    t_gpu  = time.time() - t0
    print(f"Numba GPU kernel: {t_gpu * 1000:.2f} ms")

    # Validate
    metrics_ok = np.allclose(gpu_out, ref_out, atol=1e-5)
    argmin_ok  = np.array_equal(gpu_argmin, ref_argmin)

    if metrics_ok and argmin_ok:
        print(" Passed: metrics and argmin match reference.")
    else:
        if not metrics_ok:
            diff = np.abs(gpu_out - ref_out)
            print(f" Metric mismatch — max diff: {diff.max():.6f}")
        if not argmin_ok:
            print(" Argmin mismatch")

    print(f"\nSpeedup: {t_ref / t_gpu:.2f}x")


if __name__ == "__main__":
    test_combiner()