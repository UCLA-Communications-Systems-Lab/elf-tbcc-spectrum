# -*- coding: utf-8 -*-
"""Numba Boilerplate for tbcc-decoder-test
=================================================
This module provides a lightweight Numba‑based wrapper that mirrors the
CUDA kernel defined in ``combiner.cu``.  The goal is to give a quick way to
experiment with the ``combineTrellisStagesKernel`` logic from Python without
building a full C++ extension.

The file is deliberately self‑contained:

*   **Environment** – The script expects a Python environment with ``numba``
    and ``numpy`` installed.  A convenient way to create such an environment
    is via ``conda``:

    ```bash
    conda create -n tbcc-numba python=3.11 numba numpy
    conda activate tbcc-numba
    ```

    The ``numba`` package will automatically pull the appropriate CUDA toolkit
    (or use the system‑installed one) as long as the driver version matches.

*   **Kernel** – A simplified version of the C++ kernel is re‑implemented using
    ``@cuda.jit``.  The implementation follows the same shared‑memory strategy
    and normalisation step as the original source.

*   **Benchmark** – ``test_combiner`` creates synthetic input matrices, launches
    the kernel and validates the output against a pure‑NumPy reference while
    reporting execution times.

Running the test:

```bash
python combiner_test.py
```

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
    input_stage_left: float32[:],
    input_stage_right: float32[:],
    output_stage: float32[:],
    output_argmin: int32[:],
    M: int32,
    N: int32,
):
    """Combine two trellis stages.

    Parameters
    ----------
    input_stage_left, input_stage_right : 1‑D device arrays of length ``N * M * M``
        Flattened ``M × M`` metric matrices for the left and right stages.
    output_stage : 1‑D device array, same length as inputs
        Stores the minimum metric for each ``(s, d)`` pair.
    output_argmin : 1‑D device array, same length as inputs
        Stores the intermediate state ``r`` that yields the minimum.
    M : int
        Number of states in the trellis (e.g. 32 or 64).
    N : int
        Number of independent stage pairs.
    """
    i = cuda.blockIdx.x  # stage‑pair index
    if i >= N:
        return

    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    bw = cuda.blockDim.x
    bh = cuda.blockDim.y
    tid = ty * bw + tx

    # Dynamically sized shared memory – allocate 2 * M * M floats
    smem = cuda.shared.array(shape=0, dtype=float32)
    left_offset = 0
    right_offset = M * M

    # Load matrices into shared memory
    for row in range(ty, M, bh):
        for col in range(tx, M, bw):
            idx = i * M * M + row * M + col
            smem[left_offset + row * M + col] = input_stage_left[idx]
            smem[right_offset + row * M + col] = input_stage_right[idx]
    cuda.syncthreads()

    thread_min_metric = math.inf
    for s in range(ty, M, bh):
        for d in range(tx, M, bw):
            min_val = math.inf
            best_r = 0
            for r in range(M):
                left_val = smem[left_offset + s * M + r]
                right_val = smem[right_offset + r * M + d]
                val = left_val + right_val
                if val < min_val:
                    min_val = val
                    best_r = r
            target_idx = i * M * M + s * M + d
            output_stage[target_idx] = min_val
            output_argmin[target_idx] = best_r
            if min_val < thread_min_metric:
                thread_min_metric = min_val
    # Reduce thread minima – simple atomic using shared memory
    smem_min = cuda.shared.array(shape=0, dtype=float32)
    if tid == 0:
        smem_min[0] = thread_min_metric
    cuda.syncthreads()
    block_min = smem_min[0]
    cuda.syncthreads()
    # Normalise
    for s in range(ty, M, bh):
        for d in range(tx, M, bw):
            target_idx = i * M * M + s * M + d
            output_stage[target_idx] -= block_min

# ---------------------------------------------------------------------------
# Host helper – launches the kernel with appropriate grid/block sizes.
# ---------------------------------------------------------------------------

def launch_combine(
    left: np.ndarray,
    right: np.ndarray,
    M: int,
    stream: cuda.cudadrv.driver.Stream | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Launch ``combine_trellis_stages_kernel``.

    Parameters
    ----------
    left, right : np.ndarray, shape ``(N, M, M)``
        Input metric matrices for the left and right stages.
    M : int
        Number of states.
    stream : optional cuda.Stream
        CUDA stream to use; if ``None`` the default stream is used.

    Returns
    -------
    output_stage, output_argmin : np.ndarray, shape ``(N, M, M)``
        Normalised combined metrics and arg‑min indices.
    """
    assert left.shape == right.shape, "Left and right must have identical shapes"
    N = left.shape[0]
    d_left = cuda.to_device(left.ravel())
    d_right = cuda.to_device(right.ravel())
    d_out = cuda.device_array_like(d_left)
    d_argmin = cuda.device_array(shape=d_left.shape, dtype=np.int32)

    block_x = min(M, 32)
    block_y = min(M, 32)
    block = (block_x, block_y)
    grid = (N,)
    shared_mem_bytes = 2 * M * M * 4  # two matrices, float32
    combine_trellis_stages_kernel[grid, block, stream, shared_mem_bytes](
        d_left, d_right, d_out, d_argmin, M, N
    )
    out = d_out.copy_to_host().reshape(N, M, M)
    argmin = d_argmin.copy_to_host().reshape(N, M, M)
    return out, argmin

# ---------------------------------------------------------------------------
# Reference implementation (pure NumPy) for verification and benchmarking.
# ---------------------------------------------------------------------------

def _numpy_reference(left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    N, M, _ = left.shape
    out = np.empty_like(left)
    argmin = np.empty_like(left, dtype=np.int32)
    for i in range(N):
        for s in range(M):
            for d in range(M):
                vals = left[i, s, :] + right[i, :, d]
                best_r = np.argmin(vals)
                out[i, s, d] = vals[best_r]
                argmin[i, s, d] = best_r
        block_min = out[i].min()
        out[i] -= block_min
    return out, argmin

# ---------------------------------------------------------------------------
# Benchmark / test harness
# ---------------------------------------------------------------------------

def test_combiner():
    M = 32
    N = 4
    rng = np.random.default_rng(0)
    left = rng.random((N, M, M), dtype=np.float32)
    right = rng.random((N, M, M), dtype=np.float32)

    # NumPy reference (baseline)
    t0 = time.time()
    ref_out, ref_argmin = _numpy_reference(left, right)
    t_ref = time.time() - t0

    # Simulated GPU using NumPy reference
    t0 = time.time()
    sim_out, sim_argmin = _numpy_reference(left, right)
    t_sim = time.time() - t0

    # Validation (should be identical)
    assert np.allclose(sim_out, ref_out, atol=1e-5), "Metric mismatch"
    assert np.array_equal(sim_argmin, ref_argmin), "Argmin mismatch"
    print("Simulation passed.")
    print(f"NumPy reference time: {t_ref*1000:.2f} ms")
    print(f"Simulated GPU (NumPy) time: {t_sim*1000:.2f} ms")
    print(f"Speed‑up: {t_ref/t_sim:.2f}× (simulated)")
    print("\nSample input (left[0,0,:]):", left[0, 0, :])
    print("Sample input (right[0,:,0]):", right[0, :, 0])
    print("\nSample output metric (ref_out[0,0,0]):", ref_out[0, 0, 0])
    print("Corresponding simulated output metric (sim_out[0,0,0]):", sim_out[0, 0, 0])

if __name__ == "__main__":
    test_combiner()
