import numpy as np
from numba import njit

@njit(parallel=True, fastmath=True)
def combine_trellis_stages_cpu(left: np.ndarray, right: np.ndarray):
    """Min‑plus product + block‑wise normalisation.
    Mirrors the semantics of the CUDA kernel.
    Returns:
        out (np.ndarray): shape (N, M, M) containing the min‑plus results.
        argmin (np.ndarray): shape (N, M, M) containing the intermediate state index.
    """
    N, M, _ = left.shape
    out = np.empty_like(left)
    argmin = np.empty_like(left, dtype=np.int32)
    for i in range(N):
        for s in range(M):
            for d in range(M):
                best_val = np.inf
                best_r = 0
                for r in range(M):
                    val = left[i, s, r] + right[i, r, d]
                    if val < best_val:
                        best_val = val
                        best_r = r
                out[i, s, d] = best_val
                argmin[i, s, d] = best_r
        # Normalise block by subtracting its minimum (same as CUDA kernel)
        block_min = np.min(out[i])
        out[i] -= block_min
    return out, argmin
