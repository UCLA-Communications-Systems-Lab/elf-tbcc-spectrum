import os
import ctypes
import numpy as np
from numba import cuda

lib_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "lib", "libtrellis.so"))
if not os.path.exists(lib_path):
    raise FileNotFoundError(f"Shared library not found: {lib_path}. Run compile.sh first.")

cuda_lib = ctypes.CDLL(lib_path)

# void launch_combine_kernel(void* d_left, void* d_right, void* d_out, void* d_argmin, int M, int N)
cuda_lib.launch_combine_kernel.argtypes = [
    ctypes.c_void_p,  # d_left
    ctypes.c_void_p,  # d_right
    ctypes.c_void_p,  # d_out
    ctypes.c_void_p,  # d_argmin
    ctypes.c_int,     # M
    ctypes.c_int,     # N
]
cuda_lib.launch_combine_kernel.restype = None

# void launch_reduce_kernel(void* d_in, void* d_out, int M, int K)
cuda_lib.launch_reduce_kernel.argtypes = [
    ctypes.c_void_p,  # d_in
    ctypes.c_void_p,  # d_out
    ctypes.c_int,     # M
    ctypes.c_int,     # K
]
cuda_lib.launch_reduce_kernel.restype = None


def launch_combine_cuda(left, right, M):
    assert left.shape == right.shape, "left and right must have identical shapes"
    assert left.dtype == np.float32, "inputs must be float32"
    N = left.shape[0]

    d_left   = cuda.to_device(left)
    d_right  = cuda.to_device(right)
    d_out    = cuda.device_array_like(d_left)
    d_argmin = cuda.device_array(shape=d_left.shape, dtype=np.int32)

    ptr_left   = d_left.device_ctypes_pointer.value
    ptr_right  = d_right.device_ctypes_pointer.value
    ptr_out    = d_out.device_ctypes_pointer.value
    ptr_argmin = d_argmin.device_ctypes_pointer.value

    import time
    t0 = time.perf_counter()
    cuda_lib.launch_combine_kernel(
        ptr_left,
        ptr_right,
        ptr_out,
        ptr_argmin,
        int(M),
        int(N),
    )

    cuda.synchronize()
    kernel_time_ms = (time.perf_counter() - t0) * 1000

    out    = d_out.copy_to_host()
    argmin = d_argmin.copy_to_host()
    return out, argmin, kernel_time_ms


def launch_reduce_tree_cuda(left, right, M):
    import time
    assert left.shape == right.shape, "left and right must have identical shapes"
    assert left.dtype == np.float32, "inputs must be float32"
    N = left.shape[0]

    d_left   = cuda.to_device(left)
    d_right  = cuda.to_device(right)
    
    # Ping-pong buffers for hierarchical reduction
    d_bufA   = cuda.device_array(shape=(N, M, M), dtype=np.float32)
    d_bufB   = cuda.device_array(shape=(N, M, M), dtype=np.float32)
    # Argmin is required for the first level combine kernel, but we don't need it later
    d_argmin = cuda.device_array(shape=(N, M, M), dtype=np.int32)

    ptr_left   = d_left.device_ctypes_pointer.value
    ptr_right  = d_right.device_ctypes_pointer.value
    ptr_A      = d_bufA.device_ctypes_pointer.value
    ptr_B      = d_bufB.device_ctypes_pointer.value
    ptr_argmin = d_argmin.device_ctypes_pointer.value

    t0 = time.perf_counter()

    # 1. First level: combine left and right pairs into d_bufA
    cuda_lib.launch_combine_kernel(
        ptr_left, ptr_right, ptr_A, ptr_argmin, int(M), int(N)
    )

    # 2. Hierarchical reduction on the GPU
    K = N
    curr_ptr = ptr_A
    next_ptr = ptr_B

    while K > 1:
        cuda_lib.launch_reduce_kernel(
            curr_ptr, next_ptr, int(M), int(K)
        )
        K = (K + 1) // 2
        # Swap pointers for the next level
        curr_ptr, next_ptr = next_ptr, curr_ptr

    cuda.synchronize()
    kernel_time_ms = (time.perf_counter() - t0) * 1000

    # The final combined matrix is at the front of whichever buffer curr_ptr points to
    if curr_ptr == ptr_A:
        out = d_bufA[:1].copy_to_host()
    else:
        out = d_bufB[:1].copy_to_host()

    return out, kernel_time_ms
