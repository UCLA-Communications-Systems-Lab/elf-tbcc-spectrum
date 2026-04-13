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

    cuda_lib.launch_combine_kernel(
        ptr_left,
        ptr_right,
        ptr_out,
        ptr_argmin,
        int(M),
        int(N),
    )

    cuda.synchronize()
    out    = d_out.copy_to_host()
    argmin = d_argmin.copy_to_host()
    return out, argmin
