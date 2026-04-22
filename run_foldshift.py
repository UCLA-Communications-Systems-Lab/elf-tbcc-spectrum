import os, ctypes
import numpy as np
import yaml
from numba import cuda
from setup import setup_A_Wbit_D
from step import trellisStep_shift

# import the shared library
lib_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "lib", "libfoldshift.so"))
if not os.path.exists(lib_path):
    raise FileNotFoundError(f"Shared library not found: {lib_path}. Run foldshift_compile.sh first.")

cuda_lib = ctypes.CDLL(lib_path)

cuda_lib.launchFoldshiftPipeline.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p,            # d_buffer_a, d_buffer_b
    ctypes.c_int, ctypes.c_int,                  # num_states, initial_max_weight
    ctypes.c_void_p, ctypes.c_int, ctypes.c_int, # d_W, W_dim0, W_dim1
    ctypes.c_void_p, ctypes.c_int, ctypes.c_int, # d_D, D_dim0, D_dim1
    ctypes.c_int, ctypes.c_int, ctypes.c_int,    # stages, shift, max_X
    ctypes.c_int, ctypes.c_void_p,               # basis_state, d_spectrum
]
cuda_lib.launchFoldshiftPipeline.restype = None

def main():
    path = "config/k11n22v3.yaml"
    with open(path, "r") as f:
        code_config = yaml.safe_load(f)
    output_file_name = code_config["output_file_name"]

    As, W_weight, D, basis, num_trellis_stages = setup_A_Wbit_D(path)
    A_shape = As[0].shape
    O_y, O_x = A_shape
    max_shift_per_stage = 2
    max_X = O_x + max_shift_per_stage * num_trellis_stages

    d_W = cuda.to_device(W_weight)
    d_D = cuda.to_device(D)
    d_spectrum = cuda.to_device(np.zeros(max_X, dtype=np.uint64))

    # implement cpu gate; O_y = 2^(nu + m), so pick whatever gate condition you like (cpu time will get high)
    USE_CPU = (O_y <= 1024)

    if (USE_CPU):
        cpu_spectrum = np.zeros(max_X, dtype=np.uint64)
        for i_stream, A in enumerate(As):
            result = A.copy()
            for stage in range(num_trellis_stages):
                result = trellisStep_shift(result, W_weight, D, max_shift_per_stage)
            # result is [num_states, final_width]; pad to max_X
            padded = np.zeros((result.shape[0], max_X), dtype=np.uint64)
            padded[:, :result.shape[1]] = result
            cpu_spectrum += padded[basis[i_stream], :]

    h_buf = np.zeros((O_y, max_X), dtype=np.uint64)
    d_buf_a = cuda.to_device(h_buf)
    d_buf_b = cuda.to_device(h_buf)
    
    for i_stream, A in enumerate(As):
        O_y, O_x = A.shape

        # ping-pong buffers
        h_in = np.zeros((O_y, max_X), dtype=np.uint64)
        h_in[0:A.shape[0], 0:A.shape[1]] = A
        d_buf_a.copy_to_device(h_in)

        cuda_lib.launchFoldshiftPipeline(
            d_buf_a.device_ctypes_pointer.value,
            d_buf_b.device_ctypes_pointer.value,
            O_y, O_x,
            d_W.device_ctypes_pointer.value, W_weight.shape[0], W_weight.shape[1],
            d_D.device_ctypes_pointer.value, D.shape[0], D.shape[1],
            num_trellis_stages, max_shift_per_stage, max_X,
            int(basis[i_stream]),
            d_spectrum.device_ctypes_pointer.value,
        )

    cuda.synchronize()
    gpu_spectrum = d_spectrum.copy_to_host()
    print("gpu_distance_spectrum:", gpu_spectrum)
    os.makedirs("output/fold", exist_ok=True)
    np.save("output/fold/" + output_file_name, gpu_spectrum)

    if (USE_CPU):
        if np.array_equal(cpu_spectrum, gpu_spectrum):
            print("gpu matches cpu spectrum, all good")
        else:
            print("error: cpu and gpu spectrums are different")

if __name__ == "__main__":
    main()