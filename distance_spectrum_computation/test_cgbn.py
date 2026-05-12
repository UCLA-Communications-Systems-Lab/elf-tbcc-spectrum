# fmt: off
import os, ctypes
import numpy as np
import yaml
from numba import cuda
from setup import setup_A_Wbit_D
from step import trellisStep_shift
import argparse

# import the shared library
lib_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "lib", "libfoldshift.so"))
if not os.path.exists(lib_path):
    raise FileNotFoundError(f"Shared library not found: {lib_path}. Run foldshift_compile.sh first.")

cuda_lib = ctypes.CDLL(lib_path)

cuda_lib.launchCGBNPipeline.argtypes = [
    ctypes.c_void_p,
    ctypes.c_int, ctypes.c_int,                  # num_states, initial_max_weight
    ctypes.c_void_p, ctypes.c_int, ctypes.c_int, # d_W, W_dim0, W_dim1
    ctypes.c_void_p, ctypes.c_int, ctypes.c_int, # d_D, D_dim0, D_dim1
    ctypes.c_int, ctypes.c_int, ctypes.c_int,    # stages, shift, max_X
    ctypes.c_int, ctypes.c_void_p,               # basis_state, d_spectrum
    ctypes.c_int,                                # uint64_per_value
]
cuda_lib.launchCGBNPipeline.restype = None

# getters for CGBN-values
cuda_lib.getCGBNBits.argtypes = []
cuda_lib.getCGBNBits.restype = ctypes.c_int

cuda_lib.getCGBNTPI.argtypes = []
cuda_lib.getCGBNTPI.restype = ctypes.c_int

cuda_lib.getCGBNMemSize.argtypes = []
cuda_lib.getCGBNMemSize.restype = ctypes.c_int

cuda_lib.getCGBNLimbs.argtypes = []
cuda_lib.getCGBNLimbs.restype = ctypes.c_int

CGBN_Bits = cuda_lib.getCGBNBits()
CGBN_TPI = cuda_lib.getCGBNTPI()
CGBN_Mem_Size = cuda_lib.getCGBNMemSize()
CGBN_Limbs = cuda_lib.getCGBNLimbs()

def main(path):
    with open(path, "r") as f:
        code_config = yaml.safe_load(f)
    output_file_name = code_config["output_file_name"]

    As, W_weight, D, basis, num_trellis_stages = setup_A_Wbit_D(code_config)
    A_shape = As[0].shape
    O_y, O_x = A_shape # O_y is 2^(m + nu) = 2^(num_states)
    max_shift_per_stage = 2
    max_X = O_x + max_shift_per_stage * num_trellis_stages

    d_W = cuda.to_device(W_weight)
    d_D = cuda.to_device(D)
    # allocates max_X count of (32 bits * CGBN_LIMBS) = (32 * BITS / 32) = BITS
    d_spectrum = cuda.to_device(np.zeros(max_X * CGBN_Limbs, dtype=np.uint32)) 

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

    # compute number of uint64_ts we need to pack contiguously to fit k
    # e.g., if k = 113, we need a uint128, or two uint64_ts contiguously
    # ceil(K / 64)
    needed_bits = (code_config["bch_config"]["K"] + 1)
    uint64_per_value = (needed_bits + 63) // 64;
    assert CGBN_Bits >= needed_bits

    h_buf = np.zeros((O_y, max_X, uint64_per_value), dtype=np.uint64)
    d_buf_a = cuda.to_device(h_buf.reshape(-1))

    for i_stream, A in enumerate(As):
        O_y, O_x = A.shape

        # ping-pong buffers
        # size = 2^(num_states) * max_X (n + 1) * sizeof(CGBN)
        # for k113n254v8, M = 14, size = 16.32 GB approx if we keep CGBN as 128 bit int :(
        h_in = np.zeros((O_y, max_X, uint64_per_value), dtype=np.uint64) 
        h_in[0:A.shape[0], 0:A.shape[1], 0] = A

        d_buf_a.copy_to_device(h_in.reshape(-1))

        cuda_lib.launchCGBNPipeline(
            d_buf_a.device_ctypes_pointer.value,
            O_y, O_x,
            d_W.device_ctypes_pointer.value, W_weight.shape[0], W_weight.shape[1],
            d_D.device_ctypes_pointer.value, D.shape[0], D.shape[1],
            num_trellis_stages, max_shift_per_stage, max_X,
            int(basis[i_stream]),
            d_spectrum.device_ctypes_pointer.value,
            uint64_per_value
        )

    cuda.synchronize()
    raw_limb_spectrum = d_spectrum.copy_to_host()
    gpu_spectrum = []
    for i in range(max_X):
        val = sum(int(raw_limb_spectrum[i * CGBN_Limbs + j]) << (32 * j) for j in range(CGBN_Limbs))
        gpu_spectrum.append(val)

    print("gpu_distance_spectrum:", gpu_spectrum)
    os.makedirs("output/fold", exist_ok=True)
    np.save("output/fold/" + output_file_name, gpu_spectrum)

    if (USE_CPU):
        print(cpu_spectrum)
        if np.array_equal(cpu_spectrum, gpu_spectrum):
            print("gpu matches cpu spectrum, all good")
        else:
            print("error: cpu and gpu spectrums are different")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help = "path to YAML Config File")
    args = parser.parse_args()
    path = args.config
    main(path)
