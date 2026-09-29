# fmt: off
import os, ctypes
import numpy as np
import yaml
from numba import cuda
from setup import setup_A_Wbit_D, spectrum_filename
from step import trellisStep_shift
import argparse

# import the shared library
lib_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "lib", "libfoldshift.so"))
if not os.path.exists(lib_path):
    raise FileNotFoundError(f"Shared library not found: {lib_path}. Run foldshift_compile.sh first.")

cuda_lib = ctypes.CDLL(lib_path)

cuda_lib.launchCGBNPipelinePunctured.argtypes = [
    ctypes.c_int,                                # starting_state
    ctypes.c_int, ctypes.c_int,                  # num_states, initial_max_weight
    ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_uint8),
    ctypes.c_void_p,
    ctypes.c_int, ctypes.c_int,                  # stages, max_X
    ctypes.c_int, ctypes.c_void_p,               # basis_state, d_spectrum
]
cuda_lib.launchCGBNPipelinePunctured.restype = ctypes.c_int

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
    spectra_filename = spectrum_filename(code_config)

    As, W_weight, D, basis, num_trellis_stages, _, widths = setup_A_Wbit_D(code_config)
    A_shape = As[0].shape
    O_y, O_x = A_shape # O_y is 2^(m + nu) = 2^(num_states)
    period = len(widths)
    max_X = code_config["tbcc_config"]["N"] + 1

    d_W = cuda.to_device(W_weight)
    d_D = cuda.to_device(D)
    # allocates max_X count of (32 bits * CGBN_LIMBS) = (32 * BITS / 32) = BITS
    d_spectrum = cuda.to_device(np.zeros(max_X * CGBN_Limbs, dtype=np.uint32)) 

    # implement cpu gate; O_y = 2^(nu + m), so pick whatever gate condition you like (cpu time will get high)
    USE_CPU = (O_y <= 1024 and code_config["bch_config"]["K"] < 64)

    if (USE_CPU):
        cpu_spectrum = np.zeros(max_X, dtype=np.uint64)
        for i_stream, A in enumerate(As):
            result = A.copy()
            for stage in range(num_trellis_stages):
                phase = stage % period
                result = trellisStep_shift(result, W_weight[phase], D, int(widths[phase]))
            cpu_spectrum += result[basis[i_stream], :]

    assert CGBN_Bits >= code_config["bch_config"]["K"] + 1

    for i_stream, A in enumerate(As):
        O_y, O_x = A.shape
        starting_state = int(np.nonzero(A)[0][0]);

        status = cuda_lib.launchCGBNPipelinePunctured(
            starting_state,
            O_y, O_x,
            d_W.device_ctypes_pointer.value, period,
            widths.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            d_D.device_ctypes_pointer.value,
            num_trellis_stages, max_X,
            int(basis[i_stream]),
            d_spectrum.device_ctypes_pointer.value,
        )
        if status != 0:
            raise RuntimeError(f"CUDA CGBN pipeline failed with error {status}")

    cuda.synchronize()
    raw_limb_spectrum = d_spectrum.copy_to_host()
    gpu_spectrum = []
    for i in range(max_X):
        val = sum(int(raw_limb_spectrum[i * CGBN_Limbs + j]) << (32 * j) for j in range(CGBN_Limbs))
        gpu_spectrum.append(val)

    K = code_config["bch_config"]["K"]
    N = code_config["tbcc_config"]["N"]
    print("gpu_distance_spectrum:", gpu_spectrum)

    # asserts
    if len(gpu_spectrum) != N + 1:
        raise AssertionError("GPU spectrum length does not match transmitted N")
    
    if sum(gpu_spectrum) != 2 ** K:
        raise AssertionError("GPU spectrum sum does not equal 2^K")

    os.makedirs("output/cgbn_out", exist_ok=True)
    np.save("output/cgbn_out/" + spectra_filename, gpu_spectrum)

    if (USE_CPU):
        print(cpu_spectrum)
        np.testing.assert_array_equal(cpu_spectrum, gpu_spectrum)
        print("gpu matches cpu spectrum, all good")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help = "path to YAML Config File")
    args = parser.parse_args()
    path = args.config
    main(path)
