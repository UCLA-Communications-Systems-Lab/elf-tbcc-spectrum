# fmt: off
import os, ctypes
import numpy as np
import yaml
from setup import setup_A_Wbit_D, spectrum_filename
from step import trellisStep_shift
import argparse

def load_cuda_library():
    """Load the optional CUDA library only when a GPU run is requested."""
    lib_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "lib", "libfoldshift.so"))
    if not os.path.exists(lib_path):
        raise FileNotFoundError(f"Shared library not found: {lib_path}. Run foldshift_compile.sh first.")
    cuda_lib = ctypes.CDLL(lib_path)
    cuda_lib.launchFoldshiftPipelinePunctured.argtypes = [
        ctypes.c_int,
        ctypes.c_int, ctypes.c_int,
        ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_void_p,
    ]
    cuda_lib.launchFoldshiftPipelinePunctured.restype = ctypes.c_int
    return cuda_lib


def main(path, cpu_only=False):
    with open(path, "r") as f:
        code_config = yaml.safe_load(f)
    if code_config["bch_config"]["K"] >= 64:
        raise ValueError("K >= 64 requires test_cgbn.py for exact spectrum counts")
    spectra_filename = spectrum_filename(code_config)
    

    As, W_weight, D, basis, num_trellis_stages, _, widths = setup_A_Wbit_D(code_config)
    A_shape = As[0].shape
    O_y, O_x = A_shape
    period = len(widths)
    max_X = code_config["tbcc_config"]["N"] + 1

    # O_y = 2^(nu + m); retain the automatic safety gate for comparison runs.
    use_cpu = cpu_only or O_y <= 1024

    if use_cpu:
        cpu_spectrum = np.zeros(max_X, dtype=np.uint64)
        for i_stream, A in enumerate(As):
            result = A.copy()
            for stage in range(num_trellis_stages):
                phase = stage % period
                result = trellisStep_shift(result, W_weight[phase], D, int(widths[phase]))
            cpu_spectrum += result[basis[i_stream], :]

        assert len(cpu_spectrum) == code_config["tbcc_config"]["N"] + 1
        assert int(cpu_spectrum.sum()) == 2 ** code_config["bch_config"]["K"]
        print("cpu_distance_spectrum:", cpu_spectrum)
        if cpu_only:
            return cpu_spectrum

    cuda_lib = load_cuda_library()
    from numba import cuda

    d_W = cuda.to_device(W_weight)
    d_D = cuda.to_device(D)
    d_spectrum = cuda.to_device(np.zeros(max_X, dtype=np.uint64))

    for i_stream, A in enumerate(As):
        O_y, O_x = A.shape
        starting_state = int(np.nonzero(A)[0][0])

        status = cuda_lib.launchFoldshiftPipelinePunctured(
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
            raise RuntimeError(f"CUDA foldshift pipeline failed with error {status}")

    cuda.synchronize()
    gpu_spectrum = d_spectrum.copy_to_host()
    print("gpu_distance_spectrum:", gpu_spectrum)
    assert len(gpu_spectrum) == code_config["tbcc_config"]["N"] + 1
    assert int(gpu_spectrum.sum()) == 2 ** code_config["bch_config"]["K"]
    os.makedirs("output/fold", exist_ok=True)
    np.save("output/fold/" + spectra_filename, gpu_spectrum)

    if use_cpu:
        np.testing.assert_array_equal(cpu_spectrum, gpu_spectrum)
        print("gpu matches cpu spectrum, all good")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help = "path to YAML config file")
    parser.add_argument("--cpu", action="store_true", help="run only the portable CPU reference")
    args = parser.parse_args()
    path = args.config
    main(path, cpu_only=args.cpu)
