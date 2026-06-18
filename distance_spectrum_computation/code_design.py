from dataclasses import dataclass
import os, ctypes
from pathlib import Path
import h5py
import numpy as np
import yaml
from numba import cuda
from setup import setup_A_Wbit_D
from step import trellisStep_shift
from itertools import product, combinations
from dsu_bound_plots.bounds import dsu
import argparse

# import the shared library
lib_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "lib", "libfoldshift.so")
)
if not os.path.exists(lib_path):
    raise FileNotFoundError(
        f"Shared library not found: {lib_path}. Run foldshift_compile.sh first."
    )

cuda_lib = ctypes.CDLL(lib_path)

# fmt: off
cuda_lib.launchFoldshiftPipeline.argtypes = [
    ctypes.c_int,                                # starting_state
    ctypes.c_int, ctypes.c_int,                  # num_states, initial_max_weight
    ctypes.c_void_p, ctypes.c_int, ctypes.c_int, # d_W, W_dim0, W_dim1
    ctypes.c_void_p, ctypes.c_int, ctypes.c_int, # d_D, D_dim0, D_dim1
    ctypes.c_int, ctypes.c_int, ctypes.c_int,    # stages, shift, max_X
    ctypes.c_int, ctypes.c_void_p,               # basis_state, d_spectrum
]
cuda_lib.launchFoldshiftPipeline.restype = None
# fmt: on

# Output
output_dir = Path.cwd() / "output"
output_dir.mkdir(exist_ok=True)


@dataclass
class dist_spectra:
    hamming_dist: np.array
    num_cwds: np.array


def gcd_gf2(a, b):
    # gcd(a, b) = gcd(b, a mod b)
    while b:
        if a.bit_length() >= b.bit_length():
            a ^= b << (a.bit_length() - b.bit_length())
        else:
            a, b = b, a
    return a


def reverse_polynomial(p_octal, num_bits):
    p_int = int(p_octal, 8)
    reversed_p = int(format(p_int, f"0{num_bits}b")[::-1], 2)
    return oct(reversed_p)[2:]


def triple(elf_octal, p1, p2, m, nu):
    triple = (elf_octal, *sorted([p1, p2]))
    reversed_triple = (
        reverse_polynomial(elf_octal, m + 1),
        *sorted([reverse_polynomial(p1, nu + 1), reverse_polynomial(p2, nu + 1)]),
    )
    return min(triple, reversed_triple)


def gen_all_elf_tbcc(K_elf, N_elf, m, N_tbcc, nu):
    K_tbcc = N_elf

    # --- 1. ELF Options ---
    elf_options = []
    if m > 0:
        for middle in product([0, 1], repeat=m - 1):
            poly_str = "1" + "".join(map(str, middle)) + "1"
            elf_options.append({"K": K_elf, "N": N_elf, "M": m, "polynomial": poly_str})
    else:
        elf_options.append({"K": K_elf, "N": N_elf, "M": m, "polynomial": "1"})

    # --- 2. TBCC Options ---
    tbcc_base_polys = []
    for freedom_bits in product([0, 1], repeat=nu - 1):
        # Construct binary string
        bin_str = "1" + "".join(map(str, freedom_bits)) + "1"
        # Convert binary string to octal string for your config format
        octal_val = oct(int(bin_str, 2))[2:]
        tbcc_base_polys.append(octal_val)

    # Order doesn't matter, and p1 != p2: 32C2 = 496 combinations
    tbcc_options = []
    num_skipped = 0
    for p1, p2 in combinations(tbcc_base_polys, 2):
        if gcd_gf2(int(p1, 8), int(p2, 8)) != 1:
            num_skipped += 1
            continue

        tbcc_options.append(
            {"K": K_tbcc, "N": N_tbcc, "V": nu, "gen_poly_1": p1, "gen_poly_2": p2}
        )
    print(f"Skipped {num_skipped} catastrophic combinations.")

    # --- 3. Enumerate All Combinations ---
    symmetric_polys = set()
    skipped_polys = 0
    elf_tbcc_configs = []
    for b, t in product(elf_options, tbcc_options):
        elf_oct = oct(int(b["polynomial"], 2))[2:]
        key = triple(elf_oct, t["gen_poly_1"], t["gen_poly_2"], m, nu)
        if key in symmetric_polys:
            skipped_polys += 1
            continue
        symmetric_polys.add(key)

        # Filename now includes the specific ELF polynomial string
        filename = (
            f"elf_p{b['polynomial']}_"
            f"tbcc_v{t['V']}_g{t['gen_poly_1']}_{t['gen_poly_2']}.npy"
        )

        elf_tbcc_configs.append(
            {"bch_config": b, "tbcc_config": t, "filename": filename}
        )

    print(f"Skipped Polynomials (symmetric): {skipped_polys}")
    print(f"ELF Variations: {len(elf_options)}")
    print(f"TBCC Variations: {len(tbcc_options)}")
    print(f"Total Unique Configs: {len(elf_tbcc_configs)}")
    return elf_tbcc_configs


def main(config_path: str, batch_idx: int, batch_size: int):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    example_config = {
        "bch_config": config["bch_config"],
        "tbcc_config": config["tbcc_config"],
    }
    print(f"Example config: {example_config}")

    K_elf = config["bch_config"]["K"]
    N_elf = config["bch_config"]["N"]
    m = config["bch_config"]["M"]
    N_tbcc = config["tbcc_config"]["N"]
    nu = config["tbcc_config"]["V"]

    elf_tbcc_configs = gen_all_elf_tbcc(K_elf, N_elf, m, N_tbcc, nu)
    elf_tbcc_configs.append(example_config)

    total_configs = len(elf_tbcc_configs)

    # --- Slicing logic for large configuration sets ---
    if total_configs > batch_size:
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, total_configs)

        if start_idx >= total_configs:
            print(
                f"Requested batch_idx {batch_idx} is out of bounds for total configs {total_configs}."
            )
            return

        print(
            f"Total configs ({total_configs}) > limit ({batch_size}). Processing batch {batch_idx}: elements [{start_idx} to {end_idx-1}]"
        )
        elf_tbcc_configs = elf_tbcc_configs[start_idx:end_idx]

        # Adjust base filename so concurrent runs don't overwrite or block the same HDF5 file
        base_filename = f"k{K_elf}n{N_tbcc}v{nu}_batch{batch_idx}"
    else:
        print(
            f"Total configs ({total_configs}) is within the threshold limit. Running all configs."
        )
        base_filename = f"k{K_elf}n{N_tbcc}v{nu}_all"

    file_path = Path(f"output/{base_filename}").with_suffix(".h5")
    with h5py.File(file_path, "w") as f:
        pass
    print(f"Writing results to {file_path}")

    target_ebno_dB = config["target_EbNo_dB"]
    target_ebno_linear = 10 ** (0.1 * target_ebno_dB)
    target_esno_linear = target_ebno_linear * (K_elf / N_tbcc)

    # Initialize winner
    best_dsu_pcw = 1
    best_code_config = elf_tbcc_configs[0]

    for i, code_config in enumerate(elf_tbcc_configs):

        if i % 1000 == 0:
            print(f"Local Batch Processed: {i}/{len(elf_tbcc_configs)}")

        As, W_weight, D, basis, num_trellis_stages = setup_A_Wbit_D(code_config)
        A_shape = As[0].shape
        O_y, O_x = A_shape
        max_shift_per_stage = 2
        max_X = O_x + max_shift_per_stage * num_trellis_stages

        d_W = cuda.to_device(W_weight)
        d_D = cuda.to_device(D)
        d_spectrum = cuda.to_device(np.zeros(max_X, dtype=np.uint64))

        # implement cpu gate
        USE_CPU = O_y <= 1024

        if USE_CPU:
            cpu_spectrum = np.zeros(max_X, dtype=np.uint64)
            for i_stream, A in enumerate(As):
                result = A.copy()
                for stage in range(num_trellis_stages):
                    result = trellisStep_shift(result, W_weight, D, max_shift_per_stage)
                padded = np.zeros((result.shape[0], max_X), dtype=np.uint64)
                padded[:, : result.shape[1]] = result
                cpu_spectrum += padded[basis[i_stream], :]

        for i_stream, A in enumerate(As):
            O_y, O_x = A.shape
            starting_state = int(np.nonzero(A)[0][0])

            cuda_lib.launchFoldshiftPipeline(
                starting_state,
                O_y,
                O_x,
                d_W.device_ctypes_pointer.value,
                W_weight.shape[0],
                W_weight.shape[1],
                d_D.device_ctypes_pointer.value,
                D.shape[0],
                D.shape[1],
                num_trellis_stages,
                max_shift_per_stage,
                max_X,
                int(basis[i_stream]),
                d_spectrum.device_ctypes_pointer.value,
            )

        cuda.synchronize()
        gpu_spectrum = d_spectrum.copy_to_host()

        if gpu_spectrum[0] == 1:  # only consider "good" codes
            spectra = dist_spectra(
                num_cwds=gpu_spectrum, hamming_dist=np.arange(len(gpu_spectrum))
            )

            dsub_pcw = dsu(spectra, target_esno_linear)
            config_str = (
                f"BCH_poly{code_config['bch_config']['polynomial']}_"
                f"TBCC_{code_config['tbcc_config']['gen_poly_1']}_"
                f"{code_config['tbcc_config']['gen_poly_2']}"
            )

            with h5py.File(file_path, "a") as f:
                grp = f.require_group(config_str)

                if "dsub_pcw" in grp:
                    del grp["dsub_pcw"]
                if "gpu_spectrum" in grp:
                    del grp["gpu_spectrum"]

                grp.create_dataset("dsub_pcw", data=dsub_pcw)
                grp.create_dataset(
                    "gpu_spectrum", data=gpu_spectrum, compression="gzip"
                )

            if dsub_pcw < best_dsu_pcw:
                best_dsu_pcw = dsub_pcw
                best_code_config = code_config

    print(f"Batch {batch_idx} Completed.")
    print(f"Best DSU P_cw in this batch: {best_dsu_pcw:4e}")
    print(f"Best code_config in this batch: {best_code_config}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="Path to YAML config file")
    parser.add_argument(
        "--batch_idx", type=int, default=0, help="Which 10k chunk to run (0, 1, 2...)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=10000,
        help="Maximum size of configurations evaluated per script call",
    )
    args = parser.parse_args()

    main(args.config, args.batch_idx, args.batch_size)
