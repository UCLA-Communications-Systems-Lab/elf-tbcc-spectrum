from dataclasses import dataclass
import os, ctypes
from pathlib import Path
import numpy as np
import yaml
from setup import setup_A_Wbit_D
from step import trellisStep_shift
from itertools import combinations, product
from cyclic import divides_xN_minus_1
import argparse

def load_cuda_library():
    """Load the CUDA implementation when running a GPU design sweep."""
    lib_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "lib", "libfoldshift.so")
    )
    if not os.path.exists(lib_path):
        raise FileNotFoundError(
            f"Shared library not found: {lib_path}. Run foldshift_compile.sh first."
        )
    cuda_lib = ctypes.CDLL(lib_path)
    cuda_lib.launchFoldshiftPipeline.argtypes = [
        ctypes.c_int,
        ctypes.c_int, ctypes.c_int,
        ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
        ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_void_p,
    ]
    cuda_lib.launchFoldshiftPipeline.restype = None
    return cuda_lib

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


def generator_key(elf_octal, generators, m, nu):
    forward = (elf_octal, *sorted(generators))
    reversed_key = (
        reverse_polynomial(elf_octal, m + 1),
        *sorted(reverse_polynomial(poly, nu + 1) for poly in generators),
    )
    return min(forward, reversed_key)


def polynomial_candidates(nu):
    """Return full-degree and lower-degree octal generator candidates."""
    full_degree = set()
    lower_degree = set()
    for degree in range(1, nu + 1):
        target = full_degree if degree == nu else lower_degree
        for freedom_bits in product([0, 1], repeat=degree - 1):
            bits = "1" + "".join(map(str, freedom_bits)) + "1"
            target.add(oct(int(bits, 2))[2:])
    return sorted(full_degree), sorted(lower_degree | full_degree)


def generators_are_noncatastrophic(generators):
    gcd = int(generators[0], 8)
    for poly in generators[1:]:
        gcd = gcd_gf2(gcd, int(poly, 8))
    return gcd == 1


def gen_all_elf_tbcc(
    K_elf,
    N_elf,
    m,
    N_tbcc,
    nu,
    cyclic_only=False,
    fixed_elf_poly=None,
    fixed_tbcc_polys=None,
):
    K_tbcc = N_elf

    # --- 1. ELF Options ---
    elf_options = []
    if fixed_elf_poly is not None:
        # If an ELF polynomial is explicitly passed, use ONLY that one
        elf_options.append(
            {"K": K_elf, "N": N_elf, "M": m, "polynomial": fixed_elf_poly}
        )
    elif m > 0:
        for middle in product([0, 1], repeat=m - 1):
            poly_str = "1" + "".join(map(str, middle)) + "1"

            if cyclic_only and not divides_xN_minus_1(poly_str, N_elf):
                continue

            elf_options.append({"K": K_elf, "N": N_elf, "M": m, "polynomial": poly_str})
    else:
        elf_options.append({"K": K_elf, "N": N_elf, "M": m, "polynomial": "1"})

    # --- 2. TBCC Options ---
    tbcc_options = []
    if N_tbcc % K_tbcc:
        raise ValueError("tbcc_config.N must be an integer multiple of tbcc_config.K")
    rate_denominator = N_tbcc // K_tbcc
    if rate_denominator not in (2, 3):
        raise ValueError("Only rate-1/2 and rate-1/3 TBCC searches are supported")

    if fixed_tbcc_polys is not None:
        if len(fixed_tbcc_polys) != rate_denominator:
            raise ValueError("gen_polys length must match tbcc_config.N / tbcc_config.K")
        tbcc_options.append(
            {"K": K_tbcc, "N": N_tbcc, "V": nu, "gen_polys": list(fixed_tbcc_polys)}
        )
    else:
        full_degree_polys, all_polys = polynomial_candidates(nu)
        full_degree_set = set(full_degree_polys)
        num_skipped = 0
        if rate_denominator == 2:
            candidates = (
                (p1, p2)
                for p1, p2 in product(full_degree_polys, all_polys)
            )
        else:
            candidates = (
                generators
                for generators in combinations(all_polys, 3)
                if any(poly in full_degree_set for poly in generators)
            )

        for generators in candidates:
            if not generators_are_noncatastrophic(generators):
                num_skipped += 1
                continue
            tbcc_options.append(
                {"K": K_tbcc, "N": N_tbcc, "V": nu, "gen_polys": list(generators)}
            )
        print(f"Skipped {num_skipped} catastrophic combinations.")

    # --- 3. Enumerate Combinations ---
    symmetric_polys = set()
    skipped_polys = 0
    elf_tbcc_configs = []
    for b, t in product(elf_options, tbcc_options):
        elf_oct = oct(int(b["polynomial"], 2))[2:]
        key = generator_key(elf_oct, t["gen_polys"], m, nu)
        if key in symmetric_polys:
            skipped_polys += 1
            continue
        symmetric_polys.add(key)

        filename = (
            f"elf_p{b['polynomial']}_"
            f"tbcc_v{t['V']}_g{'_'.join(t['gen_polys'])}.npy"
        )

        elf_tbcc_configs.append(
            {"bch_config": b, "tbcc_config": t, "filename": filename}
        )

    print(f"Skipped Polynomials (symmetric): {skipped_polys}")
    print(f"ELF Variations: {len(elf_options)}")
    print(f"TBCC Variations: {len(tbcc_options)}")
    print(f"Total Unique Configs: {len(elf_tbcc_configs)}")
    return elf_tbcc_configs


def main(config_path: str, batch_idx: int, batch_size: int, cyclic_only: bool):
    import h5py
    from numba import cuda
    from dsu_bound_plots.bounds import dsu

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    print(f"Loaded config: {config}")
    cuda_lib = load_cuda_library()

    K_elf = config["bch_config"]["K"]
    N_elf = config["bch_config"]["N"]
    m = config["bch_config"]["M"]
    N_tbcc = config["tbcc_config"]["N"]
    nu = config["tbcc_config"]["V"]

    # Check for specific predefined constraints in config
    fixed_elf_poly = config["bch_config"].get("polynomial")
    fixed_tbcc_polys = config["tbcc_config"].get("gen_polys")

    elf_tbcc_configs = gen_all_elf_tbcc(
        K_elf=K_elf,
        N_elf=N_elf,
        m=m,
        N_tbcc=N_tbcc,
        nu=nu,
        cyclic_only=cyclic_only,
        fixed_elf_poly=fixed_elf_poly,
        fixed_tbcc_polys=fixed_tbcc_polys,
    )

    total_configs = len(elf_tbcc_configs)
    if total_configs == 0:
        print("No configurations generated to process.")
        return

    # --- Construct Dynamic Output Filename Based on Constraints ---
    if fixed_elf_poly is not None and fixed_tbcc_polys is not None:
        # Both are explicitly fixed: name it uniquely down to the specific polynomials
        base_name = f"k{K_elf}n{N_tbcc}v{nu}_ELF_{fixed_elf_poly}_TBCC_{'_'.join(fixed_tbcc_polys)}"
    elif fixed_elf_poly is not None:
        # Only ELF is fixed, TBCC is generating
        base_name = f"k{K_elf}n{N_tbcc}v{nu}_ELF_{fixed_elf_poly}"
    elif fixed_tbcc_polys is not None:
        # Only TBCC is fixed, ELF is generating
        base_name = f"k{K_elf}n{N_tbcc}v{nu}_TBCC_{'_'.join(fixed_tbcc_polys)}"
    else:
        # Fully combinatorial sweep
        base_name = f"k{K_elf}n{N_tbcc}v{nu}_all_combos"

    # --- Append batch index if slicing is active ---
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

        base_filename = f"{base_name}_batch{batch_idx}"
    else:
        print(
            f"Total configs ({total_configs}) is within the threshold limit. Running all configs."
        )
        base_filename = base_name

    file_path = Path(f"output/{base_filename}").with_suffix(".h5")
    with h5py.File(file_path, "w") as f:
        pass
    print(f"Writing results to {file_path}")

    target_ebno_dB = config["target_EbNo_dB"]
    target_ebno_linear = 10 ** (0.1 * target_ebno_dB)
    target_esno_linear = target_ebno_linear * (K_elf / N_tbcc)

    best_dsu_pcw = 1
    best_code_config = elf_tbcc_configs[0]

    for i, code_config in enumerate(elf_tbcc_configs):
        if i % 1000 == 0:
            print(f"Local Batch Processed: {i}/{len(elf_tbcc_configs)}")

        As, W_weight, D, basis, num_trellis_stages, num_output_bits = setup_A_Wbit_D(code_config)
        A_shape = As[0].shape
        O_y, O_x = A_shape
        max_shift_per_stage = num_output_bits
        max_X = O_x + max_shift_per_stage * num_trellis_stages

        d_W = cuda.to_device(W_weight)
        d_D = cuda.to_device(D)
        d_spectrum = cuda.to_device(np.zeros(max_X, dtype=np.uint64))

        USE_CPU = O_y <= 1024

        if USE_CPU:
            cpu_spectrum = np.zeros(max_X, dtype=np.uint64)
            for i_stream, A in enumerate(As):
                result = A.copy()
                for stage in range(num_trellis_stages):
                    result = trellisStep_shift(result, W_weight, D, max_shift_per_stage)
                padded = np.zeros((result.shape[0], max_X), dtype=result.dtype)
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

        if gpu_spectrum[0] == 1:
            spectra = dist_spectra(
                num_cwds=gpu_spectrum, hamming_dist=np.arange(len(gpu_spectrum))
            )
            dsub_pcw = dsu(spectra, target_esno_linear)
            config_str = (
                f"BCH_poly{code_config['bch_config']['polynomial']}_"
                f"TBCC_{'_'.join(code_config['tbcc_config']['gen_polys'])}"
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
    # Added action="store_true" flag so that providing it maps to True, and omitting it defaults to False
    parser.add_argument(
        "--cyclic",
        action="store_true",
        help="Filter and only evaluate cyclic ELF configuration matrices",
    )
    args = parser.parse_args()

    # Pass the argument flag to main
    main(args.config, args.batch_idx, args.batch_size, args.cyclic)
