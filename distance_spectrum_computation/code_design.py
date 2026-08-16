from dataclasses import dataclass
import os, ctypes
from pathlib import Path
import numpy as np
import yaml
from setup import setup_A_Wbit_D
from itertools import combinations, product
from cyclic import divides_xN_minus_1

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


def search_output_prefix(config_path, label="gridsearch"):
    """Build a stable result prefix from a YAML file and optional run label."""
    if not label:
        label = "gridsearch"
    return f"{Path(config_path).stem}_{label}"


def search_output_path(output_dir, prefix, batch_index=None):
    """Return the HDF5 path for a complete search or one search batch."""
    suffix = f"_batch{batch_index}" if batch_index is not None else ""
    return Path(output_dir) / f"{prefix}{suffix}.h5"


def run_grid_search(
    config_path,
    output_dir="output",
    batch_index=0,
    batch_size=10000,
    cyclic_only=False,
    label="gridsearch",
):
    """Evaluate one GPU grid-search batch and store its distance spectra in HDF5."""
    import h5py
    from numba import cuda
    from dsu_bound_plots.bounds import dsu

    config_path = Path(config_path)
    output_dir = Path(output_dir)
    with config_path.open("r") as f:
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

    prefix = search_output_prefix(config_path, label)
    batch_output = None
    if total_configs > batch_size:
        start_idx = batch_index * batch_size
        end_idx = min(start_idx + batch_size, total_configs)

        if start_idx >= total_configs:
            print(
                f"Requested batch index {batch_index} is out of bounds for {total_configs} configurations."
            )
            return

        print(
            f"Processing batch {batch_index}: configurations [{start_idx} to {end_idx - 1}] of {total_configs}."
        )
        elf_tbcc_configs = elf_tbcc_configs[start_idx:end_idx]
        batch_output = batch_index
    else:
        print(f"Running all {total_configs} configurations.")

    output_dir.mkdir(parents=True, exist_ok=True)
    file_path = search_output_path(output_dir, prefix, batch_output)
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

    print(f"Grid search completed: {file_path}")
    print(f"Best DSU P_cw in this batch: {best_dsu_pcw:4e}")
    print(f"Best code_config in this batch: {best_code_config}")
    return file_path
