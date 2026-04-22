# fmt: off
import os, ctypes
import numpy as np
import yaml
from numba import cuda
from setup import setup_A_Wbit_D
from step import trellisStep_shift
from itertools import product, combinations

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


def gen_all_elf_nu6_tbcc():

    # --- 1. ELF Options (8 total) ---
    # Polynomial M=4 (5 bits): 1 [3 bits freedom] 1
    elf_options = []
    for middle in product([0, 1], repeat=3):
        poly_str = "1" + "".join(map(str, middle)) + "1"
        elf_options.append({"K": 11, "N": 15, "M": 4, "polynomial": poly_str})

    # --- 2. TBCC Options (nu=6) ---
    # nu=6 means 7-bit polynomials. 
    # "5 bits freedom" implies the first and last bits are fixed to 1.
    # Format: 1 [5 bits freedom] 1
    tbcc_base_polys = []
    for freedom_bits in product([0, 1], repeat=5):
        # Construct binary string
        bin_str = "1" + "".join(map(str, freedom_bits)) + "1"
        # Convert binary string to octal string for your config format
        octal_val = oct(int(bin_str, 2))[2:]
        tbcc_base_polys.append(octal_val)

    # Order doesn't matter, and p1 != p2: 32C2 = 496 combinations
    tbcc_options = []
    for p1, p2 in combinations(tbcc_base_polys, 2):
        tbcc_options.append({
            "K": 15, 
            "N": 30, 
            "V": 6, 
            "gen_poly_1": p1, 
            "gen_poly_2": p2
        })

    # --- 3. Enumerate All Combinations ---
    elf_tbcc_configs = []
    for b, t in product(elf_options, tbcc_options):
        # Filename now includes the specific ELF polynomial string
        filename = (
            f"elf_p{b['polynomial']}_"
            f"tbcc_v{t['V']}_g{t['gen_poly_1']}_{t['gen_poly_2']}.npy"
        )
        
        elf_tbcc_configs.append({
            "bch_config": b,
            "tbcc_config": t,
            "filename": filename
        })

    print(f"ELF Variations: {len(elf_options)}")
    print(f"TBCC Variations: {len(tbcc_options)}")
    print(f"Total Unique Configs: {len(elf_tbcc_configs)}")
    print(elf_tbcc_configs[0])

def main():
    
    elf_tbcc_configs = gen_all_elf_nu6_tbcc()[:10]

    # elf_options = [
    #     {"K": 11, "N": 15, "M": 4, "polynomial": "10011"}
    # ]

    # tbcc_options = [
    #     {"K": 15, "N": 30, "V": 3, "gen_poly_1": "13", "gen_poly_2": "17"},
    #     # {"K": 15, "N": 30, "V": 4, "gen_poly_1": "27", "gen_poly_2": "31"},
    #     # {"K": 15, "N": 30, "V": 5, "gen_poly_1": "53", "gen_poly_2": "75"},
    #     # {"K": 15, "N": 30, "V": 6, "gen_poly_1": "133", "gen_poly_2": "171"},
    #     # {"K": 15, "N": 30, "V": 7, "gen_poly_1": "247", "gen_poly_2": "371"},
    #     # {"K": 15, "N": 30, "V": 8, "gen_poly_1": "561", "gen_poly_2": "753"},
    #     # {"K": 15, "N": 30, "V": 9, "gen_poly_1": "1131", "gen_poly_2": "1537"},
    #     # {"K": 15, "N": 30, "V": 10, "gen_poly_1": "2473", "gen_poly_2": "3217"},
    #     # {"K": 15, "N": 30, "V": 11, "gen_poly_1": "4325", "gen_poly_2": "6747"},
    #     # {"K": 15, "N": 30, "V": 12, "gen_poly_1": "10627", "gen_poly_2": "16765"},
    #     # {"K": 15, "N": 30, "V": 13, "gen_poly_1": "27251", "gen_poly_2": "37363"},
    #     # {"K": 15, "N": 30, "V": 14, "gen_poly_1": "75063", "gen_poly_2": "56711"}
    # ]

    # elf_tbcc_configs = []

    # for b, t in product(elf_options, tbcc_options):
    #     # 1. Create the descriptive filename
    #     # We prefix with 'b' for BCH/ELF and 't' for TBCC to avoid confusion
    #     filename = (
    #         f"elf_k{b['K']}_n{t['N']}_m{b['M']}_"
    #         f"tbcc_v{t['V']}_g{t['gen_poly_1']}_{t['gen_poly_2']}.npy"
    #     )
        
    #     # 2. Store the config along with its filename
    #     elf_tbcc_configs.append({
    #         "bch_config": b,
    #         "tbcc_config": t,
    #         "filename": filename
    # })
    
    for code_config in elf_tbcc_configs:

      As, W_weight, D, basis, num_trellis_stages = setup_A_Wbit_D(code_config)
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
      np.save("output/fold/" + code_config['filename'], gpu_spectrum)

if __name__ == "__main__":
    main()
