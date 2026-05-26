import numpy as np
import sys
import argparse
from run_combiner import run_cpu, run_cuda

def run_testcase(N, M, seed=42, print_all_mismatches=False, left=None, right=None):
    print(f"=== Test Case: N={N}, M={M}, seed={seed} ===")
    
    if left is None or right is None:
        np.random.seed(seed)
        # Generate random left and right metric tensors
        # shape: (N, M, M)
        left = np.random.rand(N, M, M).astype(np.float32)
        right = np.random.rand(N, M, M).astype(np.float32)
    
    # run CPU
    final_cpu, best_state_cpu, best_metric_cpu, path_cpu = run_cpu(left, right)
    
    # run CUDA
    try:
        final_cuda, best_state_cuda, best_metric_cuda, path_cuda = run_cuda(left, right, M)
    except Exception as e:
        print(f"CUDA failed with exception: {e}")
        return

    cpu_M = np.array(final_cpu.M)
    cuda_M = np.array(final_cuda.M)
    
    print("\nCPU Final Matrix:")
    print(cpu_M)
    
    print("\nCUDA Final Matrix:")
    print(cuda_M)
    
    # We compare with a small tolerance due to potential floating point differences
    # particularly with normalizations
    atol = 1e-4
    match = np.allclose(cpu_M, cuda_M, atol=atol)
    
    if match:
        print("\nResult: MATCHES")
    else:
        print("\nResult: MISMATCH")
        
        mismatches = []
        nan_inf_mismatches = []
        for i in range(M):
            for j in range(M):
                if not np.isclose(cpu_M[i, j], cuda_M[i, j], atol=atol):
                    mismatches.append((i, j))
                elif np.isnan(cpu_M[i, j]) != np.isnan(cuda_M[i, j]) or np.isinf(cpu_M[i, j]) != np.isinf(cuda_M[i, j]):
                    nan_inf_mismatches.append((i, j))
                    
        total_mismatches = len(mismatches) + len(nan_inf_mismatches)
        print(f"Number of mismatched positions: {total_mismatches}")
        
        if print_all_mismatches:
            if len(mismatches) > 0:
                print("Mismatched positions (row, col) | CPU Value | CUDA Value | Difference:")
                for (i, j) in mismatches:
                    diff = abs(cpu_M[i, j] - cuda_M[i, j])
                    print(f"Position ({i}, {j}) | CPU: {cpu_M[i, j]:.6f} | CUDA: {cuda_M[i, j]:.6f} | Diff: {diff:.6f}")
            if len(nan_inf_mismatches) > 0:
                print("NaN/Inf mismatched positions:")
                for (i, j) in nan_inf_mismatches:
                    print(f"NaN/Inf mismatch at ({i}, {j}) | CPU: {cpu_M[i, j]} | CUDA: {cuda_M[i, j]}")
                        
    print("\n")

def main():
    parser = argparse.ArgumentParser(description="Run combiner test cases.")
    parser.add_argument("--print-all-mismatches", action="store_true", help="Print all mismatched positions if a mismatch occurs.")
    parser.add_argument("--yaml", type=str, default=None, help="Path to YAML config file")
    args = parser.parse_args()

    if args.yaml:
        from utils.yaml_loader import load_config, build_metric_tensors
        cfg = load_config(args.yaml)
        left, right, M, N = build_metric_tensors(cfg, seed=42)
        print(f"Loaded config from {args.yaml}")
        run_testcase(N, M, seed=42, print_all_mismatches=args.print_all_mismatches, left=left, right=right)
    else:
        # Generate 24 test cases with N in [2, 4, 8, 16, 32, 64] and M in [8, 16, 32, 64]
        test_cases = [
            (N, M)
            for N in [2, 4, 8, 16, 32, 64]
            for M in [8, 16, 32, 64]
        ]
        
        for (N, M) in test_cases:
            run_testcase(N, M, print_all_mismatches=args.print_all_mismatches)
        
if __name__ == "__main__":
    main()
