import argparse
import time
import numpy as np
from utils.yaml_loader import load_config, build_metric_tensors
from cpu_combiner import combine_trellis_stages_cpu

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--yaml", required=True, help="Path to TBCC config yaml")
    p.add_argument("--cpu", action="store_true", help="Run ONLY CPU version")
    p.add_argument("--compare", action="store_true", default=True, help="Compare results of CPU and CUDA (default: True)")
    args = p.parse_args()

    # 1. Load config and generate data
    cfg = load_config(args.yaml)
    left, right, M, N = build_metric_tensors(cfg, seed=42)
    print(f"Loaded config: {args.yaml}")
    print(f"Stages: {N}, States: {M}")

    # 2. Run CPU version
    start_cpu = time.time()
    cpu_out, cpu_argmin = combine_trellis_stages_cpu(left, right)
    t_cpu = time.time() - start_cpu
    print(f"CPU (Numba) finished in {t_cpu:.3f}s")

    if not args.cpu:
        # 3. Run CUDA version (via compiled shared library)
        try:
            from utils.cuda_driver import launch_combine_cuda
            start_cuda = time.time()
            cuda_out, cuda_argmin = launch_combine_cuda(left, right, M)
            t_cuda = time.time() - start_cuda
            print(f"CUDA (Compiled .so) finished in {t_cuda:.3f}s")
            print(f"Speedup: {t_cpu / t_cuda:.2f}x")

            if args.compare:
                # 4. Verification
                is_close = np.allclose(cpu_out, cuda_out, atol=1e-5)
                is_equal = np.array_equal(cpu_argmin, cuda_argmin)
                if is_close and is_equal:
                    print("✅ Comparison Success: CPU and CUDA results match.")
                else:
                    if not is_close:
                        print("❌ Comparison Failure: Metric mismatch.")
                    if not is_equal:
                        print("❌ Comparison Failure: Argmin mismatch.")
        except Exception as e:
            print(f"⚠️ CUDA execution failed or not available: {e}")
            print("Running in CPU fallback mode.")

if __name__ == "__main__":
    main()
