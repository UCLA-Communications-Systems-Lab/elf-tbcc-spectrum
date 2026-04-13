import argparse
import time
import numpy as np
from utils.yaml_loader import load_config, build_metric_tensors
from cpu_combiner import combine_trellis_stages_cpu

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--yaml", required=True, help="Path to TBCC config yaml")
    p.add_argument("--cpu", action="store_true", help="Run ONLY CPU version")
    p.add_argument("--compare", action="store_true", default=True, help="Compare CPU and CUDA results (default: True)")
    args = p.parse_args()

    cfg = load_config(args.yaml)
    left, right, M, N = build_metric_tensors(cfg, seed=42)

    print(f"\nConfig:  {args.yaml}")
    print(f"Stages:  {N}   States: {M}   Tensor shape: {left.shape}")
    print("-" * 48)

    # CPU run
    start_cpu = time.time()
    cpu_out, cpu_argmin = combine_trellis_stages_cpu(left, right)
    t_cpu = time.time() - start_cpu

    print(f"CPU (Numba JIT)")
    print(f"  Total:     {t_cpu*1000:.2f} ms")
    print(f"  Per stage: {t_cpu*1000/N:.2f} ms")

    if not args.cpu:
        try:
            from utils.cuda_driver import launch_combine_cuda

            # warm-up pass so first-call JIT overhead doesn't pollute timing
            _ = launch_combine_cuda(left, right, M)

            start_cuda = time.time()
            cuda_out, cuda_argmin = launch_combine_cuda(left, right, M)
            t_cuda = time.time() - start_cuda

            print(f"\nCUDA (compiled .so)")
            print(f"  Total:     {t_cuda*1000:.2f} ms")
            print(f"  Per stage: {t_cuda*1000/N:.2f} ms")
            print(f"  Speedup:   {t_cpu/t_cuda:.2f}x")

            if args.compare:
                print("-" * 48)
                is_close = np.allclose(cpu_out, cuda_out, atol=1e-5)
                is_equal = np.array_equal(cpu_argmin, cuda_argmin)

                if is_close and is_equal:
                    max_diff = np.max(np.abs(cpu_out - cuda_out))
                    print(f"Comparison: matches (max |diff| = {max_diff:.2e})")
                else:
                    if not is_close:
                        max_diff = np.max(np.abs(cpu_out - cuda_out))
                        print(f"Comparison: metric mismatch  (max |diff| = {max_diff:.2e})")
                    if not is_equal:
                        n_wrong = np.sum(cpu_argmin != cuda_argmin)
                        print(f"Comparison: argmin mismatch  ({n_wrong}/{cpu_argmin.size} entries differ)")

        except Exception as e:
            print(f"\nCUDA failed: {e}")
            print("Falling back to CPU-only mode.")

    print()

if __name__ == "__main__":
    main()
