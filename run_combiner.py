import argparse
import time
import numpy as np
from utils.yaml_loader import load_config, build_metric_tensors
from cpu_combiner import combine_trellis_stages_cpu

N_REPS = 20  # timed repetitions after warm-up

def bench(fn, *args, reps=N_REPS):
    """Warm up once, then time reps runs. Returns (mean_ms, min_ms, max_ms)."""
    fn(*args)  # warm-up
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn(*args)
        times.append((time.perf_counter() - t0) * 1000)
    return np.mean(times), np.min(times), np.max(times)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--yaml", required=True, help="Path to TBCC config yaml")
    p.add_argument("--cpu", action="store_true", help="Run ONLY CPU version")
    p.add_argument("--compare", action="store_true", default=True, help="Compare CPU and CUDA results (default: True)")
    p.add_argument("--reps", type=int, default=N_REPS, help=f"Timed repetitions after warm-up (default: {N_REPS})")
    args = p.parse_args()

    cfg = load_config(args.yaml)
    left, right, M, N = build_metric_tensors(cfg, seed=42)

    print(f"\nConfig:  {args.yaml}")
    print(f"Stages:  {N}   States: {M}   Tensor shape: {left.shape}")
    print(f"Reps:    {args.reps} (+ 1 warm-up each)")
    print("-" * 52)

    # CPU benchmark
    mean_cpu, min_cpu, max_cpu = bench(combine_trellis_stages_cpu, left, right, reps=args.reps)
    cpu_out, cpu_argmin = combine_trellis_stages_cpu(left, right)

    print(f"CPU (Numba JIT)")
    print(f"  mean: {mean_cpu:.3f} ms   min: {min_cpu:.3f} ms   max: {max_cpu:.3f} ms")
    print(f"  per stage (mean): {mean_cpu/N:.3f} ms")

    if not args.cpu:
        try:
            from utils.cuda_driver import launch_combine_cuda

            mean_cuda, min_cuda, max_cuda = bench(launch_combine_cuda, left, right, M, reps=args.reps)
            cuda_out, cuda_argmin = launch_combine_cuda(left, right, M)

            print(f"\nCUDA (compiled .so)")
            print(f"  mean: {mean_cuda:.3f} ms   min: {min_cuda:.3f} ms   max: {max_cuda:.3f} ms")
            print(f"  per stage (mean): {mean_cuda/N:.3f} ms")
            print(f"  speedup (mean): {mean_cpu/mean_cuda:.2f}x   speedup (min/min): {min_cpu/min_cuda:.2f}x")

            if args.compare:
                print("-" * 52)
                is_close = np.allclose(cpu_out, cuda_out, atol=1e-5)
                is_equal = np.array_equal(cpu_argmin, cuda_argmin)

                if is_close and is_equal:
                    max_diff = np.max(np.abs(cpu_out - cuda_out))
                    print(f"Comparison: match  (max |diff| = {max_diff:.2e})")
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
