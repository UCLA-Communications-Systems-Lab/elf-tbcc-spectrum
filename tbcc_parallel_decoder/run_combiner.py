import argparse
import time
import numpy as np
from utils.yaml_loader import load_config, build_metric_tensors
from cpu_combiner import MetaStage, reduce_tree, best_tailbiting_state, traceback

N_REPS = 20

def bench(fn, *args, reps=N_REPS):
    fn(*args)  # warm-up
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn(*args)
        times.append((time.perf_counter() - t0) * 1000)
    return np.mean(times), np.min(times), np.max(times)


def tensors_to_metastages(tensor: np.ndarray) -> list[MetaStage]:
    """Convert (N, M, M) numpy array into a list of N MetaStage objects."""
    N, M, _ = tensor.shape
    return [
        MetaStage(k=1, num_states=M, M=tensor[i].tolist(), argmin=[[-1]*M for _ in range(M)])
        for i in range(N)
    ]


def build_stages(left: np.ndarray, right: np.ndarray) -> list[MetaStage]:
    """Interleave left and right tensors into a flat list of MetaStages."""
    N, M, _ = left.shape
    stages = []
    for i in range(N):
        stages.append(MetaStage(k=1, num_states=M, M=left[i].tolist(),  argmin=[[-1]*M for _ in range(M)]))
        stages.append(MetaStage(k=1, num_states=M, M=right[i].tolist(), argmin=[[-1]*M for _ in range(M)]))
    return stages


def run_cpu(left: np.ndarray, right: np.ndarray):
    """His logic end to end — no GPU involved."""
    stages = build_stages(left, right)
    final = reduce_tree(stages)
    best_state, best_metric = best_tailbiting_state(final)
    path = traceback(stages, best_state)
    return final, best_state, best_metric, path


def run_cuda(left: np.ndarray, right: np.ndarray, M: int):
    from utils.cuda_driver import launch_combine_cuda
    
    # keep original stages for traceback
    original_stages = build_stages(left, right)
    
    combined, _ = launch_combine_cuda(left, right, M)
    stages = tensors_to_metastages(combined)
    final = reduce_tree(stages)
    best_state, best_metric = best_tailbiting_state(final)
    
    # traceback on original stages, same as CPU
    path = traceback(original_stages, best_state)
    return final, best_state, best_metric, path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--yaml", required=True)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--compare", action="store_true", default=True)
    p.add_argument("--reps", type=int, default=N_REPS)
    args = p.parse_args()

    cfg = load_config(args.yaml)
    left, right, M, N = build_metric_tensors(cfg, seed=42)

    print(f"\nConfig:  {args.yaml}")
    print(f"Stages:  {N}   States: {M}   Tensor shape: {left.shape}")
    print(f"Reps:    {args.reps} (+ 1 warm-up each)")
    print("-" * 52)

    # CPU path
    mean_cpu, min_cpu, max_cpu = bench(run_cpu, left, right, reps=args.reps)
    final_cpu, best_state_cpu, best_metric_cpu, path_cpu = run_cpu(left, right)

    print(f"CPU (tree reduce)")
    print(f"  mean: {mean_cpu:.3f} ms   min: {min_cpu:.3f} ms   max: {max_cpu:.3f} ms")
    print(f"  best state:  {best_state_cpu}")
    print(f"  best metric: {best_metric_cpu:.4f}")
    print(f"  path: {path_cpu}")

    if not args.cpu:
        try:
            mean_cuda, min_cuda, max_cuda = bench(run_cuda, left, right, M, reps=args.reps)
            final_cuda, best_state_cuda, best_metric_cuda, path_cuda = run_cuda(left, right, M)

            print(f"\nCUDA (GPU combines + tree reduce)")
            print(f"  mean: {mean_cuda:.3f} ms   min: {min_cuda:.3f} ms   max: {max_cuda:.3f} ms")
            print(f"  speedup (mean): {mean_cpu/mean_cuda:.2f}x")
            print(f"  best state:  {best_state_cuda}")
            print(f"  best metric: {best_metric_cuda:.4f}")
            print(f"  path: {path_cuda}")

            if args.compare:
                print("-" * 52)
                # Compare best state and path — safe across normalization differences
                states_match = best_state_cpu == best_state_cuda
                path_match   = path_cpu == path_cuda

                if states_match and path_match:
                    print("Comparison: match")
                    print(f"  best state: {best_state_cpu} (both agree)")
                    print(f"  path: {path_cpu}")
                else:
                    if not states_match:
                        print(f"Comparison: best state mismatch  (CPU: {best_state_cpu}, CUDA: {best_state_cuda})")
                    if not path_match:
                        print(f"Comparison: path mismatch")
                        print(f"  CPU  path: {path_cpu}")
                        print(f"  CUDA path: {path_cuda}")

        except Exception as e:
            print(f"\nCUDA failed: {e}")
            print("Falling back to CPU-only mode.")

    print()

if __name__ == "__main__":
    main()