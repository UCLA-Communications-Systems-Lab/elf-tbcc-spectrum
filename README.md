# TBCC Decoder Test Suite

A lightweight test harness for the parallel trellis-stage combiner described in the ISIT paper. Combines pairs of trellis stages via a min-plus matrix product, with a compiled CUDA kernel for GPU acceleration and a Numba CPU implementation for verification.

---

## Setup

```bash
pip install numpy numba pyyaml
```

If you want the GPU path, you also need the CUDA toolkit with `nvcc` on your PATH.

---

## Build the CUDA kernel

```bash
bash compile.sh
```

This compiles `combiner.cu` into `lib/libtrellis.so`. The default target is `sm_89` (RTX 40-series). Change the `-gencode` flag in `compile.sh` if you have a different GPU:

- RTX 20-series / T4 → `sm_75`
- A100 → `sm_80`
- RTX 30-series → `sm_86`
- RTX 40-series → `sm_89`

**Windows:** Run the script inside **Git Bash** or **WSL**. Or run `nvcc` directly in PowerShell:
```powershell
mkdir -Force lib
nvcc -shared -Xcompiler -fPIC -O3 -gencode arch=compute_89,code=sm_89 combiner.cu -o lib/libtrellis.so
```

---

## Run

```bash
# GPU + CPU, compare results (default)
python run_combiner.py --yaml config/k11n22v3.yaml

# CPU only (skip CUDA even if GPU is present)
python run_combiner.py --yaml config/k11n22v3.yaml --cpu
```

If the compiled `.so` is missing or no GPU is available, the script automatically falls back to CPU and prints a warning.

Typical output:
```
Loaded config: config/k11n22v3.yaml
Stages: 11, States: 8
CPU (Numba) finished in 0.012s
CUDA (Compiled .so) finished in 0.004s
Speedup: 3.00x
Comparison Success: CPU and CUDA results match.
```

---

## Project Structure

```
├── combiner.cu              # CUDA kernel + extern "C" launcher
├── compile.sh               # builds lib/libtrellis.so
├── run_combiner.py          # main driver: CPU + CUDA paths, timing, comparison
├── cpu_combiner.py          # Numba @njit CPU reference implementation
├── combiner_test.py         # standalone Numba CUDA prototype + NumPy reference
├── utils/
│   ├── yaml_loader.py       # loads config YAML and builds input tensors
│   └── cuda_driver.py       # ctypes bridge into the compiled .so
├── config/
│   └── k11n22v3.yaml        # example TBCC configuration
└── lib/
    └── libtrellis.so        # compiled output (generated — not tracked by git)
```

---

## How it works

The kernel computes a min-plus matrix product over `N` independent stage pairs:

```
output[i, s, d] = min over r of ( left[i, s, r] + right[i, r, d] )
```

One CUDA block handles one stage pair. Each block loads its M×M matrices into shared memory, finds the best intermediate state `r` for every `(s, d)` pair, then normalizes by subtracting the block minimum to prevent metric overflow over many iterations.

---

## Standalone benchmark

```bash
python combiner_test.py
```

Runs the Numba CUDA kernel prototype and a pure NumPy reference side by side and checks they match.

---

## No GPU? Use Google Colab

Set runtime to T4 GPU (`Runtime > Change runtime type > T4 GPU`), then:

```python
!git clone https://github.com/UCLA-Communications-Systems-Lab/tbcc-decoder-test
%cd tbcc-decoder-test
!pip install numba numpy pyyaml
!bash compile.sh
!python run_combiner.py --yaml config/k11n22v3.yaml
```