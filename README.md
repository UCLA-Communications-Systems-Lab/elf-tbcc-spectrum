# TBCC Decoder Test

## What This Code Does

`combiner.cu` implements a min-plus matrix product — for every (s, d) pair, find the intermediate state r that minimizes the total path cost across two trellis stages:
```
output[s, d] = min over r of ( left[s, r] + right[r, d] )
```

Inputs: left and right are M×M matrices of branch metrics for the two stages being merged. Currently filled with random float32 values as placeholders.

Outputs: The combined M×M metric matrix, and an M×M argmin matrix recording which r achieved each minimum. The argmin is used for the traceback.

The GPU version assigns one CUDA block per stage pair, with up to M² threads each handling one (s, d) entry in parallel. Each output matrix is normalized by subtracting its block minimum to prevent metric overflow across many merges.

## Files Included in Project
- `run_combiner.py` : the entry point. Loads the config, generates the input tensors, runs both implementations, prints timing, and compares outputs.
- `combiner.cu` : the CUDA kernel. Compiled once with `compile.sh` into `lib/libtrellis.so`.
- `utils/cuda_driver.py` : loads `libtrellis.so` via ctypes, allocates GPU memory with Numba, passes raw device pointers to the C launcher, and copies results back.
- `cpu_combiner.py` : the ground truth. A Numba CPU implementation of the same min-plus product, used to verify the GPU output is correct.
- `utils/yaml_loader.py` : reads the TBCC config and builds the psuedo-random input tensors.
- `config/k11n22v3.yaml` : example code parameters (K=11, N=22, constraint length V=3).

---

## Setup

```bash
pip install numpy numba pyyaml
```

---

## Build

```bash
bash compile.sh
```

Compiles `combiner.cu` → `lib/libtrellis.so`. Default target is `sm_89` (RTX 40-series). Change the `-gencode` flag in `compile.sh` for your GPU:

- T4 / RTX 20-series → `sm_75`
- A100 → `sm_80`
- RTX 30-series → `sm_86`
- RTX 40-series → `sm_89`

---

## Run code

```bash
# GPU + CPU, compare results
python run_combiner.py --yaml config/k11n22v3.yaml

# CPU only
python run_combiner.py --yaml config/k11n22v3.yaml --cpu
```

---


## Running on Google Colab (for the free T4 GPU)


**Step 1 Change runtime to GPU**

`Runtime → Change runtime type → T4 GPU`

**Step 2 Verify GPU is available**

```python
!nvidia-smi
!nvcc --version
```

Both should print output. If `nvidia-smi` says command not found, you forgot Step 1.

**Step 3 — Mount Drive and unzip**

```python
from google.colab import drive
drive.mount('/content/drive')

!unzip "/content/drive/MyDrive/<path-to-your-zip>/tbcc-decoder-test.zip" -d /content/
%cd /content/tbcc-decoder-test
```

**Step 4 — Install dependencies**

```python
!pip install numba numpy pyyaml
```

**Step 5 — Compile the kernel**

Colab's free tier uses a T4 (`sm_75`), so compile for that explicitly:

```python
!mkdir -p lib
!nvcc -shared -Xcompiler -fPIC -O3 -gencode arch=compute_75,code=sm_75 combiner.cu -o lib/libtrellis.so

# Verify it was created
import os
print(os.path.exists("lib/libtrellis.so"))  # should print True
```

**Step 6 — Run**

```python
!python run_combiner.py --yaml config/k11n22v3.yaml
```