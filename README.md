# TBCC Decoder Test

## What this code does and how it works 

Decoding a Tail-Biting Convolutional Code (TBCC) involves running a Viterbi-like search through a trellis — a graph where each node is a decoder state and each edge carries a branch metric (a cost). To combine two adjacent trellis stages into one, you need to find, for every (source state `s`, destination state `d`) pair, the best intermediate state `r` that minimizes the total path cost:

```
output[s, d] = min over r of ( left[s, r] + right[r, d] )
```

The kernel `combiner.cu` uses a min-plus matrix product which has the same structure as regular matrix multiply but with (min, +) instead of (multiply, add).

Inputs: `left` and `right` are M×M matrices of branch metrics for the two stages being merged. Currently these are filled with random float32 values as placeholder.
 
Outputs: The combined M×M metric matrix (the min-plus result), and an M×M argmin matrix recording which intermediate state `r` achieved each minimum. The argmin is needed for traceback reconstructing the most likely transmitted sequence.

The GPU version parallelizes this. Each CUDA block handles one stage pair. Within a block, up to M² threads each take one `(s, d)` entry and race through the M intermediate states in parallel. Bringing wall-clock time down to roughly O(M) per combine step.

After combining, each output matrix is normalized by subtracting its minimum value to prevent metrics from growing unboundedly across many merges.

`run_combiner.py` runs both CPU and GPU versions on the same input, times them, and checks the outputs match.

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