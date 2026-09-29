# Distance Spectrum Computation

This directory computes distance spectra for ELF–TBCC codes. The portable CPU path is suitable for small configurations; CUDA is required for GPU spectrum and grid-search runs.

## Local CPU Checks

Install the Python dependencies, activate the project environment, then run a CPU-only spectrum check:

```bash
conda activate codex-research
cd distance_spectrum_computation
python test_foldshift.py config/k11n22v3.yaml --cpu
python test_foldshift.py config/k4n6v1.yaml --cpu
python -m unittest -v test_puncturing.py test_rate_support.py test_gridsearch.py
```

Example: a rate-1/3 YAML uses `gen_polys: ["13", "15", "17"]`; the number of polynomials determines the TBCC output rate.

## Puncturing

Add `tbcc_config.puncture_pattern` to select which generator outputs are
transmitted. Each **row** belongs to the generator at the same index in
`gen_polys`; each **column** is one trellis stage in the repeating period.
`1` retains that output bit and `0` removes it. For example:

```yaml
tbcc_config:
  K: 4
  N: 6
  V: 1
  gen_polys: ["3", "1"]
  puncture_pattern:
    - [1, 1, 0, 1]  # generator 0: omit stage 2
    - [1, 0, 1, 1]  # generator 1: omit stage 1
```

The four stage columns retain respectively 2, 1, 1, and 2 bits. Column 0
repeats at stage 4. `tbcc_config.N` is the **transmitted** codeword length:
the sum of retained bits over all `bch_config.K + bch_config.M` stages. In
this example `N = 6`, and the distance spectrum is
`[1, 1, 2, 6, 5, 1, 0]`. A column of all zeros is allowed: its transitions
still update the trellis state, but add zero Hamming weight. With no
`puncture_pattern`, all generator bits are transmitted as before.

The CPU and CUDA implementations select the corresponding precomputed
branch-weight slice on each existing stage. CUDA shared memory keeps the
same size; its weights are reloaded from that stage's slice. The standard
GPU grid search supports `bch_config.K < 64`; use the CGBN runner for larger
single configurations.

On a CUDA host, verify the example with both kernels:

```bash
conda activate codex-research
cd distance_spectrum_computation
make compile-kernel
python test_foldshift.py config/k4n6v1.yaml
python gridsearch.py run config/k4n6v1.yaml --output-dir /tmp/puncture-check
make compile-cgbn CGBN_INCLUDE=/path/to/CGBN/include
python test_cgbn.py config/k4n6v1.yaml
```

## CUDA and Colab Setup

CUDA builds require `nvcc`; CGBN runs additionally need CGBN and GMP. On Colab, manually clone the intended branch before opening `gridsearch.ipynb`:

```bash
git clone --branch dev-spectrum-codex https://github.com/UCLA-Communications-Systems-Lab/tbcc-decoder-test.git /content/tbcc-decoder-test
cd /content/tbcc-decoder-test/distance_spectrum_computation/compile
bash foldshift_compile.sh
```

The notebook does not clone or pull Git repositories. Its configuration cell creates a YAML file in `config/` from editable `K`, `ELF_MEMORY`, `TBCC_MEMORY`, `RATE_DENOMINATOR`, and optional fixed-polynomial variables. Leave `ELF_POLYNOMIAL` and `GEN_POLYS` as `None` to search all candidates.

## Grid Search Commands

Run a search on a CUDA host. Results default to the chosen output directory and use the `gridsearch` suffix:

```bash
cd distance_spectrum_computation
python gridsearch.py run config/k11n30v5.yaml --output-dir /content/drive/MyDrive/TBCC_Results
python gridsearch.py run config/k11n30v5.yaml --output-dir /content/drive/MyDrive/TBCC_Results --batch-index 0 --batch-size 500
```

Merge batch results and inspect the minimum DSU result:

```bash
python gridsearch.py merge /content/drive/MyDrive/TBCC_Results k11n30v5_gridsearch
python gridsearch.py report /content/drive/MyDrive/TBCC_Results/k11n30v5_gridsearch_merged.h5
```

An unbatched run writes `k11n30v5_gridsearch.h5`; batches write `k11n30v5_gridsearch_batch0.h5`; merging writes `k11n30v5_gridsearch_merged.h5`. Each HDF5 configuration group contains `dsub_pcw` and `gpu_spectrum`.
Punctured runs add a mask fingerprint to the filename and store the
generator-major mask and both mother and transmitted lengths as HDF5
attributes. Merge batches only when these attributes agree.
