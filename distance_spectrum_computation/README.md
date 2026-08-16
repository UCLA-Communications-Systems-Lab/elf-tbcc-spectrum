# Distance Spectrum Computation

This directory computes distance spectra for ELF–TBCC codes. The portable CPU path is suitable for small configurations; CUDA is required for GPU spectrum and grid-search runs.

## Local CPU Checks

Install the Python dependencies, activate the project environment, then run a CPU-only spectrum check:

```bash
conda activate research
cd distance_spectrum_computation
python test_foldshift.py config/k11n22v3.yaml --cpu
python -m unittest -v test_rate_support.py test_gridsearch.py
```

Example: a rate-1/3 YAML uses `gen_polys: ["13", "15", "17"]`; the number of polynomials determines the TBCC output rate.

## CUDA and Colab Setup

CUDA builds require `nvcc`; CGBN searches additionally need CGBN and GMP. On Colab, manually clone the intended branch before opening `gridsearch.ipynb`:

```bash
git clone --branch dev-spectrum-codex https://github.com/UCLA-Communications-Systems-Lab/elf-tbcc-spectrum.git /content/tbcc-decoder-test
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
