# ELF-TBCC Spectrum & Parallel Decoder

This repo contains two folders, one for computing distance spectrums across different code configurations, and one for a TBCC decoder implementation for different codes.

---

## Distance Spectrum Computation

Computes the distance spectrum for ELF-TBCC concatenated codes across the configurations defined in `config/`.

Follow the notebook inside `distance_spectrum_computation/` to build and run.

Output spectrums are saved as `.npy` files under `output/fold/`.

---

## TBCC Parallel Decoder

A C++ reference TBCC decoder to verify trellis next-states.

Follow `tbcc_parallel_decoder/cpp_decoder/cpp_decoder_test.ipynb` to build and run.

---

## Requirements

A CUDA-capable GPU is required (`sm_75` / `sm_80` / `sm_86` / `sm_89`).

```bash
pip install -r requirements.txt
```