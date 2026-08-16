# Repository Guidelines

## Project Structure & Module Organization

- `distance_spectrum_computation/` contains the main Python and CUDA distance-spectrum pipeline. YAML experiment definitions live in `config/`, CUDA sources in `compile/` and the package root, plotting scripts in `dsu_bound_plots/`, and generated shared libraries in `lib/`.
- `tbcc_parallel_decoder/` contains the CPU reference decoder, CUDA combiner, command-line runner, utilities, and decoder configuration.
- `cyclic_codes_lyndon_approach/` provides a C++20/pybind11 encoder (`src/`, `include/`) plus Lyndon-word data and analysis notebooks.
- `makefile-example.ipynb` and other notebooks are exploratory examples. Do not commit notebook outputs or generated build artifacts unless they are intentional results.

## Build, Test, and Development Commands

Create an isolated Python environment, then install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

From `distance_spectrum_computation/`, run `make CONFIG=config/k11n30v6.yaml` to select the appropriate CUDA kernel, build `lib/libfoldshift.so`, and execute the matching verification script. Use `make clean` to remove that library. CUDA builds require `nvcc`; CGBN cases also require `CGBN_INCLUDE` and GMP.

Run the decoder's portable reference path from `tbcc_parallel_decoder/` with:

```bash
python run_combiner.py --yaml config/k51n126v6.yaml --cpu
```

Build the pybind11 module with `cmake -S cyclic_codes_lyndon_approach -B cyclic_codes_lyndon_approach/build` followed by `cmake --build cyclic_codes_lyndon_approach/build`.

## Coding Style & Naming Conventions

Use four-space indentation and PEP 8 conventions for Python: `snake_case` functions and modules, `PascalCase` classes, and uppercase constants. Keep imports grouped and prefer `pathlib.Path` for new filesystem code. Match existing CUDA/C++ formatting, use descriptive kernel names, and retain C++20 compatibility. YAML configurations follow `k{K}n{N}v{V}.yaml`, for example `k21n62v6.yaml`.

## Testing Guidelines

There is no centralized test runner or coverage threshold. Treat `test_foldshift.py` and `test_cgbn.py` as GPU integration checks; they compare GPU results with CPU results for tractable state spaces and validate spectrum size/counts. For algorithm changes, run at least one small CPU-comparable configuration. For decoder changes, run `run_combiner.py` both with `--cpu` and, when CUDA is available, without it to confirm matching states and paths.

## Commit & Pull Request Guidelines

History favors short, imperative, scoped subjects such as `feat:`, `fix:`, `bugfix:`, `refactor:`, `chore:`, and `config:`. Keep commits focused and name affected code parameters when relevant. Pull requests should explain the algorithm or configuration change, list exact commands and hardware used for validation, link related issues, and include plots or benchmark output when numerical behavior or performance changes.
