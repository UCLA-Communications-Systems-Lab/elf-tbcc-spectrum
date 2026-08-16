"""Command-line tools for GPU grid searches and their HDF5 result files."""

import argparse
from pathlib import Path

import numpy as np

from code_design import run_grid_search, search_output_path


def merge_batches(output_dir, prefix):
    """Merge all ``<prefix>_batch*.h5`` files into one HDF5 result file."""
    import h5py

    output_dir = Path(output_dir)
    batch_files = sorted(output_dir.glob(f"{prefix}_batch*.h5"))
    if not batch_files:
        raise FileNotFoundError(f"No batch files found for prefix '{prefix}' in {output_dir}")

    merged_path = output_dir / f"{prefix}_merged.h5"
    with h5py.File(merged_path, "w") as merged_file:
        for batch_path in batch_files:
            with h5py.File(batch_path, "r") as batch_file:
                for group_name in batch_file:
                    if group_name not in merged_file:
                        batch_file.copy(group_name, merged_file)
    return merged_path


def report_results(result_path, pattern=None, atol=1e-12):
    """Return minimum-DSU records, or every record matching an explicit pattern."""
    import h5py

    result_path = Path(result_path)
    records = []
    with h5py.File(result_path, "r") as result_file:
        for name in sorted(result_file):
            if pattern and pattern not in name:
                continue
            group = result_file[name]
            if "dsub_pcw" not in group or "gpu_spectrum" not in group:
                continue
            records.append(
                {
                    "name": name,
                    "dsub_pcw": float(group["dsub_pcw"][()]),
                    "spectrum": group["gpu_spectrum"][:],
                }
            )

    if not records:
        return []

    # A pattern is a targeted lookup, not another minimization pass.
    if pattern:
        return records
    minimum = min(record["dsub_pcw"] for record in records)
    return [
        record
        for record in records
        if np.isclose(record["dsub_pcw"], minimum, rtol=0, atol=atol)
    ]


def print_report(result_path, records, pattern=None):
    """Print a compact, reproducible report for search minima."""
    if not records:
        label = f" matching '{pattern}'" if pattern else ""
        print(f"No valid result records{label} found in {result_path}.")
        return

    print(f"Result file: {result_path}")
    if pattern:
        print(f"Configurations matching '{pattern}': {len(records)}")
    else:
        print(f"Minimum DSU P_cw: {records[0]['dsub_pcw']:.8e}")
        print(f"Matching configurations: {len(records)}")
    for record in records:
        spectrum = record["spectrum"]
        nonzero = np.flatnonzero(spectrum[1:])
        dmin = int(nonzero[0] + 1) if len(nonzero) else None
        nearest_neighbors = int(spectrum[dmin]) if dmin is not None else None
        print(f"{record['name']}: d_min={dmin}, A_dmin={nearest_neighbors}")
        print(f"  spectrum={spectrum.tolist()}")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="run one GPU grid-search batch")
    run.add_argument("config", help="YAML configuration path")
    run.add_argument("--output-dir", default="output", help="HDF5 result directory")
    run.add_argument("--batch-index", type=int, default=0)
    run.add_argument("--batch-size", type=int, default=10000)
    run.add_argument("--cyclic", action="store_true", help="search only cyclic ELF polynomials")
    run.add_argument("--label", default="gridsearch", help="result-name suffix")

    merge = commands.add_parser("merge", help="merge grid-search batch HDF5 files")
    merge.add_argument("output_dir", help="directory containing batch result files")
    merge.add_argument("prefix", help="result prefix, e.g. k11n30v5_gridsearch")

    report = commands.add_parser("report", help="report minimum-DSU search results")
    report.add_argument("result_file", help="HDF5 result file")
    report.add_argument("--pattern", help="optional substring matched against HDF5 group names")
    report.add_argument("--atol", type=float, default=1e-12)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.command == "run":
        result = run_grid_search(
            args.config,
            output_dir=args.output_dir,
            batch_index=args.batch_index,
            batch_size=args.batch_size,
            cyclic_only=args.cyclic,
            label=args.label,
        )
        if result:
            print(f"Result file: {result}")
    elif args.command == "merge":
        print(f"Merged result file: {merge_batches(args.output_dir, args.prefix)}")
    else:
        print_report(args.result_file, report_results(args.result_file, args.pattern, args.atol), args.pattern)


if __name__ == "__main__":
    main()
