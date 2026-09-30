"""Plot DSU performance CDFs at 6.5 dB for available K11/N30 searches."""

from collections import Counter
from pathlib import Path

import h5py
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler


matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "Liberation Sans"]

gem12_colors = [
    "#0072BD", "#D95319", "#EDB120", "#7E2F8E", "#77AC30", "#4DBEEE",
    "#A2142F", "#003E67", "#722C0D", "#7C5D10", "#42194B", "#3E5A19",
]
plt.rcParams["axes.prop_cycle"] = cycler(color=gem12_colors)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULT_FILES = {
    2: PROJECT_ROOT / "output" / "k11n30m4v2_gridsearch.h5",
    3: PROJECT_ROOT / "output" / "k11n30m4v3_gridsearch.h5",
    4: PROJECT_ROOT / "output" / "k11n30m4v4_gridsearch.h5",
    5: PROJECT_ROOT / "output" / "k11n30m4v5_gridsearch_merged.h5",
    6: PROJECT_ROOT / "output" / "k11n30m4v6_gridsearch_merged.h5",
}


def process_pcw_data(h5_file: Path, precision: int = 8) -> np.ndarray:
    """Return ``(P_cw, count)`` samples aggregated by rounded log10 value."""
    if not h5_file.exists():
        raise FileNotFoundError(f"Grid-search result file not found: {h5_file}")

    values = []
    with h5py.File(h5_file, "r") as result_file:
        for group in result_file.values():
            if "dsub_pcw" not in group:
                continue
            value = float(group["dsub_pcw"][()])
            if value > 0:
                values.append(value)

    if not values:
        return np.empty((0, 2), dtype=float)

    rounded_logs = [round(np.log10(value), precision) for value in values]
    counts = Counter(rounded_logs)
    representatives = {}
    for key, value in zip(rounded_logs, values):
        representatives.setdefault(key, value)

    return np.asarray(
        [(representatives[key], count) for key, count in counts.items()], dtype=float
    )


def plot_pcw_cdf(data_by_nu: dict[int, np.ndarray]) -> Path:
    """Save and display the V2-V6 K11/N30 CDF figure."""
    fig, ax = plt.subplots(figsize=(7, 5))

    for nu, data in data_by_nu.items():
        if data.size == 0:
            continue

        sorted_data = data[np.argsort(data[:, 0])]
        values = sorted_data[:, 0]
        counts = sorted_data[:, 1]
        cdf = np.cumsum(counts) / counts.sum()
        if not np.all(np.diff(cdf) >= 0) or not np.isclose(cdf[-1], 1.0):
            raise ValueError(f"Invalid CDF constructed for nu={nu}")

        ax.step(
            np.insert(values, 0, values[0]),
            np.insert(cdf, 0, 0.0),
            where="post",
            linewidth=2.5,
            label=rf"$\nu={nu}, m=4$",
        )
        ax.fill_between(
            np.insert(values, 0, values[0]),
            np.insert(cdf, 0, 0.0),
            step="post",
            alpha=0.05,
        )

    ax.legend(loc="upper left")
    ax.set_xscale("log")
    ax.set_xlim(1e-7, 1e-2)
    ax.set_ylim(0, 1.05)
    ax.grid(True, which="both", linestyle="--", linewidth=0.3)
    ax.set_xlabel(r"$P_{cw}$ at $6.5$ dB", fontsize=15)
    ax.set_ylabel(r"Cumulative Probability $F(x)$", fontsize=15)
    fig.tight_layout()

    output_path = Path(__file__).with_name("figures") / "n30k11_operating_points_cdf.pdf"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.show()
    return output_path


def main():
    data_by_nu = {nu: process_pcw_data(path) for nu, path in RESULT_FILES.items()}
    plot_pcw_cdf(data_by_nu)


if __name__ == "__main__":
    main()
