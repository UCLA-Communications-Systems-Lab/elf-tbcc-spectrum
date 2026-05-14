import h5py
import numpy as np
import argparse
from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib

# 1. Fix the Font Type (Type 42 is TrueType, avoids the Type 3 error)
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

# 2. Set font family to Sans-Serif (like MATLAB's Helvetica/Arial)
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "Liberation Sans"]
from cycler import cycler

gem12_colors = [
    "#0072BD",  # Blue
    "#D95319",  # Orange
    "#EDB120",  # Yellow
    "#7E2F8E",  # Purple
    "#77AC30",  # Green
    "#4DBEEE",  # Light Blue
    "#A2142F",  # Dark Red
    "#003E67",  # Navy Blue
    "#722C0D",  # Burnt Orange/Brown
    "#7C5D10",  # Olive
    "#42194B",  # Dark Purple
    "#3E5A19",  # Forest Green
]
plt.rcParams["axes.prop_cycle"] = cycler(color=gem12_colors)


def process_pcw_data(filename: str, results_path: str = "data", precision: int = 8):
    """
    Processes H5 files and returns values mapped to their occurrence counts.
    """
    results_dir = Path(results_path)
    h5_file = (results_dir / filename).with_suffix(".h5")

    if not h5_file.exists():
        print(f"Error: File {h5_file} not found.")
        return None

    all_raw_values = []

    # 1. Extraction
    with h5py.File(h5_file, "r") as f:
        for config_name in f.keys():
            try:
                val = f[config_name]["dsub_pcw"][()]
                if val > 0:
                    all_raw_values.append(float(val))
            except (KeyError, TypeError):
                continue

    if not all_raw_values:
        print("No valid dsub_pcw data found in file.")
        return np.array([])

    # 2. Identification logic (Log-Domain)
    # We map the rounded log-key back to the first raw value we encountered
    log_rounded = [round(np.log10(v), precision) for v in all_raw_values]
    counts = Counter(log_rounded)

    # To preserve the actual values (not the logs), we map key -> representative raw value
    val_map = {}
    for i, key in enumerate(log_rounded):
        if key not in val_map:
            val_map[key] = all_raw_values[i]

    # 3. Final Output Construction
    # We create a structured NumPy array: [value, count]
    output_data = []
    for key, count in counts.items():
        original_val = val_map[key]
        output_data.append((original_val, count))

    # Returning as a NumPy array for easy manipulation
    # Shape: (N, 2) where column 0 is the value and column 1 is the frequency
    return np.array(output_data)


def plot_pcw_cdf(data_list, labels=None, x_min=1e-7, x_max=1e-2):
    if isinstance(data_list, np.ndarray):
        data_list = [data_list]

    plt.figure(figsize=(7, 5))

    # Lists to store handles and labels for the legend
    handles_list = []
    labels_list = []
    best_code_handle = None

    for i, data in enumerate(data_list):
        if data.size == 0:
            continue

        sorted_idx = np.argsort(data[:, 0])
        sorted_data = data[sorted_idx]
        values = sorted_data[:, 0]
        counts = sorted_data[:, 1]

        y_raw = np.cumsum(counts) / np.sum(counts)
        plot_x = np.insert(values, 0, values[0])
        plot_y = np.insert(y_raw, 0, 0)

        label = labels[i] if labels and i < len(labels) else rf"$\nu={i+3}, m=4$"

        # 4. Plot Curve - Capture the handle (line)
        (line,) = plt.step(plot_x, plot_y, where="post", linewidth=2.5, label=label)
        plt.fill_between(plot_x, plot_y, step="post", alpha=0.05)

        handles_list.append(line)
        labels_list.append(label)

        # 5. Highlight the lowest Pcw point
        # dot = plt.scatter(values[0], 0, color="#020202", s=120, zorder=5)
        # if i == 0:
        #     best_code_handle = dot

    # --- LEGEND REORDERING LOGIC ---

    # Example: Reverse the order (nu=8 down to nu=3)
    # and put "Best Code" at the very bottom
    new_order = [0, 1, 2, 3, 4, 5]  # These are indices of labels_list

    ordered_handles = [handles_list[idx] for idx in new_order]
    ordered_labels = [labels_list[idx] for idx in new_order]

    # Add the "Best Code" dot to the end of the legend
    if best_code_handle:
        ordered_handles.append(best_code_handle)
        ordered_labels.append("Best Code")

    # Apply the ordered legend
    plt.legend(ordered_handles, ordered_labels, loc="upper left")

    # (Keep rest of the scaling and labels code here...)
    plt.xscale("log")
    plt.xlim(x_min, x_max)
    plt.grid(True, which="both", linestyle="--", linewidth=0.3)
    plt.xlabel(r"$P_{cw}$ at $6.5$ dB", fontsize=15)
    plt.ylabel(r"Cumulative Probability $F(x)$", fontsize=15)
    plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.show()


def main():
    # parser = argparse.ArgumentParser(
    #     description="Process PCW values from HDF5 results using log-domain precision."
    # )
    # parser.add_argument(
    #     "filename", type=str, help="Name of the .h5 file (without extension)"
    # )
    # parser.add_argument(
    #     "--path", type=str, default="data", help="Directory containing the file"
    # )
    # parser.add_argument(
    #     "--precision",
    #     type=int,
    #     default=8,
    #     help="Significant decimal places in log10 domain (Default: 8)",
    # )

    # args = parser.parse_args()

    data_list = []
    nu3_data = process_pcw_data("k11n30v3_results")
    nu4_data = process_pcw_data("k11n30v4_results")
    nu5_data = process_pcw_data("k11n30v5_results")
    nu6_data = process_pcw_data("k11n30v6_results")
    nu7_data = process_pcw_data("k11n30v7_results")
    nu8_data = process_pcw_data("k11n30v8_results")
    data_list.append(nu3_data)
    data_list.append(nu4_data)
    data_list.append(nu5_data)
    data_list.append(nu6_data)
    data_list.append(nu7_data)
    data_list.append(nu8_data)

    plot_pcw_cdf(data_list)


if __name__ == "__main__":
    main()
