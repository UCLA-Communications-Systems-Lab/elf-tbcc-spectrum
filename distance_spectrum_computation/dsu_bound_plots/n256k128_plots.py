from dataclasses import dataclass
from matplotlib.patches import Ellipse
from typing import Optional
from bounds import dsu

import numpy as np
from scipy import special, integrate
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


@dataclass
class dist_spectra:
    crc: str
    hamming_dist: np.array
    num_cwds: np.array
    dmin: int


N = 256
K = 128
R = K / N
# EbNo
ebno_dB = np.arange(1, 3.5, 0.25)
ebno_linear = 10 ** (0.1 * ebno_dB)
# EsNo
esno_linear = ebno_linear * R
esno_dB = 10 * np.log10(esno_linear)
# Es/sigma^2
es_over_sigma_sqrd_linear = esno_linear * 2
es_over_sigma_sqrd_dB = 10 * np.log10(es_over_sigma_sqrd_linear)

## - bounds
# dsu
# TODO


# SP59
k128n256_sp59 = np.loadtxt("data/k128n256_sp59.csv", delimiter=",")

# RCU
k128n256_rcu = np.loadtxt("data/k128n256_rcu.csv", delimiter=",")

fig, ax = plt.subplots(figsize=(7, 5))
ax.set_yscale("log")

(sp59,) = plt.semilogy(
    k128n256_sp59[:, 1],
    k128n256_sp59[:, 3],
    linewidth=1.5,
    label=f"Sphere Packing Bound",
)

(rcu,) = plt.semilogy(
    k128n256_rcu[:, 1],
    k128n256_rcu[:, 3],
    linewidth=1.5,
    linestyle="--",
    label=f"Random Coding Union Bound",
)
plt.fill_between(
    k128n256_rcu[:, 1],
    k128n256_sp59[:, 3],
    k128n256_rcu[:, 3],
    color="skyblue",
    alpha=0.4,
    label="Area Between",
)

legend = ax.legend(
    handles=[
        rcu,
        sp59,
    ],
    loc="upper right",
    fontsize=12,
    framealpha=0.5,
)
ax.add_artist(legend)

plt.title(rf"Reference curves for $({N}, {K})$ Block Code")
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.xlim([1, 3.5])
plt.ylim([1e-9, 1e-2])
plt.xlabel(r"$\frac{E_b}{N_o} (\mathrm{dB})$", fontsize=15)
plt.ylabel(r"Probability of codeword error, $P_{cw}$", fontsize=15)
plt.tight_layout()
plt.show()
