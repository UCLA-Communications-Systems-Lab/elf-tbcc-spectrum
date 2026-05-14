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


# fmt: off
k21n62_dist_spectra = dist_spectra(
        crc="11101101001",
        hamming_dist=np.array(
            [0, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 62]
        ),
        num_cwds=np.array(
            [1, 217, 1457, 9207, 30969, 87699, 191301, 315859, 411866, 411866, 315859, 191301, 87699, 30969, 9207, 1457, 217, 1]
        ),
        dmin=16,
    )
# fmt: on

N = 126
K = 51
R = K / N
# EbNo
ebno_dB = np.arange(2, 5.6, 0.25)
ebno_linear = 10 ** (0.1 * ebno_dB)
# EsNo
esno_linear = ebno_linear * R
esno_dB = 10 * np.log10(esno_linear)
# Es/sigma^2
es_over_sigma_sqrd_linear = esno_linear * 2
es_over_sigma_sqrd_dB = 10 * np.log10(es_over_sigma_sqrd_linear)

## - bounds
# dsu
dsu_v6_filename = "data/k51n126v6_ds.npy"
dsu_v6_ds = np.load(dsu_v6_filename)
k51n126_dist_spectra = dist_spectra(
    crc="1010100111001",
    hamming_dist=np.arange(0, dsu_v6_ds.shape[0]),
    num_cwds=dsu_v6_ds,
    dmin=14,
)
dsub_k51n126v6 = dsu(k51n126_dist_spectra, esno_linear)

dsu_v8_filename = "data/k51n126v8_ds.npy"
dsu_v8_ds = np.load(dsu_v8_filename)
k51n126v8_dist_spectra = dist_spectra(
    crc="1010100111001",
    hamming_dist=np.arange(0, dsu_v8_ds.shape[0]),
    num_cwds=dsu_v8_ds,
    dmin=18,
)
dsub_k51n126v8 = dsu(k51n126v8_dist_spectra, esno_linear)

# SP59
k51n126_sp59 = np.loadtxt("data/k51n126_sp59.csv", delimiter=",")

# RCU
k51n126_rcu = np.loadtxt("data/k51n126_rcu.csv", delimiter=",")

fig, ax = plt.subplots(figsize=(7, 5))
ax.set_yscale("log")

(dsubv6,) = plt.semilogy(
    ebno_dB,
    dsub_k51n126v6,
    marker="o",
    markerfacecolor="none",
    linewidth=1.5,
    markevery=2,
    label=r"$m=12, g_e(x)=13071, \nu=6, g_1=133, g_2=171, A_{16}=52$",
)
(dsubv8,) = plt.semilogy(
    ebno_dB,
    dsub_k51n126v8,
    marker="o",
    markerfacecolor="none",
    linewidth=1.5,
    markevery=2,
    label=r"$m=12, g_e(x)=13071, \nu=8, g_1=561, g_2=753, A_{18}=4$",
)
(sp59,) = plt.semilogy(
    k51n126_sp59[:, 1],
    k51n126_sp59[:, 3],
    linewidth=1.5,
    label=f"Sphere Packing Bound",
)

(rcu,) = plt.semilogy(
    k51n126_rcu[:, 1],
    k51n126_rcu[:, 3],
    linewidth=1.5,
    linestyle="--",
    label=f"Random Coding Union Bound",
)
plt.fill_between(
    k51n126_rcu[:, 1],
    k51n126_sp59[:, 3],
    k51n126_rcu[:, 3],
    color="skyblue",
    alpha=0.4,
    label="Area Between",
)

legend = ax.legend(
    handles=[
        rcu,
        dsubv6,
        dsubv8,
        sp59,
    ],
    loc="upper right",
    fontsize=12,
    framealpha=0.5,
)
ax.add_artist(legend)


plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.xlim([2, 7])
plt.ylim([1e-9, 1e-2])
plt.xlabel(r"$\frac{E_b}{N_o} (\mathrm{dB})$", fontsize=15)
plt.ylabel(r"Probability of codeword error, $P_{cw}$", fontsize=15)
plt.tight_layout()
plt.show()
