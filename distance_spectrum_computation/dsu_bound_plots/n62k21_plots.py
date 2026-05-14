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
    hamming_dist: np.array
    num_cwds: np.array


# fmt: off
k21n62_bob_ds = dist_spectra(
    hamming_dist=np.array(
        [0, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 62]
    ),
    num_cwds=np.array(
        [1, 217, 1457, 9207, 30969, 87699, 191301, 315859, 411866, 411866, 315859, 191301, 87699, 30969, 9207, 1457, 217, 1]
    ),
)

k21n62v4_best_dist_spectra = dist_spectra(
    hamming_dist=np.arange(63),
    num_cwds=np.array(
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 93, 372, 837, 2294, 4557, 7595, 14694, 28024, 45167, 65410, 94147, 130758, 163401, 181133, 
    195920, 217280, 213094, 180544, 155899, 132866, 96999, 63829, 42222, 27528, 
    16461, 8618, 4061, 1922, 899, 403, 124, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0]
    ),
)

k21n62v5_best_dist_spectra = dist_spectra(
    hamming_dist=np.arange(63),
    num_cwds=np.array([
    1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    62, 310, 930, 2046, 4278, 8742, 15128, 27993, 
    46035, 62682, 91574, 131564, 163308, 183706, 200012, 
    217435, 211482, 178746, 153326, 131502, 98518, 65162, 
    42655, 28303, 16213, 8246, 4154, 1736, 744, 310, 
    124, 93, 31, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
]),
)
# fmt: on

N = 62
K = 21
R = K / N
# EbNo
ebno_dB = np.arange(2, 7.1, 0.1)
ebno_linear = 10 ** (0.1 * ebno_dB)
# EsNo
esno_linear = ebno_linear * R
esno_dB = 10 * np.log10(esno_linear)
# Es/sigma^2
es_over_sigma_sqrd_linear = esno_linear * 2
es_over_sigma_sqrd_dB = 10 * np.log10(es_over_sigma_sqrd_linear)

## - bounds
dsub_k21n62 = dsu(k21n62_bob_ds, esno_linear)
dsub_k21n62v4_best = dsu(k21n62v4_best_dist_spectra, esno_linear)
dsub_k21n62v5_best = dsu(k21n62v5_best_dist_spectra, esno_linear)

# SP59
k21n62_sp59 = np.loadtxt("data/k21n62_sp59.csv", delimiter=",")

# RCU
k21n62_rcu = np.loadtxt("data/k21n62_rcu.csv", delimiter=",")

fig, ax = plt.subplots(figsize=(7, 5))
ax.set_yscale("log")

(dsub62,) = plt.semilogy(
    ebno_dB,
    dsub_k21n62,
    marker="o",
    markerfacecolor="none",
    linewidth=1.5,
    markevery=5,
    label=r"$m=10, g_e=3551, \nu=6, g_1=133, g_2=171, A_{16}=217$",
)
(dsub_best_nu4,) = plt.semilogy(
    ebno_dB,
    dsub_k21n62v4_best,
    marker="o",
    markerfacecolor="none",
    linewidth=1.5,
    markevery=5,
    label=r"$m=10, g_e=3013, \nu=4, g_1=23, g_2=35, A_{16}=93$",
)
(dsub_best_nu5,) = plt.semilogy(
    ebno_dB,
    dsub_k21n62v5_best,
    marker="o",
    markerfacecolor="none",
    linewidth=1.5,
    markevery=5,
    label=r"$m=10, g_e=3557, \nu=5, g_1=53, g_2=75, A_{16}=62$",
)
(sp59,) = plt.semilogy(
    k21n62_sp59[:, 1],
    k21n62_sp59[:, 3],
    linewidth=1.5,
    label=f"Sphere Packing Bound",
)

(rcu,) = plt.semilogy(
    k21n62_rcu[:, 1],
    k21n62_rcu[:, 3],
    linewidth=1.5,
    linestyle="--",
    label=f"Random Coding Union Bound",
)
plt.fill_between(
    k21n62_rcu[:, 1],
    k21n62_sp59[:, 3],
    k21n62_rcu[:, 3],
    color="skyblue",
    alpha=0.4,
    label="Area Between",
)

legend = ax.legend(
    handles=[
        rcu,
        dsub62,
        dsub_best_nu4,
        dsub_best_nu5,
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
