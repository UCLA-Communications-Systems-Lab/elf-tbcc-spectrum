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
ShortCode_5G = dist_spectra(
    crc="1",
    hamming_dist=np.arange(31),
    num_cwds=np.array([1, 0, 0, 0, 0, 0, 0, 0, 5, 32, 61, 112, 175, 224, 270, 288, 270, 224, 175, 112, 61, 32, 5, 0, 0, 0, 0, 0, 0, 0, 1]),
    dmin=8,
)

AppleProp_6G = dist_spectra(
    crc="1",
    hamming_dist=np.arange(31),
    num_cwds=np.array([1,0,0,0,0,0,0,0,0,0,66,240,190,0,255,544,255,0,190,240,66,0,0,0,0,0,0,0,0,0,1]),
    dmin=10
)

BestNu4 = dist_spectra(
    crc="11111",
    hamming_dist=np.arange(31),
    num_cwds=np.array([1,0,0,0,0,0,0,0,0,0,138,0,390,0,495,0,495,0,390,0,138,0,0,0,0,0,0,0,0,0,1]),
    dmin=10
)

hamming_distance = np.array([0, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 30])

CRC_11111 = dist_spectra(
    crc="11111",
    hamming_dist=hamming_distance,
    num_cwds=np.array([1, 30, 15, 33, 315, 630, 630, 315, 33, 15, 30, 1]),
    dmin=6,
)
CRC_11001 = dist_spectra(
    crc="11001",
    hamming_dist=hamming_distance,
    num_cwds=np.array([1, 45, 45, 258, 270, 405, 405, 270, 258, 45, 45, 1]),
    dmin=6,
)
CRC_10011 = dist_spectra(
    crc="10011",
    hamming_dist=hamming_distance,
    num_cwds=np.array([1, 0, 30, 108, 300, 585, 585, 300, 108, 30, 0, 1]),
    dmin=8,
)
ELF_10101 = dist_spectra(
    crc="10101",
    hamming_dist=hamming_distance,
    num_cwds=np.array([1, 0, 20, 103, 360, 561, 511, 361, 100, 31, 0, 0]),
    dmin=8,
)
# fmt: on


N = 30
K = 11
R = K / N
# EbNo
ebno_dB = np.arange(1, 9.1, 0.1)
ebno_linear = 10 ** (0.1 * ebno_dB)
# EsNo
esno_linear = ebno_linear * R
esno_dB = 10 * np.log10(esno_linear)
# Es/sigma^2
es_over_sigma_sqrd_linear = esno_linear * 2
es_over_sigma_sqrd_dB = 10 * np.log10(es_over_sigma_sqrd_linear)

## - bounds
dsub_10011 = dsu(CRC_10011, esno_linear)
dsub_10101 = dsu(ELF_10101, esno_linear)
dsub_5g = dsu(ShortCode_5G, esno_linear)
dsub_apple = dsu(AppleProp_6G, esno_linear)
dsub_bestnu4 = dsu(BestNu4, esno_linear)

# SP59
k11n30_sp59 = np.loadtxt("data/k11n30_sp59.csv", delimiter=",")

# RCU
k11n30_rcu = np.loadtxt("data/k11n30_rcu.csv", delimiter=",")

fig, ax = plt.subplots(figsize=(7, 5))
ax.set_yscale("log")

(dsu_elf25,) = plt.semilogy(
    ebno_dB,
    dsub_10101,
    linewidth=1.5,
    marker="o",
    markerfacecolor="none",
    markevery=5,
    label=r"$m=4, g_e=25, \nu=6, g_1=133, g_2=171, A_{8}=20$",
)

(dsu_crc23,) = plt.semilogy(
    ebno_dB,
    dsub_10011,
    linewidth=1.5,
    marker="D",
    markerfacecolor="none",
    markevery=5,
    label=r"$m=4, g_e=23, \nu=6, g_1=133, g_2=171, A_{8}=30$",
)

(dsu_etsi,) = plt.semilogy(
    ebno_dB,
    dsub_5g,
    linewidth=1.5,
    marker="d",
    markerfacecolor="none",
    markevery=5,
    label=r"5G ETSI, $A_{8}=5$",
)
(dsu_bestnu4,) = plt.semilogy(
    ebno_dB,
    dsub_bestnu4,
    linewidth=1.5,
    marker="*",
    markerfacecolor="none",
    markevery=5,
    label=r"$m=4, g_e=23, \nu=4, g_1=23, g_2=25, A_{10}=138$",
)
(dsu_apple,) = plt.semilogy(
    ebno_dB,
    dsub_apple,
    linewidth=1.5,
    marker="p",
    markerfacecolor="none",
    markevery=5,
    label=r"Apple proposal, $A_{10}=66$",
)
(sp59,) = plt.semilogy(
    k11n30_sp59[:, 1],
    k11n30_sp59[:, 3],
    linewidth=1.5,
    label=f"Sphere Packing Bound",
)

(rcu,) = plt.semilogy(
    k11n30_rcu[:, 1],
    k11n30_rcu[:, 3],
    linewidth=1.5,
    linestyle="--",
    label=f"Random Coding Union Bound",
)
plt.fill_between(
    k11n30_rcu[:, 1],
    k11n30_sp59[:, 3],
    k11n30_rcu[:, 3],
    color="skyblue",
    alpha=0.4,
    label="Area Between",
)

legend = ax.legend(
    handles=[
        rcu,
        dsu_crc23,
        dsu_elf25,
        dsu_etsi,
        dsu_bestnu4,
        dsu_apple,
        sp59,
    ],
    loc="upper right",
    fontsize=12,
    framealpha=0.5,
)
ax.add_artist(legend)


plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.xlim([4, 9])
plt.ylim([1e-9, 1e-2])
plt.xlabel(r"$\frac{E_b}{N_o} (\mathrm{dB})$", fontsize=15)
plt.ylabel(r"Probability of codeword error, $P_{cw}$", fontsize=15)
plt.tight_layout()
plt.show()
