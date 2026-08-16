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
# fmt: off
k51n126BCHBestCC_dist_spectra = dist_spectra(
    crc="1010100111001",
    hamming_dist=np.arange(N+1),
    num_cwds=np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    63, 0, 63, 0, 252, 0, 1197, 0, 4851, 0, 20349, 0, 132237, 0, 
    873306, 0, 6361173, 0, 48135024, 0, 348446385, 0, 2270534966, 
    0, 12902804712, 0, 63414384264, 0, 269590632924, 0, 994034414205, 
    0, 3189554042724, 0, 8933784616890, 0, 21898120941963, 0, 47065738844691, 
    0, 88839605167369, 0, 147455538576471, 0, 215430628660065, 0, 277254878518065,
    0, 314489440728414, 0, 314489440728414, 0, 277254878518065, 0, 215430628660065, 
    0, 147455538576471, 0, 88839605167369, 0, 47065738844691, 0, 21898120941963, 0, 
    8933784616890, 0, 3189554042724, 0, 994034414205, 0, 269590632924, 0, 63414384264, 
    0, 12902804712, 0, 2270534966, 0, 348446385, 0, 48135024, 0, 6361173, 0, 873306, 0, 
    132237, 0, 20349, 0, 4851, 0, 1197, 0, 252, 0, 63, 0, 63, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 1]
    ),
    dmin=14,
)
dsub_k51n126v6BCHBestCC = dsu(k51n126BCHBestCC_dist_spectra, esno_linear)

k51n126v7BCHBestCC_dist_spectra = dist_spectra(
    crc="1010100111001",
    hamming_dist=np.arange(N+1),
    num_cwds=np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 63, 0, 0, 0, 0, 
    252, 315, 1323, 2772, 6132, 15876, 40194, 107982, 303975, 882945, 2633778, 7796124, 
    22377474, 62962410, 171272934, 448608780, 1133567953, 2757968766, 6463420362, 14594285664, 
    31756802868, 66631332474, 134864634159, 263454225678, 496951114653, 905528674008, 1594461188817, 
    2713874319576, 4466521803345, 7109970226956, 10949351619843, 16316707183350, 23533930740789, 
    32859194948622, 44420966543583, 58150634338062, 73726511146677, 90541711834476, 107713092803463, 
    124142696051346, 138625875215079, 149989885505586, 157247654699457, 159742731371978, 157247103730122, 
    149990852373264, 138627955649379, 124143408786726, 107712434799963, 90541403656788, 73726422781689, 
    58149877749390, 44420075217721, 32858985472488, 23534215507341, 16317017022810, 10949610759759, 
    7110212945040, 4466670723795, 2713888278297, 1594394819451, 905443264852, 496881156681, 263403608706, 
    134835137355, 66630117519, 31767535170, 14605352118, 6474978027, 2765440944, 1136605435, 449436600, 
    171045126, 62555493, 21871017, 7355754, 2356263, 723366, 215595, 55510, 14616, 2646, 945, 378, 63, 126,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ),
    dmin=15,
)
dsub_k51n126v7BCHBestCC = dsu(k51n126v7BCHBestCC_dist_spectra, esno_linear)

# ELF=0x1639
dsu_v8_filename = "data/k51n126v8_ds.npy"
dsu_v8_ds = np.load(dsu_v8_filename)
k51n126v8_dist_spectra = dist_spectra(
    crc="1011000111001",
    hamming_dist=np.arange(0, dsu_v8_ds.shape[0]),
    num_cwds=dsu_v8_ds,
    dmin=18,
)
dsub_k51n126v8 = dsu(k51n126v8_dist_spectra, esno_linear)

# ELF=0x1539
k51n126v8ELF1539_dist_spectra = dist_spectra(
    crc="1010100111001",
    hamming_dist=np.arange(0, dsu_v8_ds.shape[0]),
    num_cwds=np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 1071, 0, 5796, 0, 46998, 0, 481644, 0, 4885398, 
    0, 43927821, 0, 341209134, 0, 2270330258, 0, 12943494837, 0, 63534194262, 
    0, 269704652376, 0, 993828487050, 0, 3188780428239, 0, 8933091770925, 0, 
    21899098938150, 0, 47068424385696, 0, 88841104896439, 0, 147452640310251, 
    0, 215425090458144, 0, 277253791111542, 0, 314495212826592, 0, 314495212826592, 
    0, 277253791111542, 0, 215425090458144, 0, 147452640310251, 0, 88841104896439, 0, 
    47068424385696, 0, 21899098938150, 0, 8933091770925, 0, 3188780428239, 0, 
    993828487050, 0, 269704652376, 0, 63534194262, 0, 12943494837, 0, 2270330258, 
    0, 341209134, 0, 43927821, 0, 4885398, 0, 481644, 0, 46998, 0, 5796, 0, 1071, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]),
    dmin=18,
)
dsub_k51n126v8ELF1539 = dsu(k51n126v8ELF1539_dist_spectra, esno_linear)



k51n126v5CyclicBest_dist_spectra = dist_spectra(
    crc="1010100111001",
    hamming_dist=np.arange(127),
    num_cwds=np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 2331, 0, 0, 0, 84987, 0, 0, 0, 2897631, 0, 0, 0, 119119581, 0, 
    0, 0, 4734304659, 0, 0, 0, 126908737578, 0, 0, 0, 1984581957582, 0, 0, 0, 
    17869784685648, 0, 0, 0, 94150244799198, 0, 0, 0, 294876701919162, 0, 0, 
    0, 554509101615714, 0, 0, 0, 629029279271043, 0, 0, 0, 430819271465814, 0, 
    0, 0, 177678255731368, 0, 0, 0, 43810806512592, 0, 0, 0, 6375119541840, 0, 
    0, 0, 537991275303, 0, 0, 0, 26117180019, 0, 0, 0, 773839143, 0, 0, 0, 18247551, 
    0, 0, 0, 480375, 0, 0, 0, 15750, 0, 0, 0, 378, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0]),
    dmin=20
)
dsub_k51n126v5CyclicBest = dsu(k51n126v5CyclicBest_dist_spectra, esno_linear)

# fmt: on


# SP59
k51n126_sp59 = np.loadtxt("data/k51n126_sp59.csv", delimiter=",")

# RCU
k51n126_rcu = np.loadtxt("data/k51n126_rcu.csv", delimiter=",")

fig, ax = plt.subplots(figsize=(7, 5))
ax.set_yscale("log")

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

(dsubv6,) = plt.semilogy(
    ebno_dB,
    dsub_k51n126v6BCHBestCC,
    marker="s",
    markerfacecolor="none",
    linewidth=1.5,
    markevery=2,
    label=r"Best stand-alone CRC and $\nu=6$ CC,  $A_{14}=63$",
)

(dsubv7,) = plt.semilogy(
    ebno_dB,
    dsub_k51n126v7BCHBestCC,
    marker="o",
    markerfacecolor="none",
    linewidth=1.5,
    markevery=2,
    label=r"Best stand-alone CRC and $\nu=7$ CC,  $A_{15}=63$",
)

(dsubv8,) = plt.semilogy(
    ebno_dB,
    dsub_k51n126v8ELF1539,
    marker="^",
    markerfacecolor="none",
    linewidth=1.5,
    markevery=2,
    label=r"Best stand-alone CRC and $\nu=8$ CC, $A_{22}=1071$",
)

(dsubv5CyclicBest,) = plt.semilogy(
    ebno_dB,
    dsub_k51n126v5CyclicBest,
    marker="*",
    markerfacecolor="none",
    linewidth=1.5,
    markevery=2,
    label=r"Joint Optimization of CRC and $\nu=5$ TBCC, $A_{20}=2331$",
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
        dsubv7,
        dsubv5CyclicBest,
        dsubv8,
        sp59,
    ],
    loc="upper right",
    fontsize=12,
    framealpha=0.5,
)
ax.add_artist(legend)

plt.title(rf"$({N}, {K})$ ELF-TBCC")
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.xlim([2, 7])
plt.ylim([1e-9, 1e-2])
plt.xlabel(r"$\frac{E_b}{N_o} (\mathrm{dB})$", fontsize=15)
plt.ylabel(r"Probability of codeword error, $P_{cw}$", fontsize=15)
plt.tight_layout()
plt.show()
