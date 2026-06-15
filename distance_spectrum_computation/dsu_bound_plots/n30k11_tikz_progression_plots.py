from dataclasses import dataclass
from cycler import cycler
from bounds import dsu
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# 1. Fix Font Type and Styling (Matches your setup)
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "Liberation Sans"]

gem12_colors = [
    "#0072BD",
    "#D95319",
    "#EDB120",
    "#7E2F8E",
    "#77AC30",
    "#4DBEEE",
    "#A2142F",
    "#003E67",
    "#722C0D",
    "#7C5D10",
    "#42194B",
    "#3E5A19",
]
plt.rcParams["axes.prop_cycle"] = cycler(color=gem12_colors)


@dataclass
class dist_spectra:
    crc: str
    hamming_dist: np.array
    num_cwds: np.array
    dmin: int


# --- Data Configurations ---
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

HammingAndBestNu4 = dist_spectra(
    crc="10011",
    hamming_dist=np.arange(31),
    num_cwds=np.array([1, 0, 0, 0, 0, 0, 0, 0, 15, 50, 60, 75, 95, 240, 390, 249, 240, 330, 160, 45, 33, 20, 30, 15, 0, 0, 0, 0, 0, 0, 0]),
    dmin=8
)

BestELFforBestNu4 = dist_spectra(
    crc="11001",
    hamming_dist=np.arange(31),
    num_cwds=np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 35, 75, 90, 140, 285, 345, 204, 195, 285, 205, 90, 48, 35, 15, 0, 0, 0, 0, 0, 0, 0, 0]),
    dmin=9
)

GridSearchBest = dist_spectra(
    crc="11111",
    hamming_dist=np.arange(31),
    num_cwds=np.array([1,0,0,0,0,0,0,0,0,0,138,0,390,0,495,0,495,0,390,0,138,0,0,0,0,0,0,0,0,0,1]),
    dmin=10
)

hamming_distance = np.array([0, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 30])

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
ebno_dB = np.arange(1, 9.1, 0.1)
ebno_linear = 10 ** (0.1 * ebno_dB)
esno_linear = ebno_linear * R

# Real DSU Calculations using your local 'bounds' module
dsub_10011 = dsu(CRC_10011, esno_linear)
dsub_10101 = dsu(ELF_10101, esno_linear)
dsub_5g = dsu(ShortCode_5G, esno_linear)
dsub_apple = dsu(AppleProp_6G, esno_linear)
dsub_GridSearchBest = dsu(GridSearchBest, esno_linear)
dsub_HammingAndBestNu4 = dsu(HammingAndBestNu4, esno_linear)
dsub_bestELFforBestNu4 = dsu(BestELFforBestNu4, esno_linear)

# Local CSV Data File Reads
k11n30_sp59 = np.loadtxt("data/k11n30_sp59.csv", delimiter=",")
k11n30_rcu = np.loadtxt("data/k11n30_rcu.csv", delimiter=",")


# --- Plot Progression Setup ---
# Added "7_grid_search_best" to generate Plot 7 in the display loop
steps = [
    "0_empty",
    "1_sp_bound",
    "2_rcu_and_fill",
    "3_dsu_curves",
    "4_final_complete",
    "5_best_elf_nu4",
    "6_apple_prop",
    "7_grid_search_best",
]

for step in steps:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.set_yscale("log")

    handles = []

    # Step 1 onwards: Sphere Packing Bound from local CSV data
    if step != "0_empty":
        (sp59,) = ax.semilogy(
            k11n30_sp59[:, 1],
            k11n30_sp59[:, 3],
            linewidth=1.5,
            label="Sphere Packing Bound",
        )
        handles.append(sp59)

    # Step 2 onwards: RCU Bound and the shaded blue region between them
    if step in [
        "2_rcu_and_fill",
        "3_dsu_curves",
        "4_final_complete",
        "5_best_elf_nu4",
        "6_apple_prop",
        "7_grid_search_best",
    ]:
        (rcu,) = ax.semilogy(
            k11n30_rcu[:, 1],
            k11n30_rcu[:, 3],
            linewidth=1.5,
            linestyle="--",
            label="Random Coding Union Bound",
        )
        ax.fill_between(
            k11n30_rcu[:, 1],
            k11n30_sp59[:, 3],
            k11n30_rcu[:, 3],
            color="skyblue",
            alpha=0.4,
            label="Area Between",
        )
        handles.insert(0, rcu)

    # Step 3 ONLY: Show exclusively the 5G ETSI curve (Explicit Yellow)
    if step == "3_dsu_curves":
        (dsu_etsi,) = ax.semilogy(
            ebno_dB,
            dsub_5g,
            linewidth=1.5,
            color="#EDB120",
            marker="d",
            markerfacecolor="none",
            markevery=5,
            label=r"5G ETSI, $A_{8}=5$",
        )
        handles.append(dsu_etsi)

    # Step 4 ONLY: Show exclusively 5G ETSI and the HammingAndBestNu4 curve along with baselines
    if step == "4_final_complete":
        (dsu_hamming_bestnu4,) = ax.semilogy(
            ebno_dB,
            dsub_HammingAndBestNu4,
            linewidth=1.5,
            color="#7E2F8E",
            marker="s",
            markerfacecolor="none",
            markevery=5,
            label=r"$g_e=23, (g_1, g_2) = (27,31)$, $A_{8}=15$",
        )
        (dsu_etsi,) = ax.semilogy(
            ebno_dB,
            dsub_5g,
            linewidth=1.5,
            color="#EDB120",
            marker="d",
            markerfacecolor="none",
            markevery=5,
            label=r"5G ETSI, $A_{8}=5$",
        )
        handles.extend([dsu_hamming_bestnu4, dsu_etsi])

    # Step 5 ONLY: Show 5G ETSI, HammingAndBestNu4, AND the BestELFforBestNu4 curve (Explicit Green)
    if step == "5_best_elf_nu4":
        (dsu_best_elf_nu4,) = ax.semilogy(
            ebno_dB,
            dsub_bestELFforBestNu4,
            linewidth=1.5,
            color="#77AC30",
            marker="o",
            markerfacecolor="none",
            markevery=5,
            label=r"$g_e=31, (g_1, g_2) = (27,31)$, $A_{9}=35$",
        )
        (dsu_hamming_bestnu4,) = ax.semilogy(
            ebno_dB,
            dsub_HammingAndBestNu4,
            linewidth=1.5,
            color="#7E2F8E",
            marker="s",
            markerfacecolor="none",
            markevery=5,
            label=r"$g_e=23, (g_1, g_2) = (27,31)$, $A_{8}=15$",
        )
        (dsu_etsi,) = ax.semilogy(
            ebno_dB,
            dsub_5g,
            linewidth=1.5,
            color="#EDB120",
            marker="d",
            markerfacecolor="none",
            markevery=5,
            label=r"5G ETSI, $A_{8}=5$",
        )
        handles.extend([dsu_best_elf_nu4, dsu_hamming_bestnu4, dsu_etsi])

    # Step 6 ONLY: Show all step 5 curves plus the Apple Proposal curve (Explicit Light Blue)
    if step == "6_apple_prop":
        (dsu_apple,) = ax.semilogy(
            ebno_dB,
            dsub_apple,
            linewidth=1.5,
            color="#4DBEEE",
            marker="p",
            markerfacecolor="none",
            markevery=5,
            label=r"Apple proposal, $A_{10}=66$",
        )
        (dsu_best_elf_nu4,) = ax.semilogy(
            ebno_dB,
            dsub_bestELFforBestNu4,
            linewidth=1.5,
            color="#77AC30",
            marker="o",
            markerfacecolor="none",
            markevery=5,
            label=r"$g_e=31, (g_1, g_2) = (27,31)$, $A_{9}=35$",
        )
        (dsu_hamming_bestnu4,) = ax.semilogy(
            ebno_dB,
            dsub_HammingAndBestNu4,
            linewidth=1.5,
            color="#7E2F8E",
            marker="s",
            markerfacecolor="none",
            markevery=5,
            label=r"$g_e=23, (g_1, g_2) = (27,31)$, $A_{8}=15$",
        )
        (dsu_etsi,) = ax.semilogy(
            ebno_dB,
            dsub_5g,
            linewidth=1.5,
            color="#EDB120",
            marker="d",
            markerfacecolor="none",
            markevery=5,
            label=r"5G ETSI, $A_{8}=5$",
        )
        handles.extend([dsu_apple, dsu_best_elf_nu4, dsu_hamming_bestnu4, dsu_etsi])

    # Step 7 ONLY: Show all step 6 curves plus the GridSearchBest curve (Explicit Deep Blue)
    if step == "7_grid_search_best":
        (dsu_grid_search_best,) = ax.semilogy(
            ebno_dB,
            dsub_GridSearchBest,
            linewidth=1.5,
            color="#003E67",
            marker="*",
            markerfacecolor="none",
            markevery=5,
            label=r"$g_e=23, (g_1, g_2) = (23,25)$, $A_{10}=138$",
        )
        (dsu_apple,) = ax.semilogy(
            ebno_dB,
            dsub_apple,
            linewidth=1.5,
            color="#4DBEEE",
            marker="p",
            markerfacecolor="none",
            markevery=5,
            label=r"Apple proposal, $A_{10}=66$",
        )
        (dsu_best_elf_nu4,) = ax.semilogy(
            ebno_dB,
            dsub_bestELFforBestNu4,
            linewidth=1.5,
            color="#77AC30",
            marker="o",
            markerfacecolor="none",
            markevery=5,
            label=r"$g_e=31, (g_1, g_2) = (27,31)$, $A_{9}=35$",
        )
        (dsu_hamming_bestnu4,) = ax.semilogy(
            ebno_dB,
            dsub_HammingAndBestNu4,
            linewidth=1.5,
            color="#7E2F8E",
            marker="s",
            markerfacecolor="none",
            markevery=5,
            label=r"$g_e=23, (g_1, g_2) = (27,31)$, $A_{8}=15$",
        )
        (dsu_etsi,) = ax.semilogy(
            ebno_dB,
            dsub_5g,
            linewidth=1.5,
            color="#EDB120",
            marker="d",
            markerfacecolor="none",
            markevery=5,
            label=r"5G ETSI, $A_{8}=5$",
        )
        handles.extend(
            [
                dsu_grid_search_best,
                dsu_apple,
                dsu_best_elf_nu4,
                dsu_hamming_bestnu4,
                dsu_etsi,
            ]
        )

    # Canvas Cosmetics, Titles & Constraints
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    ax.set_xlim([4, 9])
    ax.set_ylim([1e-9, 1e-2])
    ax.set_xlabel(r"$\frac{E_b}{N_o} (\mathrm{dB})$", fontsize=15)
    ax.set_ylabel(r"Probability of codeword error, $P_{cw}$", fontsize=15)

    ax.set_title(
        rf"Probability of codeword error vs. Eb/No for $({N}, {K})$ ELF-TBCC",
        fontsize=13,
        pad=10,
    )

    if handles:
        ax.legend(handles=handles, loc="upper right", fontsize=12, framealpha=0.5)

    plt.tight_layout()
    plt.show()
