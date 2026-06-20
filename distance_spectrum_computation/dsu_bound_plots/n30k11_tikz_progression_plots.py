from dataclasses import dataclass
from cycler import cycler
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# 1. Fix Font Type and Styling
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
    "#003E67",
    "#A2142F",
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
# fmt: on

N = 30
K = 11
R = K / N
ebno_dB = np.arange(1, 9.1, 0.1)
ebno_linear = 10 ** (0.1 * ebno_dB)
esno_linear = ebno_linear * R

# Dummy DSU calculation engine fallback
try:
    from bounds import dsu
except ImportError:

    def dsu(spectra, esno):
        return 1e-2 * np.exp(-esno / spectra.dmin)


dsub_5g = dsu(ShortCode_5G, esno_linear)
dsub_GridSearchBest = dsu(GridSearchBest, esno_linear)
dsub_HammingAndBestNu4 = dsu(HammingAndBestNu4, esno_linear)
dsub_bestELFforBestNu4 = dsu(BestELFforBestNu4, esno_linear)

# Local CSV imports fallback handler
try:
    k11n30_sp59 = np.loadtxt("data/k11n30_sp59.csv", delimiter=",")
    k11n30_rcu = np.loadtxt("data/k11n30_rcu.csv", delimiter=",")
except OSError:
    mock_ebno = np.linspace(1, 10, 100)
    k11n30_sp59 = np.column_stack(
        [mock_ebno, mock_ebno, mock_ebno, 1e-2 * np.exp(-mock_ebno / 2)]
    )
    k11n30_rcu = np.column_stack(
        [mock_ebno, mock_ebno, mock_ebno, 2e-2 * np.exp(-mock_ebno / 2.2)]
    )


# --- Updated 4-Plot Progression Setup (Without Apple) ---
plot_steps = [
    "1_baselines_etsi",
    "2_add_hamming_tbcc",
    "3_add_optimized_elf",
    "4_add_joint_optimization",
]

for step in plot_steps:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.set_yscale("log")

    lower_handles = []
    upper_handles = []

    # ==========================================
    # --- Lower Right Legend Group (Baselines) ---
    # ==========================================

    (rcu,) = ax.semilogy(
        k11n30_rcu[:, 1],
        k11n30_rcu[:, 3],
        linewidth=1.5,
        linestyle="--",
        color="#D95319",
        label="Random Coding Union Bound",
    )
    (dsu_etsi,) = ax.semilogy(
        ebno_dB,
        dsub_5g,
        linewidth=1.5,
        color="#EDB120",
        marker="d",
        markerfacecolor="none",
        markevery=5,
        label=r"Current ETSI Standard, $A_8=5$",
    )
    (sp59,) = ax.semilogy(
        k11n30_sp59[:, 1],
        k11n30_sp59[:, 3],
        linewidth=1.5,
        color="#0072BD",
        label="Sphere Packing Bound",
    )

    # Underlying shading gap area
    ax.fill_between(
        k11n30_rcu[:, 1],
        k11n30_sp59[:, 3],
        k11n30_rcu[:, 3],
        color="skyblue",
        alpha=0.4,
    )

    lower_handles = [rcu, dsu_etsi, sp59]

    # ==========================================
    # --- Upper Right Legend Group (Progression) ---
    # ==========================================

    # --- Plot 2+: Best stand-alone CRC and TBCC ---
    if step in [
        "2_add_hamming_tbcc",
        "3_add_optimized_elf",
        "4_add_joint_optimization",
    ]:
        (dsu_hamming_bestnu4,) = ax.semilogy(
            ebno_dB,
            dsub_HammingAndBestNu4,
            linewidth=1.5,
            color="#7E2F8E",
            marker="s",
            markerfacecolor="none",
            markevery=5,
            label=r"Best stand-alone CRC and TBCC, $A_8=15$",
        )
        upper_handles.append(dsu_hamming_bestnu4)

    # --- Plot 3+: Optimized ELF for the BEST TBCC ---
    if step in ["3_add_optimized_elf", "4_add_joint_optimization"]:
        (dsu_best_elf_nu4,) = ax.semilogy(
            ebno_dB,
            dsub_bestELFforBestNu4,
            linewidth=1.5,
            color="#77AC30",
            marker="o",
            markerfacecolor="none",
            markevery=5,
            label=r"Optimized ELF for the best TBCC, $A_9=35$",
        )
        upper_handles.append(dsu_best_elf_nu4)

    # --- Plot 4+: Joint Optimization ---
    if step == "4_add_joint_optimization":
        (dsu_grid_search_best,) = ax.semilogy(
            ebno_dB,
            dsub_GridSearchBest,
            linewidth=1.5,
            color="#003E67",
            marker="*",
            markerfacecolor="none",
            markevery=5,
            label=r"Joint Optimization of ELF and TBCC, $A_{10}=138$ ",
        )
        upper_handles.append(dsu_grid_search_best)

    # --- Canvas Cosmetics & Clean Title ---
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    ax.set_xlim([4, 9])
    ax.set_ylim([1e-9, 1e-2])
    ax.set_xlabel(r"$\frac{E_b}{N_o} (\mathrm{dB})$", fontsize=15)
    ax.set_ylabel(r"Probability of codeword error, $P_{cw}$", fontsize=15)

    # Cleaned Title
    ax.set_title(
        rf"$({N}, {K})$ $\nu=4, m=4$ ELF-TBCC",
        fontsize=13,
        pad=10,
    )

    # --- Add Dual Legends Instantiations ---
    # Legend 1: Lower Right Bound Set
    leg_lower = ax.legend(
        handles=lower_handles,
        labels=[h.get_label() for h in lower_handles],
        loc="lower left",
        fontsize=11,
        framealpha=0.8,
    )
    ax.add_artist(leg_lower)  # Lock lower layer baseline legend placement

    # Legend 2: Upper Right Progression Features
    if upper_handles:
        ax.legend(
            handles=upper_handles,
            labels=[h.get_label() for h in upper_handles],
            loc="upper right",
            fontsize=11,
            framealpha=0.8,
        )

    plt.tight_layout()
    plt.show()
