from pathlib import Path
from bounds import dsu
from reference_spectra import load_reference_spectra

import numpy as np
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


SPECTRA = load_reference_spectra(
    Path(__file__).with_name("spectra") / "k11n30_reference_spectra.yaml"
)
ShortCode_5G = SPECTRA["short_code_5g"]
AppleProp_6G = SPECTRA["apple_proposal_6g"]
ELF_37_G13_7 = SPECTRA["elf37_g13_7"]
ELF_31_G27_31 = SPECTRA["elf31_g27_31"]
NO_ELF_NU5_R13 = SPECTRA["no_elf_nu5_r13"]

Aw = ELF_37_G13_7.num_cwds
w = ELF_37_G13_7.hamming_dist
print("Aw:", Aw)
print("w:", w)

# The no-ELF reference has K=10; each DSU curve uses its own rate to map
# Eb/N0 to Es/N0 while sharing the common Eb/N0 horizontal axis.
R = ShortCode_5G.K / ShortCode_5G.N
no_elf_nu5_r13_rate = NO_ELF_NU5_R13.K / NO_ELF_NU5_R13.N
# EbNo
ebno_dB = np.arange(1, 9.1, 0.1)
ebno_linear = 10 ** (0.1 * ebno_dB)
# EsNo
esno_linear = ebno_linear * R
esno_dB = 10 * np.log10(esno_linear)
no_elf_nu5_r13_esno_linear = ebno_linear * no_elf_nu5_r13_rate
# Es/sigma^2
es_over_sigma_sqrd_linear = esno_linear * 2
es_over_sigma_sqrd_dB = 10 * np.log10(es_over_sigma_sqrd_linear)

## - bounds
dsub_elf31_g27_31 = dsu(ELF_31_G27_31, esno_linear)
dsub_5g = dsu(ShortCode_5G, esno_linear)
dsub_apple = dsu(AppleProp_6G, esno_linear)
dsub_elf37_g13_7 = dsu(ELF_37_G13_7, esno_linear)
dsub_no_elf_nu5_r13 = dsu(NO_ELF_NU5_R13, no_elf_nu5_r13_esno_linear)

# SP59
k11n30_sp59 = np.loadtxt("data/k11n30_sp59.csv", delimiter=",")

# RCU
k11n30_rcu = np.loadtxt("data/k11n30_rcu.csv", delimiter=",")

fig, ax = plt.subplots(figsize=(7, 5))
ax.set_yscale("log")

(dsu_elf31_g27_31,) = plt.semilogy(
    ebno_dB,
    dsub_elf31_g27_31,
    linewidth=1.5,
    marker="o",
    markerfacecolor="none",
    markevery=5,
    label=ELF_31_G27_31.label,
)

(dsu_etsi,) = plt.semilogy(
    ebno_dB,
    dsub_5g,
    linewidth=1.5,
    marker="d",
    markerfacecolor="none",
    markevery=5,
    label=ShortCode_5G.label,
)
(dsu_elf37_g13_7,) = plt.semilogy(
    ebno_dB,
    dsub_elf37_g13_7,
    linewidth=1.5,
    color="#003E67",
    marker="*",
    markerfacecolor="none",
    markevery=5,
    label=ELF_37_G13_7.label,
)
(dsu_apple,) = plt.semilogy(
    ebno_dB,
    dsub_apple,
    linewidth=1.5,
    marker="p",
    markerfacecolor="none",
    markevery=5,
    label=AppleProp_6G.label,
)
(dsu_no_elf_nu5_r13,) = plt.semilogy(
    ebno_dB,
    dsub_no_elf_nu5_r13,
    linewidth=1.5,
    marker="s",
    markerfacecolor="none",
    markevery=5,
    label=NO_ELF_NU5_R13.label,
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
        dsu_etsi,
        dsu_elf31_g27_31,
        dsu_no_elf_nu5_r13,
        dsu_elf37_g13_7,
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
plt.title(r"DSU Bounds for $(30,11)$ ELF-TBCC Codes", fontsize=15)
plt.xlabel(r"$\frac{E_b}{N_o} (\mathrm{dB})$", fontsize=15)
plt.ylabel(r"Probability of codeword error, $P_{cw}$", fontsize=15)
plt.tight_layout()
output_path = Path(__file__).with_name("figures") / "n30k11_dsu.pdf"
output_path.parent.mkdir(exist_ok=True)
fig.savefig(output_path, bbox_inches="tight")
print(f"Saved figure: {output_path}")
plt.close(fig)
