from pathlib import Path

from bounds import dsu
from reference_spectra import load_reference_spectra

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


SPECTRA = load_reference_spectra(
    Path(__file__).with_name("spectra") / "k21n62_reference_spectra.yaml"
)
BobReference = SPECTRA["bob_reference"]
BestNu4 = SPECTRA["gridsearch_best_nu4"]
BestNu3 = SPECTRA["gridsearch_best_nu3"]
ComponentNu4 = SPECTRA["component_nu4_ge3203_g27_31"]
RateOneThirdNu6 = SPECTRA["rate_one_third_nu6"]

# The new standalone TBCC is rate 1/3; the existing ELF-TBCC curves are rate 1/2.
elf_tbcc_rate = BestNu4.K / BestNu4.N
rate_one_third_tbcc_rate = RateOneThirdNu6.K / RateOneThirdNu6.N

ebno_dB = np.arange(2, 7.1, 0.1)
ebno_linear = 10 ** (0.1 * ebno_dB)
esno_linear = ebno_linear * elf_tbcc_rate
rate_one_third_esno_linear = ebno_linear * rate_one_third_tbcc_rate

dsub_k21n62v4_best = dsu(BestNu4, esno_linear)
dsub_best_nu3 = dsu(BestNu3, esno_linear)
dsub_component_nu4 = dsu(ComponentNu4, esno_linear)
dsub_rate_one_third_nu6 = dsu(RateOneThirdNu6, rate_one_third_esno_linear)

k21n62_sp59 = np.loadtxt("data/k21n62_sp59.csv", delimiter=",")
k21n62_rcu = np.loadtxt("data/k21n62_rcu.csv", delimiter=",")

fig, ax = plt.subplots(figsize=(7, 5))
ax.set_yscale("log")

(sp59,) = plt.semilogy(
    k21n62_sp59[:, 1], k21n62_sp59[:, 3], linewidth=1.5, label="Sphere Packing Bound"
)
(rcu,) = plt.semilogy(
    k21n62_rcu[:, 1], k21n62_rcu[:, 3], linewidth=1.5,
    linestyle="--", label="Random Coding Union Bound"
)
(dsub_component_nu4_curve,) = plt.semilogy(
    ebno_dB, dsub_component_nu4, marker="d", markerfacecolor="none",
    linewidth=1.5, markevery=5,
    label=ComponentNu4.label,
)
(dsub_best_nu3_curve,) = plt.semilogy(
    ebno_dB, dsub_best_nu3, marker="^", markerfacecolor="none",
    linewidth=1.5, markevery=5,
    label=r"$m=10, g_e=2655, \nu=3, G=[13,17], A_{15}=31$",
)
(dsub_best_nu4,) = plt.semilogy(
    ebno_dB, dsub_k21n62v4_best, marker="o", markerfacecolor="none",
    linewidth=1.5, markevery=5,
    label=r"$m=10, g_e=3013, \nu=4, G=[23,35], A_{16}=93$",
)
(dsub_rate_one_third_nu6_curve,) = plt.semilogy(
    ebno_dB, dsub_rate_one_third_nu6, marker="s", markerfacecolor="none",
    linewidth=1.5, markevery=5, label=RateOneThirdNu6.label,
)

plt.fill_between(
    k21n62_rcu[:, 1], k21n62_sp59[:, 3], k21n62_rcu[:, 3],
    color="skyblue", alpha=0.4, label="Area Between",
)

legend = ax.legend(
    handles=[rcu, dsub_rate_one_third_nu6_curve, dsub_best_nu3_curve, dsub_component_nu4_curve, dsub_best_nu4, sp59],
    loc="upper right", fontsize=12, framealpha=0.5,
)
ax.add_artist(legend)

plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.xlim([2, 7])
plt.ylim([1e-9, 1e-2])
plt.xlabel(r"$\frac{E_b}{N_o} (\mathrm{dB})$", fontsize=15)
plt.ylabel(r"Probability of codeword error, $P_{cw}$", fontsize=15)
plt.tight_layout()
output_path = Path(__file__).with_name("figures") / "k21n62_dsu_bounds.pdf"
output_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(output_path, bbox_inches="tight")
plt.show()
