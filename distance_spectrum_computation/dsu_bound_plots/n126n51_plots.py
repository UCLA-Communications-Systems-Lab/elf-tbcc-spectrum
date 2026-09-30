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
    Path(__file__).with_name("spectra") / "k51n126_reference_spectra.yaml"
)
StandaloneNu6 = SPECTRA["standalone_nu6"]
StandaloneNu7 = SPECTRA["standalone_nu7"]
StandaloneNu8Elf1539 = SPECTRA["standalone_nu8_elf1539"]
JointNu5Cyclic = SPECTRA["joint_nu5_cyclic"]

N = StandaloneNu6.N
K = StandaloneNu6.K
rates = {spectrum.K / spectrum.N for spectrum in SPECTRA.values()}
if len(rates) != 1:
    raise ValueError("n126n51 plot compares spectra with different rates")
R = rates.pop()

ebno_dB = np.arange(2, 5.6, 0.25)
ebno_linear = 10 ** (0.1 * ebno_dB)
esno_linear = ebno_linear * R

dsub_k51n126v6 = dsu(StandaloneNu6, esno_linear)
dsub_k51n126v7 = dsu(StandaloneNu7, esno_linear)
dsub_k51n126v8_elf1539 = dsu(StandaloneNu8Elf1539, esno_linear)
dsub_k51n126v5_cyclic = dsu(JointNu5Cyclic, esno_linear)

k51n126_sp59 = np.loadtxt("data/k51n126_sp59.csv", delimiter=",")
k51n126_rcu = np.loadtxt("data/k51n126_rcu.csv", delimiter=",")

fig, ax = plt.subplots(figsize=(7, 5))
ax.set_yscale("log")

(sp59,) = plt.semilogy(
    k51n126_sp59[:, 1], k51n126_sp59[:, 3], linewidth=1.5, label="Sphere Packing Bound"
)
(rcu,) = plt.semilogy(
    k51n126_rcu[:, 1], k51n126_rcu[:, 3], linewidth=1.5,
    linestyle="--", label="Random Coding Union Bound"
)
(dsubv6,) = plt.semilogy(
    ebno_dB, dsub_k51n126v6, marker="s", markerfacecolor="none",
    linewidth=1.5, markevery=2, label=StandaloneNu6.label,
)
(dsubv7,) = plt.semilogy(
    ebno_dB, dsub_k51n126v7, marker="o", markerfacecolor="none",
    linewidth=1.5, markevery=2, label=StandaloneNu7.label,
)
(dsubv8,) = plt.semilogy(
    ebno_dB, dsub_k51n126v8_elf1539, marker="^", markerfacecolor="none",
    linewidth=1.5, markevery=2, label=StandaloneNu8Elf1539.label,
)
(dsubv5_cyclic,) = plt.semilogy(
    ebno_dB, dsub_k51n126v5_cyclic, marker="*", markerfacecolor="none",
    linewidth=1.5, markevery=2, label=JointNu5Cyclic.label,
)

plt.fill_between(
    k51n126_rcu[:, 1], k51n126_sp59[:, 3], k51n126_rcu[:, 3],
    color="skyblue", alpha=0.4, label="Area Between",
)

legend = ax.legend(
    handles=[rcu, dsubv6, dsubv7, dsubv5_cyclic, dsubv8, sp59],
    loc="upper right", fontsize=12, framealpha=0.5,
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
