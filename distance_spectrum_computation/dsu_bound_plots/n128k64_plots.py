"""Plot the punctured (128, 64) code's DSU and RCU approximation."""

import argparse
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LogLocator

from bounds import dsu


HERE = Path(__file__).resolve().parent
DEFAULT_SPECTRUM = (
    HERE.parent / "output" / "k64n128v8_g561_753_p547bcf8a47ba_dist_spectrum.npy"
)
RCU_CSV = HERE / "data" / "k64n128_rcu.csv"
EBNO_DB = np.linspace(1.0, 6.0, 51)
K = 64
N = 128


def load_spectrum(path):
    counts = np.load(path, allow_pickle=False)
    if counts.shape != (N + 1,) or not np.issubdtype(counts.dtype, np.integer):
        raise ValueError("spectrum must contain 129 integer weight counts")
    if np.any(counts < 0) or int(counts[0]) != 1:
        raise ValueError("spectrum must be nonnegative with A_0 = 1")
    if sum(int(count) for count in counts) != 2**K:
        raise ValueError("spectrum counts must total 2**64")
    return counts


def load_rcu(path):
    data = np.loadtxt(path, delimiter=",")
    if data.shape != (len(EBNO_DB), 4):
        raise ValueError("RCU CSV must contain 51 rows and four columns")
    ebno_db, probability = data[:, 1], data[:, 3]
    if not np.allclose(ebno_db, EBNO_DB, atol=1e-10, rtol=0):
        raise ValueError("RCU Eb/N0 values must run from 1 to 6 dB in 0.1 dB steps")
    if not np.all(np.isfinite(probability)) or not np.all(probability > 0):
        raise ValueError("RCU probabilities must be finite and positive")
    if not np.all(np.diff(probability) < 0):
        raise ValueError("RCU probabilities must decrease with Eb/N0")
    return probability


def crossing_db(probability, target=1e-6):
    """Interpolate Eb/N0 against log10(error probability)."""
    if not probability[-1] < target < probability[0]:
        raise ValueError("target probability is outside the computed curve")
    return float(np.interp(np.log10(target), np.log10(probability[::-1]), EBNO_DB[::-1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spectrum", type=Path, default=DEFAULT_SPECTRUM)
    args = parser.parse_args()

    counts = load_spectrum(args.spectrum)
    rcu_probability = load_rcu(RCU_CSV)
    spectrum = SimpleNamespace(hamming_dist=np.arange(N + 1), num_cwds=counts)
    esno_linear = (K / N) * 10 ** (EBNO_DB / 10)
    dsu_probability = np.minimum(1.0, dsu(spectrum, esno_linear))

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.semilogy(
        EBNO_DB, dsu_probability, marker="o", markerfacecolor="none",
        markersize=4, linewidth=1.5, label="DSU (punctured spectrum)"
    )
    ax.semilogy(
        EBNO_DB, rcu_probability, linestyle="--", label="RCU saddlepoint approximation"
    )
    ax.set(xlim=(1, 6), ylim=(1e-10, 1e-1))
    ax.set_yticks(10.0 ** np.arange(-10, 0))
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs=range(2, 10), numticks=100))
    ax.set_xlabel(r"$E_b/N_0$ (dB)")
    ax.set_ylabel("Codeword error probability")
    ax.set_title("Punctured (128, 64) ELF-TBCC")
    ax.grid(True, which="major", linestyle="--", linewidth=0.5, alpha=0.5)
    ax.grid(True, which="minor", axis="y", linestyle="--", linewidth=0.4, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    figures = HERE / "figures"
    figures.mkdir(exist_ok=True)
    for suffix in ("pdf", "png"):
        path = figures / f"k64n128_dsu_rcu.{suffix}"
        fig.savefig(path, dpi=300)
        print(f"Saved {path}")
    plt.close(fig)

    dsu_db = crossing_db(dsu_probability)
    rcu_db = crossing_db(rcu_probability)
    print(
        f"At error probability 1e-6: DSU {dsu_db:.4f} dB, "
        f"RCU {rcu_db:.4f} dB, gap {dsu_db - rcu_db:.4f} dB"
    )


if __name__ == "__main__":
    main()
