"""Load and validate curated distance spectra used by DSU plot scripts."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml


@dataclass(frozen=True)
class DistanceSpectrum:
    """A verified spectrum and the metadata needed to plot its DSU bound."""

    hamming_dist: np.ndarray
    num_cwds: np.ndarray
    dmin: int
    N: int
    K: int
    label: str
    crc: str | None = None
    plot: bool = True


def load_reference_spectra(path: Path) -> dict[str, DistanceSpectrum]:
    """Load YAML spectra and require complete enumerators for plotted entries."""
    with path.open() as handle:
        entries = yaml.safe_load(handle)["spectra"]

    spectra = {}
    for name, entry in entries.items():
        hamming_dist = np.asarray(entry["hamming_dist"], dtype=int)
        num_cwds = np.asarray(entry["num_cwds"], dtype=np.int64)
        N = int(entry["N"])
        K = int(entry["K"])
        plot = bool(entry.get("plot", True))

        if len(hamming_dist) != len(num_cwds):
            raise ValueError(f"{name}: hamming_dist and num_cwds have different lengths")
        if np.any(hamming_dist < 0) or np.any(hamming_dist > N):
            raise ValueError(f"{name}: Hamming distances must be within 0..N")
        if np.any(num_cwds < 0):
            raise ValueError(f"{name}: spectrum counts must be nonnegative")
        if plot and int(num_cwds.sum()) != 2**K:
            raise ValueError(
                f"{name}: plotted spectrum count must equal 2**K; mark legacy data plot: false"
            )
        positive_weights = hamming_dist[(hamming_dist > 0) & (num_cwds > 0)]
        if not len(positive_weights):
            raise ValueError(f"{name}: spectrum has no nonzero codeword")

        spectra[name] = DistanceSpectrum(
            hamming_dist=hamming_dist,
            num_cwds=num_cwds,
            dmin=int(positive_weights.min()),
            N=N,
            K=K,
            label=entry.get("label", name),
            crc=entry.get("crc"),
            plot=plot,
        )
    return spectra
