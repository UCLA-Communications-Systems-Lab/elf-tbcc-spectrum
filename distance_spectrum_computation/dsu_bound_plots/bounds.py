import numpy as np
from scipy import special, integrate


def qfunc(x):
    return 0.5 - 0.5 * special.erf(x / np.sqrt(2.0, dtype=np.float64))


def dsu(dist_spectra, esno_linear):
    """Complete Union Bound based on code spectrum"""
    if len(dist_spectra.hamming_dist) != len(dist_spectra.num_cwds):
        print("lengths not compatible, abort!")
        return
    Es_over_sigma_squared = esno_linear * 2.0
    dsub = np.zeros_like(esno_linear, dtype=np.float64)
    for i_w in range(len(dist_spectra.hamming_dist)):
        Aw = dist_spectra.num_cwds[i_w]
        w = dist_spectra.hamming_dist[i_w]
        if w == 0:
            # skip the all-zero codeword
            continue
        dsub += Aw * qfunc(np.sqrt(w * Es_over_sigma_squared))

    return dsub


def bi_awgn_capacity_stable(rho):
    def capacity_integrand(x, rho):
        # Calculate the exponent
        exponent = -2 * rho + 2 * np.sqrt(rho) * x

        # Stability check: log2(1 + exp(u))
        # If u is very large, log2(1 + exp(u)) is approximately log2(exp(u)) = u * log2(e)
        if exponent > 700:
            log_term = exponent * np.log2(np.e)
        else:
            log_term = np.log2(1 + np.exp(exponent))

        gaussian_pdf = (1 / np.sqrt(2 * np.pi)) * np.exp(-(x**2) / 2)
        return gaussian_pdf * (1 - log_term)

    # Use a list comprehension or np.vectorize if rho is an array
    C, err = integrate.quad(capacity_integrand, -10, 10, args=(rho,))
    return C


def bi_awgn_metrics(esno_linear):
    """Calculates Capacity (C) and Dispersion (V) for BI-AWGN."""
    rho = esno_linear * 2  # SNR

    def info_density(x, rho):
        # Using log1p and logic to avoid exp(large_val)
        exponent = -2 * rho + 2 * np.sqrt(rho) * x
        # log2(1 + e^u) stability
        term = np.piecewise(
            exponent,
            [exponent > 700, exponent <= 700],
            [lambda u: u * np.log2(np.e), lambda u: np.log2(1 + np.exp(u))],
        )
        return 1 - term

    # 1. Calculate Capacity C
    c_integrand = (
        lambda x: (1 / np.sqrt(2 * np.pi)) * np.exp(-(x**2) / 2) * info_density(x, rho)
    )
    C, _ = integrate.quad(c_integrand, -10, 10)

    # 2. Calculate Dispersion V
    v_integrand = (
        lambda x: (1 / np.sqrt(2 * np.pi))
        * np.exp(-(x**2) / 2)
        * (info_density(x, rho) - C) ** 2
    )
    V, _ = integrate.quad(v_integrand, -10, 10)

    return C, V


def normal_approx(esno_linear, n, R):
    C, V = bi_awgn_metrics(esno_linear)

    # Standard Polyanskiy Normal Approximation formula
    # P_err \approx Q( (nC - nR + 0.5*log2(n)) / sqrt(nV) )
    numerator = n * C - n * R + 0.5 * np.log2(n)
    denominator = np.sqrt(n * V)

    return qfunc(numerator / denominator)


def pcw_easy(dist_spectra, esno_linear):
    Es_over_sigma_squared = esno_linear * 2
    AofW = 0
    for i_w in range(len(dist_spectra.hamming_dist)):
        Aw = dist_spectra.num_cwds[i_w]
        w = dist_spectra.hamming_dist[i_w]
        if w == 0:
            continue
        AofW += Aw * np.exp(-w * Es_over_sigma_squared / 2)
    pcw = 0.5 * AofW
    return pcw


def pcw_better(dist_spectra, esno_linear):
    Es_over_sigma_squared = esno_linear * 2
    AofW = 0
    for i_w in range(len(dist_spectra.hamming_dist)):
        Aw = dist_spectra.num_cwds[i_w]
        w = dist_spectra.hamming_dist[i_w]
        if w == 0:
            continue
        AofW += Aw * np.exp(-w * Es_over_sigma_squared / 2)
    pcw = np.zeros_like(esno_linear)
    pcw = (
        qfunc(np.sqrt(dist_spectra.dmin * Es_over_sigma_squared))
        * np.exp(dist_spectra.dmin * Es_over_sigma_squared / 2)
        * AofW
    )
    return pcw
