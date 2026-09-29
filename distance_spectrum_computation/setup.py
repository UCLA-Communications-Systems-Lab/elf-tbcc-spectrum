import argparse
import hashlib
import sys

import numpy as np
import yaml


def octal_to_binary_list(octal_str):
    # Convert octal string to integer, then to a binary string
    # '0b' prefix is sliced off using [2:]
    binary_str = bin(int(octal_str, 8))[2:]

    # Convert the string '101' into a list of integers [1, 0, 1]
    return [int(bit) for bit in binary_str]


def bin2dec(binary):
    """
    Note: bin_list is intentionally not reversed to read the number as 'right-msb'.
    It can be "reversed" if the next states is intended to be read as 'left-msb'.
    """
    return [
        int(sum(val * (2**idx) for idx, val in enumerate(bin_list)))
        for bin_list in binary
    ]


def normalize_puncture_pattern(pattern, num_generators, num_stages):
    """Return generator-major mask, phase widths, and transmitted length."""
    if pattern is None:
        mask = np.ones((num_generators, 1), dtype=np.uint8)
    else:
        if not isinstance(pattern, list) or len(pattern) != num_generators:
            raise ValueError("puncture_pattern must have one row per generator polynomial")
        if not all(isinstance(row, list) for row in pattern):
            raise ValueError("puncture_pattern rows must be lists")
        period = len(pattern[0])
        if not 1 <= period <= num_stages or any(len(row) != period for row in pattern):
            raise ValueError("puncture_pattern rows must have equal length in 1..number of stages")
        if any(type(bit) is not int or bit not in (0, 1) for row in pattern for bit in row):
            raise ValueError("puncture_pattern entries must be integers 0 or 1")
        mask = np.ascontiguousarray(pattern, dtype=np.uint8)

    widths = np.sum(mask, axis=0, dtype=np.uint8)
    transmitted_length = sum(int(widths[stage % mask.shape[1]]) for stage in range(num_stages))
    if transmitted_length == 0:
        raise ValueError("puncture_pattern must retain at least one bit")
    return mask, widths, transmitted_length


def puncture_fingerprint(mask):
    """Stable short identifier that includes the mask's shape and bit order."""
    shape = f"{mask.shape[0]}x{mask.shape[1]}:".encode("ascii")
    return hashlib.sha256(shape + mask.tobytes()).hexdigest()[:12]


def spectrum_filename(code_config):
    """Keep legacy names for mother codes; identify punctured outputs."""
    bch = code_config["bch_config"]
    tbcc = code_config["tbcc_config"]
    base = f"k{bch['K']}n{tbcc['N']}v{tbcc['V']}"
    mask, _, _ = normalize_puncture_pattern(
        tbcc.get("puncture_pattern"), len(tbcc["gen_polys"]), bch["K"] + bch["M"]
    )
    if np.any(mask == 0):
        generators = "_".join(tbcc["gen_polys"])
        base += f"_g{generators}_p{puncture_fingerprint(mask)}"
    return f"{base}_dist_spectrum.npy"


def validate_code_config(code_config):
    """Validate the binary rate-1/n TBCC configuration used by this pipeline."""
    try:
        bch = code_config["bch_config"]
        tbcc = code_config["tbcc_config"]
        gen_polys = tbcc["gen_polys"]
    except KeyError as exc:
        raise ValueError(
            f"Missing required configuration field: {exc.args[0]}"
        ) from exc

    if not isinstance(gen_polys, list) or len(gen_polys) not in (2, 3):
        raise ValueError(
            "tbcc_config.gen_polys must contain exactly two or three octal polynomials"
        )
    if not all(
        isinstance(poly, str) and poly and set(poly) <= set("01234567")
        for poly in gen_polys
    ):
        raise ValueError("tbcc_config.gen_polys must be non-empty octal strings")
    if tbcc["K"] != bch["N"]:
        raise ValueError("tbcc_config.K must equal bch_config.N")
    num_stages = bch["K"] + bch["M"]
    if bch["N"] != num_stages:
        raise ValueError("bch_config.N must equal bch_config.K + bch_config.M")
    _, _, transmitted_length = normalize_puncture_pattern(
        tbcc.get("puncture_pattern"), len(gen_polys), num_stages
    )
    if tbcc["N"] != transmitted_length:
        raise ValueError("tbcc_config.N must equal the transmitted punctured length")

    return len(gen_polys)


def setup_A_Wbit_D(code_config):

    num_output_bits = validate_code_config(code_config)
    nu = code_config["tbcc_config"]["V"]
    m = code_config["bch_config"]["M"]
    num_concat_memory = nu + m
    num_total_states = 2 ** (num_concat_memory)
    states = np.arange(0, num_total_states, dtype=np.int32)
    num_trellis_stages = code_config["bch_config"]["K"] + code_config["bch_config"]["M"]
    mask, widths, _ = normalize_puncture_pattern(
        code_config["tbcc_config"].get("puncture_pattern"),
        num_output_bits,
        num_trellis_stages,
    )
    generator_polys = [
        np.flip(octal_to_binary_list(poly))
        for poly in code_config["tbcc_config"]["gen_polys"]
    ]
    p_crc = np.flip([int(bit) for bit in code_config["bch_config"]["polynomial"]])

    # Convolve each TBCC generator with the ELF polynomial, then pad to a
    # common length before evaluating all output bits in one matrix product.
    combined_polys = [
        np.mod(np.convolve(poly, p_crc, mode="full"), 2) for poly in generator_polys
    ]
    max_len = max(map(len, combined_polys))
    combined_polys = [
        np.pad(poly, (0, max_len - len(poly)), "constant") for poly in combined_polys
    ]
    generator_matrix = np.vstack(combined_polys)

    # states
    # states_matrix[s, i] is bit i of state s (LSB at index 0)
    bits = (states[:, None] >> np.arange(num_concat_memory, dtype=np.uint32)) & 1
    v_zeros = np.zeros(shape=(num_total_states, 1), dtype=bits.dtype)
    v_ones = np.ones(shape=(num_total_states, 1), dtype=bits.dtype)
    input_0 = np.hstack((v_zeros, bits))
    input_1 = np.hstack((v_ones, bits))
    dst_0 = ((states << 1) & (num_total_states - 1)).astype(np.uint32)
    dst_1 = (dst_0 | 1).astype(np.uint32)

    # output
    out0 = np.mod(np.matmul(input_0, generator_matrix.T), 2)
    out1 = np.mod(np.matmul(input_1, generator_matrix.T), 2)
    # W[phase, state, input] is the weight of the bits retained by that
    # column of the generator-major puncture mask.
    W_weight = np.empty((mask.shape[1], num_total_states, 2), dtype=np.uint8)
    for phase in range(mask.shape[1]):
        W_weight[phase, :, 0] = out0 @ mask[:, phase]
        W_weight[phase, :, 1] = out1 @ mask[:, phase]

    # set up A
    # A_in: [num_states] x [max weight up to this meta-stage]
    basis = np.arange(0, num_total_states, 2 ** code_config["bch_config"]["M"])
    As = []
    for valid_starting_state in basis:
        A = np.zeros(shape=(num_total_states, 1), dtype=np.uint64)
        A[valid_starting_state] = 1
        As.append(A)

    # set up D
    # D_in: [num_states] x [input]
    D = np.stack((dst_0, dst_1), axis=1).astype(np.uint32)

    return As, W_weight, D, basis, num_trellis_stages, num_output_bits, widths


def computeMetaStage(W, D, dtype=np.uint64):
    """
    Join num_stages_combine basic stages together into a meta-stage.
    The

    Args:
        - W: [num_inputs, num_states, max_weight with 1 stage]
        - D: [num_states, num_inputs]

    Outs:
        - newW: [num_states, num_states, max_weight with meta-stage]
        - newD: [num_states, 2^num_stages_combine]

    """
    assert W.shape[0] == D.shape[1]
    num_inputs = W.shape[0]
    num_states = W.shape[1]
    curr_max_weight = W.shape[2]

    # Compute newW
    newW = np.zeros(
        shape=(num_states, num_states, 2 * curr_max_weight - 1), dtype=dtype
    )
    newD = np.zeros(shape=(num_states, num_inputs**2), dtype=dtype)

    for i_b in range(num_states):
        for i_first_input in range(num_inputs):
            # compute mid state after first meta-stage transition
            mid_state = int(D[i_b, i_first_input])

            for i_second_input in range(num_inputs):
                # compute end state after the second meta-stage transition
                end_state = int(D[mid_state, i_second_input])
                newW[i_b, end_state, :] += np.convolve(
                    W[i_first_input, i_b, :], W[i_second_input, mid_state, :]
                )

                # compute newD
                shift_amt = np.log2(num_inputs)
                concat_input = (int(i_first_input) << int(shift_amt)) | i_second_input
                newD[i_b, concat_input] = end_state

    layer_indices = np.arange(num_states)[:, np.newaxis]
    newW = np.moveaxis(
        newW[layer_indices, newD.astype(np.int32)], source=0, destination=1
    )

    return newW, newD
