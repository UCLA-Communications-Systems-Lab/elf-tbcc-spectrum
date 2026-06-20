def divides_xN_minus_1(poly_str, N):
    """Checks if a binary polynomial string divides x^N - 1 over GF(2).

    Parameters:
    poly_str (str): Polynomial coefficients from HIGH to LOW order (e.g., '100101')
    N (int): The exponent in x^N - 1

    Returns:
    bool: True if it divides perfectly (remainder is 0), False otherwise.
    """
    # Convert the high-to-low bit string into a list of integers
    poly_bits = [int(bit) for bit in poly_str]

    # Clean up any potential leading zeros in the string input
    while len(poly_bits) > 0 and poly_bits[0] == 0:
        poly_bits = poly_bits[1:]

    if not poly_bits:
        raise ValueError("The divisor polynomial cannot be zero.")

    # High-to-low representation of x^N - 1 is a 1, followed by N-1 zeros, then a 1
    dividend = [1] + [0] * (N - 1) + [1]

    # Polynomial Long Division (XORing binary coefficients)
    for i in range(len(dividend) - len(poly_bits) + 1):
        if dividend[i] == 1:
            for j in range(len(poly_bits)):
                dividend[i + j] ^= poly_bits[j]

    return sum(dividend) == 0
