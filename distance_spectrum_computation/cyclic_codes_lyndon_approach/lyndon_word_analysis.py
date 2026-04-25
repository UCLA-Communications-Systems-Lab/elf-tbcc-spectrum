import numpy as np


def generate_all_sequences(length):
    """
    Generates all 2^length possible binary sequences for a given bit length.
    Returns a 2D numpy array of shape (2^length, length).
    """
    # Total number of combinations (2^N)
    num_sequences = 1 << length

    # Generate numbers 0 to (2^length - 1) as a column vector
    # We use int64 to support lengths up to 63 bits
    numbers = np.arange(num_sequences, dtype=np.int64).reshape(-1, 1)

    # Create an array of bit shifts: [length-1, length-2, ..., 0]
    shifts = np.arange(length - 1, -1, -1)

    # Use bitwise AND to check if each bit is set, then convert to int32 (0 or 1)
    all_sequences = (numbers & (1 << shifts)) > 0
    return all_sequences.astype(np.int32)


def find_smallest_repeat(arr):
    n = len(arr)
    # 1. Find all divisors of the length (15 -> 1, 3, 5, 15)
    for d in range(1, n + 1):
        if n % d == 0:
            # 2. Extract the potential base pattern
            base = arr[:d]
            # 3. Tile it to the original length and compare
            if np.array_equal(np.tile(base, n // d), arr):
                return d
    return n


def get_canonical_root(items):
    """
    Finds the lexicographically smallest cyclic shift of a sequence.
    Works for strings, numbers, or bits.
    """
    if not items:
        return items

    n = len(items)
    # Generate all cyclic rotations of the sequence
    rotations = [items[i:] + items[:i] for i in range(n)]

    # Return the "smallest" rotation as a tuple (so it can be put in a set)
    return min(rotations)


def count_root_codewords_from_grid(file_path):
    unique_roots = set()

    try:
        with open(file_path, "r") as file:
            for line in file:
                # 1. Split line by commas to get individual bits
                # 2. Strip whitespace from each bit
                bits = list(line.strip())

                # If the line wasn't empty
                if bits:
                    # 3. Convert bits to a tuple (a 'fixed' list)
                    codeword = tuple(bits)

                    # 4. Find the 'root' and add to our unique collection
                    root = get_canonical_root(codeword)
                    unique_roots.add(root)

        return len(unique_roots), unique_roots

    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return 0
