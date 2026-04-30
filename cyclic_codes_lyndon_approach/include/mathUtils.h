#ifndef MATH_UTILS_H
#define MATH_UTILS_H

#include <vector>

namespace MathUtils {
// converts decimal input to binary output, with a given number of bits
// since we need to keep track of leading zeros
inline std::vector<int> toBinary(int input, int bit_number) {
  std::vector<int> output;
  output.assign(bit_number, -1);
  for (int i = bit_number - 1; i >= 0; i--) {
    int k = input >> i;
    if (k & 1)
      output[bit_number - 1 - i] = 1;
    else
      output[bit_number - 1 - i] = 0;
  }
  return output;
}

// converts decimal output to n-bit BPSK
inline std::vector<int> toModulatedPoint(int output, int n) {
  std::vector<int> bin_output = toBinary(output, n);
  for (int i = 0; i < n; i++) {
    bin_output[i] = -2 * bin_output[i] + 1;
  }
  return bin_output;
}
} // namespace MathUtils

#endif