#include "../include/crc.h"
#include <vector>

std::vector<int> CRC::remdr_slidingWindow(const std::vector<int> &dividend,
                                          const std::vector<int> &generator,
                                          bool raise_before_long_div) {
  /* Computes the remainder of dividend after dividing by generator using a
  sliding window approach. The length of generator is degree_of_generator + 1,
  e.g., if the generator in polynomial is x^2 + x + 1, then
  degree_of_generator=2. The length of the output is equal to the degree of the
  generator, i.e., always 1 less than the length of the generator.
  */
  unsigned int g_degree = (unsigned int)generator.size() - 1;
  unsigned int div_len = (unsigned int)dividend.size();
  std::vector<int> result;
  if (raise_before_long_div) {
    /* appending zeros to end, raising origin polynomial's power by degree of
     * CRC */
    result.reserve(div_len + g_degree);
  } else {
    result.reserve(div_len);
  }

  for (unsigned int i = 0; i < div_len; i++) {
    result[i] = dividend[i];
  }

  std::vector<int> remainder(g_degree, 0);
  if (raise_before_long_div) {
    /* long division */
    for (unsigned int i = 0; i < div_len; i++) {
      if (result[i] == 1) {
        for (unsigned int j = 0; j <= g_degree; j++) {
          result[i + j] = result[i + j] ^ generator[j];
        }
      }
    }
    /* record remainder */
    for (unsigned int i = 0; i < g_degree; i++) {
      remainder[i] = result[i + div_len];
    }

  } else {
    /* long division */
    for (unsigned int i = 0; i < div_len - g_degree; i++) {
      if (result[i] == 1) {
        for (unsigned int j = 0; j <= g_degree; j++) {
          result[i + j] = result[i + j] ^ generator[j];
        }
      }
    }
    /* record remainder */
    for (unsigned int i = 0; i < g_degree; i++) {
      remainder[i] = result[i + div_len - g_degree];
    }
  }

  return remainder;
}