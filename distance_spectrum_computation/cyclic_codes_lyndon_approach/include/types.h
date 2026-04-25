#ifndef MLA_TYPES_H
#define MLA_TYPES_H

#include <vector>

struct CodeInformation {
  int kconv;
  int nconv;
  int v;
  int crcLen;
  int crc;
  int numInfoBits;
  const std::vector<int> numerators;

  // The Constructor
  CodeInformation(int k, int n, int mem_v, int cLen, int cPoly, int infoBits,
                  std::vector<int> nums)
      : kconv(k), nconv(n), v(mem_v), crcLen(cLen), crc(cPoly),
        numInfoBits(infoBits),
        numerators(std::move(nums)) // Move the vector for efficiency
  {}
};

struct MessageInformation {
  MessageInformation() {
    message = std::vector<int>();
    codeword = std::vector<int>();
    path = std::vector<int>();
    listSize = -1;
    TBListSize = -1;
    listSizeExceeded = false;
    metric = -1.0;
  };
  std::vector<int> message;
  std::vector<int> codeword;
  std::vector<int> path;
  int listSize;
  int TBListSize;
  bool listSizeExceeded;
  double metric;
};

#endif