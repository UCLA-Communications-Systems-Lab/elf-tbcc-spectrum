#ifndef CRC_H
#define CRC_H
#include <vector>
struct CodeInformation;

class CRC {
public:
  CRC(const CodeInformation &code);
  std::vector<int> remdr_slidingWindow(const std::vector<int> &dividend,
                                       const std::vector<int> &generator,
                                       bool raise_before_long_div = false);
};

#endif