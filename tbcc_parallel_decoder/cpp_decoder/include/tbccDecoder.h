#ifndef TBCC_DECODER_H
#define TBCC_DECODER_H

#include "types.h"
#include <limits>    // <-- add this
#include <vector>    // <-- likely needed too for the vectors below
                     
class FeedForwardTrellis;
class MinHeap;

struct cell {
  int optimalFatherState = -1;
  int suboptimalFatherState = -1;
  float pathMetric = std::numeric_limits<float>::max();
  float suboptimalPathMetric = std::numeric_limits<float>::max();
  bool init = false;
};

class LowRateListDecoder {
public:
  LowRateListDecoder(const FeedForwardTrellis &FT, const CodeInformation &code,
                     int listSize);

  MessageInformation tbcc_decoding(const std::vector<float> &receivedMessage,
                                   const std::vector<int> &punctured_indices);

private:
  std::vector<std::vector<cell>>
  constructLowRateTrellis_Punctured(const std::vector<float> &receivedMessage,
                                    const std::vector<int> &punctured_indices);
  std::vector<int> pathToMessage(const std::vector<int> &path) const;
  std::vector<int> pathToCodeword(const std::vector<int> &path) const;

  const CodeInformation &code_;
  int numForwardPaths_;
  int listSize_;

  std::vector<std::vector<int>> nextStates_;
  std::vector<std::vector<int>> outputs_;
  int numStates_;
  int symbolLength_;
  int pathLength_;
};
#endif