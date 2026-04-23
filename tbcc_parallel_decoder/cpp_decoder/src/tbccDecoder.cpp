#include "../include/tbccDecoder.h"
#include "../include/feedForwardTrellis.h"
#include "../include/mathUtils.h"
#include "../include/minHeap.h"
#include <vector>
#include <cmath>

LowRateListDecoder::LowRateListDecoder(const FeedForwardTrellis &FT,
                                       const CodeInformation &code,
                                       int listSize)
    : code_(code), listSize_(listSize), numStates_(FT.getNumStates()),
      symbolLength_(FT.getN()) {
  nextStates_ = FT.getNextStates();
  outputs_ = FT.getOutputs();

  if (!nextStates_.empty()) {
    numForwardPaths_ = nextStates_[0].size();
  } else {
    numForwardPaths_ = 0;
  }
}

std::vector<std::vector<cell>>
LowRateListDecoder::constructLowRateTrellis_Punctured(
    const std::vector<float> &receivedMessage,
    const std::vector<int> &punctured_indices) {
  /* Constructs a trellis for a low rate code, with puncturing
          Args:
                  receivedMessage (std::vector<float>): the received message
                  punctured_indices (std::vector<int>): the indices of the
     punctured bits

          Returns:
                  std::vector<std::vector<cell>>: the trellis
  */

  /* ---- Code Begins ---- */
  std::vector<std::vector<cell>> trellisInfo;
  pathLength_ = (receivedMessage.size() / symbolLength_) + 1;

  trellisInfo = std::vector<std::vector<cell>>(numStates_,
                                               std::vector<cell>(pathLength_));

  // initializes all the valid starting states
  for (int i = 0; i < numStates_; i++) {
    trellisInfo[i][0].pathMetric = 0;
    trellisInfo[i][0].init = true;
  }

  // building the trellis
  for (int stage = 0; stage < pathLength_ - 1; stage++) {
    for (int currentState = 0; currentState < numStates_; currentState++) {
      // if the state / stage is invalid, we move on
      if (!trellisInfo[currentState][stage].init)
        continue;

      // otherwise, we compute the relevent information
      for (int forwardPathIndex = 0; forwardPathIndex < numForwardPaths_;
           forwardPathIndex++) {
        // since our transitions correspond to symbols, the forwardPathIndex has
        // no correlation beyond indexing the forward path

        int nextState = nextStates_[currentState][forwardPathIndex];

        // if the nextState is invalid, we move on
        if (nextState < 0)
          continue;

        float branchMetric = 0;
        std::vector<int> output_point = MathUtils::toModulatedPoint(
            outputs_[currentState][forwardPathIndex], symbolLength_);

        for (int i = 0; i < symbolLength_; i++) {
          if (std::find(punctured_indices.begin(), punctured_indices.end(),
                        symbolLength_ * stage + i) != punctured_indices.end()) {
            branchMetric += 0;
          } else {
            branchMetric +=
                std::pow(receivedMessage[symbolLength_ * stage + i] -
                             (float)output_point[i],
                         2);
          }
        }

        float totalPathMetric =
            branchMetric + trellisInfo[currentState][stage].pathMetric;

        // dealing with cases of uninitialized states, when the transition
        // becomes the optimal father state, and suboptimal father state, in
        // order
        if (!trellisInfo[nextState][stage + 1].init) {
          trellisInfo[nextState][stage + 1].pathMetric = totalPathMetric;
          trellisInfo[nextState][stage + 1].optimalFatherState = currentState;
          trellisInfo[nextState][stage + 1].init = true;
        } else if (trellisInfo[nextState][stage + 1].pathMetric >
                   totalPathMetric) {
          trellisInfo[nextState][stage + 1].suboptimalPathMetric =
              trellisInfo[nextState][stage + 1].pathMetric;
          trellisInfo[nextState][stage + 1].suboptimalFatherState =
              trellisInfo[nextState][stage + 1].optimalFatherState;
          trellisInfo[nextState][stage + 1].pathMetric = totalPathMetric;
          trellisInfo[nextState][stage + 1].optimalFatherState = currentState;
        } else {
          trellisInfo[nextState][stage + 1].suboptimalPathMetric =
              totalPathMetric;
          trellisInfo[nextState][stage + 1].suboptimalFatherState =
              currentState;
        }
      }
    }
  }
  return trellisInfo;
}

MessageInformation
LowRateListDecoder::tbcc_decoding(const std::vector<float> &receivedMessage,
                                  const std::vector<int> &punctured_indices) {
  // trellisInfo is indexed [state][stage]
  std::vector<std::vector<cell>> trellisInfo;
  trellisInfo =
      constructLowRateTrellis_Punctured(receivedMessage, punctured_indices);

  // start search
  MessageInformation output;
  MinHeap detourTree;
  std::vector<std::vector<int>> previousPaths;

  // create nodes for each valid ending state with no detours
  // std::cout<< "end path metrics:" <<std::endl;
  for (int i = 0; i < numStates_; i++) {
    DetourObject detour;
    detour.startingState = i;
    detour.pathMetric = trellisInfo[i][pathLength_ - 1].pathMetric;
    detourTree.insert(detour);
  }

  int numPathsSearched = 0;
  int TBPathsSearched = 0;

  while (numPathsSearched < listSize_) {
    DetourObject detour = detourTree.pop();
    std::vector<int> path(pathLength_);

    int newTracebackStage = pathLength_ - 1;
    float forwardPartialPathMetric = 0;
    int currentState = detour.startingState;

    // if we are taking a detour from a previous path, we skip backwards to the
    // point where we take the detour from the previous path
    if (detour.originalPathIndex != -1) {
      forwardPartialPathMetric = detour.forwardPathMetric;
      newTracebackStage = detour.detourStage;

      // while we only need to copy the path from the detour to the end, this
      // simplifies things, and we'll write over the earlier data in any case
      path = previousPaths[detour.originalPathIndex];
      currentState = path[newTracebackStage];

      float suboptimalPathMetric =
          trellisInfo[currentState][newTracebackStage].suboptimalPathMetric;

      currentState =
          trellisInfo[currentState][newTracebackStage].suboptimalFatherState;
      newTracebackStage--;

      float prevPathMetric =
          trellisInfo[currentState][newTracebackStage].pathMetric;

      forwardPartialPathMetric += suboptimalPathMetric - prevPathMetric;
    }
    path[newTracebackStage] = currentState;

    // actually tracing back
    for (int stage = newTracebackStage; stage > 0; stage--) {
      float suboptimalPathMetric =
          trellisInfo[currentState][stage].suboptimalPathMetric;
      float currPathMetric = trellisInfo[currentState][stage].pathMetric;

      // if there is a detour we add to the detourTree
      if (trellisInfo[currentState][stage].suboptimalFatherState != -1) {
        DetourObject localDetour;
        localDetour.detourStage = stage;
        localDetour.originalPathIndex = numPathsSearched;
        localDetour.pathMetric =
            suboptimalPathMetric + forwardPartialPathMetric;
        localDetour.forwardPathMetric = forwardPartialPathMetric;
        localDetour.startingState = detour.startingState;
        detourTree.insert(localDetour);
      }
      currentState = trellisInfo[currentState][stage].optimalFatherState;
      float prevPathMetric = trellisInfo[currentState][stage - 1].pathMetric;
      forwardPartialPathMetric += currPathMetric - prevPathMetric;
      path[stage - 1] = currentState;
    }

    previousPaths.push_back(path);

    std::vector<int> message = pathToMessage(path);
    std::vector<int> codeword = pathToCodeword(path);

    // std::cout << "forwardPartialPathMetric: " << forwardPartialPathMetric <<
    // std::endl;

    // one trellis decoding requires both a tb and crc check
    if (path[0] == path[pathLength_ - 1] && numPathsSearched <= listSize_) {
      output.message = message;
      output.codeword = codeword;
      output.path = path;
      output.listSize = numPathsSearched + 1;
      output.metric = forwardPartialPathMetric;
      output.TBListSize = TBPathsSearched + 1;
      return output;
    }

    numPathsSearched++;
    if (path[0] == path[pathLength_ - 1])
      TBPathsSearched++;
  } // while(numPathsSearched < this->listSize)

  output.listSizeExceeded = true;
  return output;
}

// converts a path through the tb trellis to the binary message it corresponds
// with
std::vector<int>
LowRateListDecoder::pathToMessage(const std::vector<int> &path) const {
  std::vector<int> message;
  for (size_t pathIndex = 0; pathIndex < path.size() - 1; pathIndex++) {
    for (int forwardPath = 0; forwardPath < numForwardPaths_; forwardPath++) {
      if (nextStates_[path[pathIndex]][forwardPath] == path[pathIndex + 1])
        message.push_back(forwardPath);
    }
  }
  return message;
}

// converts a path through the tb trellis to the BPSK it corresponds with
// currently does NOT puncture the codeword
std::vector<int>
LowRateListDecoder::pathToCodeword(const std::vector<int> &path) const {
  std::vector<int> nopunc_codeword;
  for (size_t pathIndex = 0; pathIndex < path.size() - 1; pathIndex++) {
    for (int forwardPath = 0; forwardPath < numForwardPaths_; forwardPath++) {
      if (nextStates_[path[pathIndex]][forwardPath] == path[pathIndex + 1]) {
        std::vector<int> output_bin = MathUtils::toBinary(
            outputs_[path[pathIndex]][forwardPath], symbolLength_);
        for (int outbit = 0; outbit < symbolLength_; outbit++) {
          nopunc_codeword.push_back(-2 * output_bin[outbit] + 1);
        }
      }
    }
  }

  return nopunc_codeword;
}