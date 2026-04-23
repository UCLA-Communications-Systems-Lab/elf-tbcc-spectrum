#ifndef FEEDFORWARDTRELLIS_H
#define FEEDFORWARDTRELLIS_H

#include "types.h"
#include <vector>

class FeedForwardTrellis {
public:
  FeedForwardTrellis(const CodeInformation &code);

  std::vector<int> encode(const std::vector<int> &originalMessage) const;
  std::vector<int> encode_zt(const std::vector<int> &originalMessage) const;

  std::vector<std::vector<int>> getNextStates() const;
  std::vector<std::vector<int>> getOutputs() const;

  /* generator matrix for TBCC code */
  void computeGeneratorMatrix();

  int getNumInputSymbols() const;
  int getNumOutputSymbols() const;
  int getNumStates() const;
  int getV() const;
  int getN() const;

private:
  int kconv;
  int nconv;
  int v;
  int numInputSymbols;
  int numOutputSymbols;
  int numStates;
  std::vector<int> numerators;
  std::vector<std::vector<int>> nextStates;
  std::vector<std::vector<int>> outputs;
  std::vector<std::vector<int>> generatorMatrix;

  void computeNextStates();

  std::vector<int> dec2Bin(int decimal, int length);
  int bin2Dec(std::vector<int> binary);
};
#endif
