#include "../include/feedForwardTrellis.h"
#include "../include/mathUtils.h"

#include <cmath>
#include <cstddef>
#include <string>

FeedForwardTrellis::FeedForwardTrellis(const CodeInformation &code)
    : kconv(code.kconv), nconv(code.nconv), v(code.v),
      numerators(code.numerators),
      numInputSymbols(static_cast<int>(std::pow(2, code.kconv))),
      numOutputSymbols(static_cast<int>(std::pow(2, code.nconv))),
      numStates(static_cast<int>(std::pow(2, code.v))) {

  nextStates.assign(numStates, std::vector<int>(numInputSymbols));
  outputs.assign(numStates, std::vector<int>(numInputSymbols));

  computeNextStates();
}

void FeedForwardTrellis::computeNextStates() {
  /* convert octal generator polys to binary numerators */
  std::vector<std::vector<int>> bin_numerators(nconv, std::vector<int>(v + 1));
  for (int i = 0; i < nconv; i++) {
    /* Octal to decimal */
    int tempNum = numerators[i];
    int decIn = 0;
    std::string in = std::to_string(tempNum);
    for (int p = (in.length() - 1); p >= 0; p--)
      decIn += (int)(in[p] - '0') * pow(8, (in.length() - p - 1));

    /* Decimal to binary */
    for (int j = v; j >= 0; j--) {
      if (decIn % 2 == 0)
        bin_numerators[i][j] = 0;
      else
        bin_numerators[i][j] = 1;
      decIn = decIn / 2;
    }
  }

  /* calculate next states and outputs */
  for (int currentState = 0; currentState < numStates; currentState++) {
    for (int input = 0; input < numInputSymbols; input++) {

      // Setup the Register for Output Calculation
      // We need the binary vector to do the XOR sum based on the polynomials
      std::vector<int> mem_elements = dec2Bin(currentState, v);
      // Add the input bit to the end (representing the "newest" bit)
      mem_elements.push_back(input);

      // Output Computation
      std::vector<int> output(nconv, 0);
      for (int x_bit = 0; x_bit < nconv; x_bit++) {
        for (int m_bit = 0; m_bit < v + 1; m_bit++) {
          if (bin_numerators[x_bit][m_bit] == 1) {
            output[x_bit] ^= mem_elements[m_bit];
          }
        }
      }
      outputs[currentState][input] = bin2Dec(output);

      // Next State Computation
      // If State 0 + Input 1 should = State 1:
      // We shift the current state left and add the new input,
      // then mask it to the constraint length.
      int next = ((currentState << 1) | input) & (numStates - 1);
      nextStates[currentState][input] = next;
    }
  }
}

void FeedForwardTrellis::computeGeneratorMatrix() {
  /*
  creates the generator matrix. Each row is the codeword corresponds to a
  message with exactly one 1's. The ith row corresponds to a message with the
  ith position being a 1.
  */

  for (int i = 0; i < this->kconv; i++) {
    std::vector<int> m(this->kconv, 0);
    m[i] = 1;
    this->generatorMatrix.push_back(this->encode(m));
  }
}

// for zero-terminated message, start encoding at state-0
std::vector<int>
FeedForwardTrellis::encode_zt(const std::vector<int> &originalMessage) const {
  std::vector<int> output;
  int State = 0;
  for (size_t i = 0; i < originalMessage.size(); i += kconv) {
    int decimal = 0;
    for (int j = 0; j < kconv; j++) {
      decimal += (originalMessage[i + j] * pow(2, kconv - j - 1));
    }
    std::vector<int> outputBinary =
        MathUtils::toModulatedPoint(outputs[State][decimal], nconv);
    State = nextStates[State][decimal];
    for (int j = 0; j < nconv; j++) {
      output.push_back(outputBinary[j]);
    }
  }
  return output;
}

std::vector<int>
FeedForwardTrellis::encode(const std::vector<int> &originalMessage) const {
  // brute force approach, there is a better way to do this assuming
  // invertibility that allows us to precompute starting / ending states,
  // reducing complexity in each encoding from O(numStates) to O(2). revisit
  // when available

  for (int m = 0; m < numStates; m++) {
    std::vector<int> output;
    int State = m;
    for (size_t i = 0; i < originalMessage.size(); i += kconv) {
      int decimal = 0;
      for (int j = 0; j < kconv; j++) {
        decimal += (originalMessage[i + j] * pow(2, kconv - j - 1));
      }
      std::vector<int> outputBinary =
          MathUtils::toBinary(outputs[State][decimal], nconv);
      State = nextStates[State][decimal];
      for (int j = 0; j < nconv; j++) {
        output.push_back(outputBinary[j]);
      }
    }
    if (m == State) {
      return output;
    }
  }
  return originalMessage;
}

std::vector<int> FeedForwardTrellis::dec2Bin(int decimal, int length) {
  std::vector<int> binary(length);
  for (int j = (length - 1); j >= 0; j--) {
    if (decimal % 2 == 0)
      binary[j] = 0;
    else
      binary[j] = 1;
    decimal = decimal / 2;
  }
  return binary;
}

int FeedForwardTrellis::bin2Dec(std::vector<int> binary) {
  int decimal = 0;
  for (int i = (binary.size() - 1); i >= 0; i--) {
    decimal += (binary[i] * pow(2, (binary.size() - i - 1)));
  }
  return decimal;
}

std::vector<std::vector<int>> FeedForwardTrellis::getNextStates() const {
  return nextStates;
}

std::vector<std::vector<int>> FeedForwardTrellis::getOutputs() const {
  return outputs;
}

int FeedForwardTrellis::getNumInputSymbols() const { return numInputSymbols; }

int FeedForwardTrellis::getNumOutputSymbols() const { return numOutputSymbols; }

int FeedForwardTrellis::getNumStates() const { return numStates; }

int FeedForwardTrellis::getV() const { return v; }

int FeedForwardTrellis::getN() const { return nconv; }