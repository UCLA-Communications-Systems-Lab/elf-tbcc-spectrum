#include "../include/minHeap.h"

MinHeap::MinHeap() {
  // constructor if necessary
}

int MinHeap::parentIndex(int index) { return (index - 1) / 2; }
int MinHeap::leftChildIndex(int index) { return (2 * index + 1); }
int MinHeap::rightChildIndex(int index) { return (2 * index + 2); }

void MinHeap::insert(DetourObject detour) {
  detourList.push_back(detour);
  int index = detourList.size() - 1;
  while (index > 0 && detourList[parentIndex(index)] > detourList[index]) {
    std::swap(detourList[parentIndex(index)], detourList[index]);
    index = parentIndex(index);
  }
}

DetourObject MinHeap::pop() {
  DetourObject detour = detourList[0];
  detourList[0] = detourList[detourList.size() - 1];
  detourList.pop_back();
  reHeap(0);

  return detour;
}

DetourObject MinHeap::top() { return detourList[0]; }

void MinHeap::reHeap(int index) {
  unsigned int leftIndex = leftChildIndex(index);
  unsigned int rightIndex = rightChildIndex(index);
  int minDetourIndex = index;
  if (leftIndex < detourList.size() &&
      detourList[leftIndex] < detourList[minDetourIndex])
    minDetourIndex = leftIndex;
  if (rightIndex < detourList.size() &&
      detourList[rightIndex] < detourList[minDetourIndex])
    minDetourIndex = rightIndex;

  if (minDetourIndex != index) {
    std::swap(detourList[index], detourList[minDetourIndex]);
    reHeap(minDetourIndex);
  }
}

int MinHeap::size() { return detourList.size(); }