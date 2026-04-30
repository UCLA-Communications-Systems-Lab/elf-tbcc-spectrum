#include "../include/feedForwardTrellis.h"
#include "../include/types.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // Allows automatic conversion of std::vector
#include <vector>

namespace py = pybind11;

PYBIND11_MODULE(tbcc_encoder, m) {
  // Bind the Struct
  py::class_<CodeInformation>(m, "CodeInformation")
      .def(py::init<int, int, int, int, int, int, std::vector<int>>())
      .def_readwrite("kconv", &CodeInformation::kconv)
      .def_readwrite("nconv", &CodeInformation::nconv)
      .def_readwrite("v", &CodeInformation::v)
      .def_readwrite("crcLen", &CodeInformation::crcLen)
      .def_readwrite("crc", &CodeInformation::crc)
      .def_readwrite("numInfoBits", &CodeInformation::numInfoBits)
      // Note: 'numerators' must be readonly because it's const in C++!
      .def_readonly("numerators", &CodeInformation::numerators);

  py::class_<FeedForwardTrellis>(m, "FeedForwardTrellis")
      .def(py::init<const CodeInformation &>())
      .def("getNextStates",
           &FeedForwardTrellis::getNextStates) // Bind this if Python needs the
                                               // data
      .def("getOutputs", &FeedForwardTrellis::getOutputs)
      .def("encode", &FeedForwardTrellis::encode);
}