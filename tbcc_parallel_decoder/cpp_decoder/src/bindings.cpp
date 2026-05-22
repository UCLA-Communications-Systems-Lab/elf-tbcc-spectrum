#include "../include/feedForwardTrellis.h"
#include "../include/tbccDecoder.h"
#include "../include/types.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // Allows automatic conversion of std::vector
#include <vector>

namespace py = pybind11;

PYBIND11_MODULE(cpp_tbcc_decoder, m) {
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

  py::class_<MessageInformation>(m, "MessageInformation")
      .def(py::init<>())
      .def_readwrite("message", &MessageInformation::message)
      .def_readwrite("codeword", &MessageInformation::codeword)
      .def_readwrite("path", &MessageInformation::path)
      .def_readwrite("listSize", &MessageInformation::listSize)
      .def_readwrite("TBListSize", &MessageInformation::TBListSize)
      .def_readwrite("listSizeExceeded", &MessageInformation::listSizeExceeded)
      .def_readwrite("metric", &MessageInformation::metric);

  // Bind the Class
  py::class_<LowRateListDecoder>(m, "LowRateListDecoder")
      .def(py::init<const FeedForwardTrellis &, const CodeInformation &, int>(),
           py::arg("trellis"), py::arg("code"),
           py::arg("list_size") = 8) // Optional: default list size
      .def("tbcc_decode", &LowRateListDecoder::tbcc_decoding);

  py::class_<FeedForwardTrellis>(m, "FeedForwardTrellis")
      .def(py::init<const CodeInformation &>())
      .def("getNextStates",
           &FeedForwardTrellis::getNextStates) // Bind this if Python needs the
                                               // data
      .def("getOutputs", &FeedForwardTrellis::getOutputs);
}