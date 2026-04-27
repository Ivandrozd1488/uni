#include <pybind11/pybind11.h>

#include "unified_ml_version.hpp"

namespace py = pybind11;

PYBIND11_MODULE(unified_ml, m) {
  m.doc() = "Early-stage Python bootstrap bindings for unified_ml";
  m.attr("version") = py::str(UNIFIED_ML_VERSION_STRING);
  m.attr("api_stage") = py::str("bootstrap");

  m.def("version_string", []() { return std::string(UNIFIED_ML_VERSION_STRING); },
        "Return unified_ml version string.");
  m.def("bindings_stage", []() { return std::string("bootstrap"); },
        "Return the maturity stage of the Python bindings surface.");
}
