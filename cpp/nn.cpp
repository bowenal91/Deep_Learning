#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "pybind11/operators.h"
#include "Tensor.cpp"

namespace py = pybind11;

PYBIND11_MODULE(nn, m) {
    py::class_<Tensor>(m, "Tensor")
        .def(py::init<>())
        .def(py::init< vector<int> >())
        .def("set_value", (void (Tensor::*) (const vector<int> &, double)) &Tensor::set_value, "Get by indices")
        .def("set_value", (void (Tensor::*) (const int, double)) &Tensor::set_value, "Get by absolute index")
        .def("get_value", (double (Tensor::*) (const vector<int> &)) &Tensor::get_value)
        .def("get_value", (double (Tensor::*) (const int)) &Tensor::get_value)
        .def("resize", &Tensor::resize)
        .def(py::self * py::self)
        .def(py::self ^ py::self)
        .def(py::self + py::self)
        .def(py::self - py::self)
        .def("subset_mult", &Tensor::subset_mult);

}

