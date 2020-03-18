#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "Tensor.cpp"

namespace py = pybind11;

PYBIND11_MODULE(nn, m) {
    py::class_<Tensor>(m, "Tensor")
        .def(py::init< vector<int> >())
        .def("set_value", &Tensor::set_value)
        .def("get_value", &Tensor::get_value);
}

/*
PYBIND11_MODULE(example, m) {
    py::class_<Pet>(m, "Pet")
        .def(py::init<const std::string &>())
        .def("setName", &Pet::setName)
        .def("getName", &Pet::getName)
        .def("__repr__",
                [](const Pet &a) {
                    return "Pet named " + a.name + "";
                }
        );
}
*/
