#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

namespace py = pybind11;

#include <utility>
#include <vector>

#include "tools/utils.hpp"
#include "algorithms/algorithm.hpp"
#include "data_structures/csr_matrix.hpp"

using Int = unsigned;
using Value = Int;
using Real = double;

using namespace sarma;

struct sparse_prefix_sum_wrapper : public sparse_prefix_sum<Int, Value> {
    sparse_prefix_sum_wrapper(std::vector<Int> &&indptr, std::vector<Int> &&indices, std::vector<Value> &&data, const Int M) : sparse_prefix_sum(Matrix<Int, Value>(std::move(indptr), std::move(indices), std::move(data), M)) {}

    sparse_prefix_sum_wrapper(const std::string s) : sparse_prefix_sum(Matrix<Int, Value>(s)) {}
};

PYBIND11_MODULE(_sarma, m) {
    py::class_<sparse_prefix_sum_wrapper>(m, "sps")
        .def(py::init<std::vector<Int> &&, std::vector<Int> &&, std::vector<Value> &&, Int>())
        .def(py::init<const std::string>())
        .def("size", &sparse_prefix_sum_wrapper::size)
        .def("query", (Value (sparse_prefix_sum_wrapper::*)(const std::pair<Int, Int>, const std::pair<Int, Int>) const)&sparse_prefix_sum_wrapper::query)
        .def("loads", &sparse_prefix_sum_wrapper::compute_loads)
        .def("max_load", (Value (sparse_prefix_sum_wrapper::*)(const std::vector<Int> &, const std::vector<Int> &) const)&sparse_prefix_sum_wrapper::compute_maxload);
    
    using namespace pybind11::literals;

    m.def("nic", [] (std::vector<Int> &&indptr, std::vector<Int> &&indices, std::vector<Value> &&data, const Int M, const Int P) {
        Matrix<Int, Value> A(std::move(indptr), std::move(indices), std::move(data), M);
        return nicol2d::partition(A, P, P);
    }, "Symmetric spatial partitioning with probe target load");
    
    m.def("pal", [] (std::vector<Int> &&indptr, std::vector<Int> &&indices, std::vector<Value> &&data, const Int M, const Int P) {
        Matrix<Int, Value> A(std::move(indptr), std::move(indices), std::move(data), M);
        return probe_a_load::partition(A, P);
    }, "Symmetric spatial partitioning with probe target load");

    m.def("rac", [] (std::vector<Int> &&indptr, std::vector<Int> &&indices, std::vector<Value> &&data, const Int M, const Int P) {
        Matrix<Int, Value> A(std::move(indptr), std::move(indices), std::move(data), M);
        return refine_a_cut::partition(A, P);
    }, "Symmetric spatial partitioning with probe target load");

    m.def("pbi", [] (std::vector<Int> &&indptr, std::vector<Int> &&indices, std::vector<Value> &&data, const Int M, const Int P) {
        Matrix<Int, Value> A(std::move(indptr), std::move(indices), std::move(data), M);
        return sarma::refine_a_cut::partition(A, P, (Value) 0, false);
    }, "Symmetric spatial partitioning with probe target load");

    m.def("uni", [] (std::vector<Int> &&indptr, std::vector<Int> &&/*indices*/, std::vector<Value> &&/*data*/, const Int /*M*/, const Int P) {
        return uniform::partition((Int)indptr.size() - 1, P);
    }, "Symmetric spatial partitioning with probe target load");
}
