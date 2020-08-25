#pragma once

#include <algorithm>
#include <vector>
#include <utility>

#include "data_structures/csr_matrix.hpp"

/**
* @brief This namespace contains required functions for
* the Ordered Probe a Load algorithm. This algorithm doesn't
* require sparse-prefix-sum data-structure.
* Note that, for now, this implementation is sequential.
**/
#if defined(ENABLE_CPP_PARALLEL)
namespace sarma::ordered_probe_a_load {
#else
namespace sarma{
    namespace ordered_probe_a_load {
#endif
    template <typename Int, typename Value>
    std::vector<Int> probe(const std::vector<Int> &indptr, const std::vector<std::pair<Int, bool>> &indices, const std::vector<Value> &data, const Value L, const Int P = std::numeric_limits<Int>::max()) {
        std::vector<Int> p;
        p.push_back(0);
        std::vector<Value> L1, L2;
        for (Int k = 0; k < indptr.size() - 1 && p.size() <= P;) {
            L1.push_back(0);
            L2.push_back(0);
            std::fill(L1.begin(), L1.end(), 0);
            std::fill(L2.begin(), L2.end(), 0);
            Value mx = 0;
            for (; k < indptr.size() - 1; k++) {
                for (auto l = indptr[k]; l < indptr[k + 1]; l++) {
#if defined(ENABLE_CPP_PARALLEL)
                    const auto [i, b] = indices[l];
#else
                    const auto i = indices[l].first;
                    const auto b = indices[l].second;
#endif
                    const auto idx = utils::lowerbound_index(p, i);
                    auto &Lr = b || idx == p.size() - 1 ? L1 : L2;
                    mx = std::max(mx, Lr[idx] += data.empty() ? 1 : data[l]);
                    if (mx > L)
                        break;
                }
                if (mx > L)
                    break;
            }
            p.push_back(k);
            if (p.back() == p[p.size() - 2]) {
                while(p.size() < P + 1)
                    p.push_back(p.back());
                break;
            }
        }
        return p;
    }

    template <typename Int, typename Value>
    auto transform(const Matrix<Int, Value> &A) {
        const auto N = std::max(A.N(), A.M);
        
        std::vector<Int> indptr(1 + N);
        for (Int i = 0; i < A.N(); i++)
            for (auto j = A.indptr[i]; j < A.indptr[i + 1]; j++)
                indptr[1 + std::max(i, A.indices[j])]++;

        std::partial_sum(indptr.begin(), indptr.end(), indptr.begin());

        std::vector<std::pair<Int, bool>> indices(A.NNZ());
        std::vector<Value> data;
        const auto use_data = A.is_using_data();
        if (use_data)
            data.resize(A.NNZ());
        auto f = indptr;

        for (Int i = 0; i < A.N(); i++)
            for (auto j = A.indptr[i]; j < A.indptr[i + 1]; j++) {
                const auto b = i < A.indices[j];
                const auto k = f[b ? A.indices[j] : i]++;
                indices[k] = {b ? i : A.indices[j], b};
                if (use_data)
                    data[k] = A.get_data()[k];
            }
        
        return std::make_tuple(indptr, indices, data);
    }

    template <typename Int, typename Value>
    std::vector<Int> probe(const Matrix<Int, Value> &A, const Value L) {
#if defined(ENABLE_CPP_PARALLEL)
        const auto [indptr, indices, data] = transform(A);
#else
        const auto terms = transform(A);
        const auto indptr = std::get<0>(terms);
        const auto indices = std::get<1>(terms);
        const auto data = std::get<2>(terms);
#endif
        return probe(indptr, indices, data, L);
    }

    /**
    * @brief Implements the Probe a Load algorithm.
    * This algorithm applies a greedy probe algorithm
    * on diagonal direction by running binary searches
    * over the possible range of maximum load targets.
    *
    * @param A Matrix
    * @param P number of parts
    * @return a cut vector
    */
    template <typename Int, typename Value>
    std::vector<Int> partition(const Matrix<Int, Value> &A, const Int P) {
        const auto N = std::max(A.N(), A.M);
        const auto M = A.total_load();

#if defined(ENABLE_CPP_PARALLEL)
        const auto [indptr, indices, data] = transform(A);
#else
        const auto terms = transform(A);
        const auto indptr = std::get<0>(terms);
        const auto indices = std::get<1>(terms);
        const auto data = std::get<2>(terms);
#endif


        auto l = (M - 1 + P * P) / P / P;
        auto r = std::min(M, (M + P - 1) / P + N);
        while (l + .5 < r) {
            const auto m = l + (r - l) / 2;
            const auto p = probe(indptr, indices, data, m, P);
            if (p.back() < N)
                l = m + 1;
            else
                r = m;
        }
        return probe(indptr, indices, data, r, P);
    }
}
#if !defined(ENABLE_CPP_PARALLEL)
} // nested namespace
#endif