#pragma once

#include <vector>
#include <cmath>

#include "data_structures/csr_matrix.hpp"

/**
* @brief This namespace contains required functions for
* the Probe a Load (PaL) partitioning algorithm. PaL partitioning
* is one of the algorithms that is used to solve
* mNC problem. Its dual solves the mLI problem
*/
#if defined(ENABLE_CPP_PARALLEL)
namespace sarma::probe_a_load {
#else
namespace sarma{
    namespace probe_a_load {
#endif

    /**
    * @brief Implements the Probe a Load algorithm.
    * This algorithm applies a greedy probe algorithm
    * on diagonal direction by running binary searches
    * over the possible range of maximum load targets.
    *
    * @param A Matrix
    * @param P number of parts
    * @tparam Ordinal data-type of indptr and indices structures in sarma::Matrix
    * @tparam Value data-type of data structure in sarma::Matrix
    * @return a cut vector
    */
    template <typename Ordinal, typename Value>
    std::vector<Ordinal> probe(const Matrix<Ordinal, Value> &A, const Value L, std::vector<Ordinal> *UB = nullptr) {
        const auto sps = A.get_sps();
        std::vector<Ordinal> p;
        if (UB)
            p.reserve(UB->size());
        p.push_back(0);
        while ((!UB && p.back() < sps.size()) || (UB && p.size() < UB->size())) {
            auto l = p.back(), r = (UB ? (*UB)[p.size()] : sps.size());
            while (l < r) {
                const auto m = (l + 1) + (r - (l + 1)) / 2;
                p.push_back(m);
                const auto CL = std::max(sps.compute_maxload(p, p, p.size() - 1, true),
                                         sps.compute_maxload(p, p, p.size() - 1, false));
                if (CL <= L)
                    l = m;
                else
                    r = m - 1;
                p.pop_back();
            }
            p.push_back(l);
            if (p.back() == p[p.size() - 2]) {
                while(UB && p.size() < UB->size())
                    p.push_back(p.back());
                break;
            }
        }
        return p;
    }

    /**
    * @brief Implements the dual of the Probe a Load (PaL) algorithm
    * using Bound a Load procedure and applying PaL as the mNC algorithm.
    * This algorithm applies a greedy probe algorithm
    * on diagonal direction by running binary searches
    * over the possible range of maximum load targets.
    *
    * @param A Matrix
    * @param P number of parts
    * @return a cut vector
    */
    template <typename Ordinal, typename Value>
    auto partition(const Matrix<Ordinal, Value> &A, const Ordinal P) {
        const auto sps = A.get_sps();
        const auto N = sps.size();
        const auto M = sps.total_load();
        auto l = (M - 1 + P * P) / P / P;
        auto r = std::min(M, (M + P - 1) / P + N);
        std::vector<Ordinal> UB(P + 1, N);
        while (l + .5 < r) {
            const auto m = l + (r - l) / 2;
            const auto p = probe(A, m, &UB);
            if (p.back() < N)
                l = m + 1;
            else {
                r = m;
                UB = p;
            }
        }
        return probe(A, r, &UB);
    }
}
#if !defined(ENABLE_CPP_PARALLEL)
} // nested namespace
#endif